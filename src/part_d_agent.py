"""Part D, tasks 5-8: refund agent with five tools, a ReAct loop over mock_llm (AGENT), three safety limits
with graceful human hand-off, and checkpoint/resume after a simulated crash.

Run:  python src/part_d_agent.py
Writes outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and state/<run_id>.json checkpoints.
"""
import glob
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

from common import DB_PATH, OUT, RAW, STATE, header
import mock_llm
from part_d_rag import retrieve

RUNS_FILE = f"{OUT}/agent_runs.jsonl"
LEDGER_FILE = f"{OUT}/refund_ledger.jsonl"
AUTO_APPROVE_LIMIT = 1000.0
LATE_MINUTES = 45

SYSTEM_PROMPT = ("You are the Instamart refund AGENT. Decide refund requests ONLY by calling tools: look the order up "
                 "with get_order, check the rules with search_policy, compute the amount with calculate_refund, then "
                 "issue_refund (INR 1000 or less) or escalate_to_human (above INR 1000). Finish with one line that "
                 "starts with REFUND, ESCALATE, DENIED or NOT_FOUND followed by a colon.")

_CTX = {"run_id": None, "scenario_id": None}     # who is calling the ledger tools


# ---------------------------------------------------------------- task 5: tools
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, deliv = row
    if deliv is not None and float(deliv).is_integer():
        deliv = int(deliv)
    return {"order_id": oid, "status": status, "order_amount": round(float(amount), 2), "delivery_minutes": deliv}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """100% of order_amount if delivered in more than 45 minutes, otherwise 0 (cancelled -> no delivery -> 0)."""
    if delivery_minutes is None or float(delivery_minutes) <= LATE_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _ledger(tool, order_id, refund_inr):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
             "order_id": order_id, "refund_inr": round(float(refund_inr), 2),
             "run_id": _CTX["run_id"], "scenario_id": _CTX["scenario_id"]}
    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    if float(refund_inr) > AUTO_APPROVE_LIMIT:               # defence in depth: never auto-pay above the limit
        raise ValueError("refunds above INR 1000 must be escalated to a human")
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": round(float(refund_inr), 2)}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": round(float(refund_inr), 2)}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order in data/instamart.db and return its status, amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "e.g. ORD-100123"}},
                    "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the three most relevant policy chunks for a query (TF-IDF).",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Apply the refund rule: full order amount if delivered in more than 45 minutes, else 0.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": ["number", "null"]}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Pay an approved refund of INR 1000 or less and record it in the refund ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
assert [t["name"] for t in TOOLS_SPEC] == list(TOOL_MAP)


# ---------------------------------------------------------------- task 8: checkpoints
class SimulatedCrash(RuntimeError):
    pass


def save_checkpoint(st):
    path = os.path.join(STATE, f"{st['run_id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)
    os.replace(tmp, path)                                     # atomic: never a half-written checkpoint


def load_checkpoint(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- tasks 6-7: ReAct loop with limits
def _handoff(st, reason):
    tools = ", ".join(s["tool"] for s in st["steps"]) or "none"
    last = json.dumps(st["steps"][-1]["observation"])[:160] if st["steps"] else "none"
    return (f"HANDOFF to human agent: run {st['run_id']} ({st['scenario_id']}) stopped - {reason}. "
            f"Request: '{st['messages'][1]['content']}'. Model calls: {st['llm_calls']}, tokens: {st['tokens_used']}. "
            f"Tools executed: {tools}. Last observation: {last}.")


def _finish(st, status, final_answer):
    st["status"], st["final_answer"] = status, final_answer
    refund = 0.0
    for s in st["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            refund = s["observation"]["refund_inr"]
    st["refund_inr"] = refund
    save_checkpoint(st)
    return {k: st[k] for k in ("run_id", "scenario_id", "status", "final_answer", "refund_inr",
                               "llm_calls", "tokens_used", "steps", "tools_executed")}


def _loop(st, crash_after_tools=None):
    _CTX.update(run_id=st["run_id"], scenario_id=st["scenario_id"])
    start = time.monotonic()                                  # the clock of this (possibly resumed) run
    while True:
        if time.monotonic() - start >= st["time_limit_seconds"]:              # (c) wall-clock limit
            return _finish(st, "FALLBACK_TIMEOUT", _handoff(st, f"time limit of {st['time_limit_seconds']} s reached"))
        if st["llm_calls"] >= st["max_iterations"]:                            # (a) iteration limit
            return _finish(st, "FALLBACK_MAX_ITERATIONS",
                           _handoff(st, f"no final answer after {st['max_iterations']} model calls"))
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:                             # (b) token limit, tool NOT executed
            return _finish(st, "FALLBACK_TOKEN_LIMIT",
                           _handoff(st, f"token limit {st['token_limit']} exceeded ({st['tokens_used']} used)"))
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                 # final answer
            content = msg.content or ""
            status = content.split(":", 1)[0].strip() if ":" in content else "UNKNOWN"
            return _finish(st, status, content)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        fn = TOOL_MAP.get(tc.name)
        try:
            obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
        except Exception as e:                                                 # tool failure becomes an observation
            obs = {"error": str(e)}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["step"] += 1
        st["steps"].append({"step": st["step"], "tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        save_checkpoint(st)                                   # checkpoint after every observation
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name}) in {st['run_id']}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
          "scenario_id": scenario_id, "status": "RUNNING", "final_answer": None,
          "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [], "tools_executed": [],
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tools)


def resume_agent(run_id):
    st = load_checkpoint(run_id)
    if st["status"] != "RUNNING":
        raise ValueError(f"run {run_id} already finished with {st['status']}")
    print(f"  resuming {run_id} from step {st['step']} (tools so far: {st['tools_executed']}, "
          f"{st['llm_calls']} model calls, {st['tokens_used']} tokens)")
    return _loop(st)


def log_run(res, run_type):
    line = {"run_type": run_type, "run_id": res["run_id"], "scenario_id": res["scenario_id"], "status": res["status"],
            "final_answer": res["final_answer"], "refund_inr": res["refund_inr"], "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": [{"step": s["step"], "tool": s["tool"], "arguments": s["arguments"], "observation": s["observation"]}
                      for s in res["steps"]]}
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def main():
    for p in (RUNS_FILE, LEDGER_FILE):                        # fresh outputs on every reproduce run
        open(p, "w").close()
    for p in glob.glob(os.path.join(STATE, "*.json")):
        os.remove(p)
    sc = pd.read_csv(f"{RAW}/agent_scenarios.csv", keep_default_na=False)

    header("PART D - TASK 5: TOOLS")
    for t in TOOLS_SPEC:
        print(f"  {t['name']:<18} {t['description']}")
    print("  example get_order:", get_order(sc.loc[0, "message"].split()[2]))
    print("  example get_order(ORD-999999):", get_order("ORD-999999"))

    header("PART D - TASK 6: SEVEN SCENARIOS (max_iterations=6, token_limit=6000, time_limit_seconds=10)")
    table = []
    for r in sc.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id, run_id=r.scenario_id)
        log_run(res, "scenario")
        ok = (res["status"] == r.expected_outcome or
              (r.expected_outcome == "FALLBACK" and res["status"].startswith("FALLBACK"))) \
            and abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01
        table.append({"scenario": r.scenario_id, "expected": r.expected_outcome, "status": res["status"],
                      "refund_inr": f"{res['refund_inr']:.2f}", "expected_inr": r.expected_refund_inr,
                      "model_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                      "tools": " > ".join(res["tools_executed"]), "match": ok})
    t = pd.DataFrame(table)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(t.to_string(index=False))
    assert t["match"].all()
    print("\nfinal answers:")
    for line in open(RUNS_FILE, encoding="utf-8"):
        d = json.loads(line)
        print(f"  {d['scenario_id']}: {d['final_answer'][:150]}")

    header("PART D - TASK 7: STOP SAFELY")
    s1 = sc.iloc[0]
    tok = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-200")
    log_run(tok, "demo_token_limit_200")
    print(f"S1 with token_limit=200 -> {tok['status']} after {tok['llm_calls']} model calls, {tok['tokens_used']} tokens; "
          f"tools executed {tok['tools_executed']}")
    print("  final_answer:", tok["final_answer"])
    tmo = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-0s")
    log_run(tmo, "demo_time_limit_0")
    print(f"S1 with time_limit_seconds=0 -> {tmo['status']} after {tmo['llm_calls']} model calls")
    print("  final_answer:", tmo["final_answer"])
    s7 = json.loads(open(RUNS_FILE, encoding="utf-8").readlines()[6])
    print(f"S7 with defaults -> {s7['status']} after {s7['llm_calls']} model calls and {s7['tokens_used']} tokens")
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tmo["status"] == "FALLBACK_TIMEOUT"
    assert s7["status"] == "FALLBACK_MAX_ITERATIONS" and s7["llm_calls"] == 6

    header("PART D - TASK 8: CRASH AND RESUME (S1)")
    run_id = "S1-crash-resume"
    try:
        run_agent(s1.message, "S1", run_id=run_id, crash_after_tools=2)
    except SimulatedCrash as e:
        print(f"  {e}")
    cp = load_checkpoint(run_id)
    print(f"  checkpoint {STATE}/{run_id}.json: step {cp['step']}, tools {cp['tools_executed']}, "
          f"{len(cp['messages'])} messages, {cp['tokens_used']} tokens")
    res = resume_agent(run_id)
    log_run(res, "crash_and_resume")
    print(f"  resumed run -> {res['status']} {res['refund_inr']:.2f}; tools executed across crash + resume: "
          f"{res['tools_executed']}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) < 0.01

    print("\nrefund ledger:")
    for line in open(LEDGER_FILE, encoding="utf-8"):
        print("  " + line.strip())
    print("checkpoints in state/:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(STATE, "*.json"))))


if __name__ == "__main__":
    main()
