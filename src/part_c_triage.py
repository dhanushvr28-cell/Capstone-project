"""Part C: support-ticket triage around mock_llm.py with input guardrails, PII redaction,
retries with exponential backoff + jitter, output validation, a token budget and segment-aware routing.

Run:  python src/part_c_triage.py      (needs outputs/customer_segments.csv from Part B)
Writes outputs/triage_budget_demo.csv, outputs/triage_results.csv and outputs/triage_retry_log.txt
"""
import json
import math
import random
import re
import time

import pandas as pd

from common import OUT, RAW, header
import mock_llm
from mock_llm import TransientLLMError, est_tokens

pd.set_option("display.width", 160)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted",
                  "error_permanent"}

# ---------------------------------------------------------------- task 2: redaction patterns (order matters)
PII_PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
                ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
                ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text):
    counts = {}
    for name, pat, repl in PII_PATTERNS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts


def contains_pii(text):
    return any(p.search(text or "") for _, p, _ in PII_PATTERNS)


# ---------------------------------------------------------------- task 3: two prompts
ZERO_SHOT = """You are a support-ticket classifier for a quick-commerce grocery app. CLASSIFY the ticket.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, exactly {"category": "...", "reason": "..."}, no other text."""

FEW_SHOT = """You are a support-ticket classifier for a quick-commerce grocery app. CLASSIFY the ticket.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, exactly {"category": "...", "reason": "..."}, no other text.
EXAMPLES:
Ticket: Someone used my card for an order I never placed. -> {"category": "fraud", "reason": "unauthorized charge"}
Ticket: The app shows an error at the payment page. -> {"category": "technical", "reason": "app error"}
Ticket: I was charged twice for one order. -> {"category": "billing", "reason": "double charge"}
Ticket: The rider has not arrived after an hour. -> {"category": "delivery", "reason": "late delivery"}"""

assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "\nEXAMPLES:\n" in FEW_SHOT and '"fraud"' in FEW_SHOT.split("EXAMPLES:")[1]
for _p in (ZERO_SHOT, FEW_SHOT):
    assert all(c in _p for c in ALLOWED) and '{"category": "...", "reason": "..."}' in _p


# ---------------------------------------------------------------- task 4: resilient call
class RetryLog:
    def __init__(self):
        self.lines = []
        self.errors = 0

    def add(self, line):
        self.lines.append(line)
        print("  " + line)


def call_with_retry(messages, ticket_id, jitter_rng, log, budget_state, token_budget):
    """Call mock_llm.chat with up to MAX_RETRIES retries. Returns (response or None, status_if_no_response)."""
    for attempt in range(MAX_RETRIES + 1):
        if budget_state["used"] >= token_budget:               # task 7 budget check before EVERY model call
            return None, "budget_exhausted"
        for m in messages:                                      # privacy guardrail: nothing identifying leaves
            assert not contains_pii(m["content"]), f"PII would reach the model in {ticket_id}"
        try:
            resp = mock_llm.chat(messages)
            budget_state["used"] += resp.usage.total_tokens
            return resp, None
        except TransientLLMError as e:
            log.errors += 1
            if attempt == MAX_RETRIES:
                log.add(f"{ticket_id} giving up after {MAX_RETRIES} retries: {e}")
                return None, "error_permanent"
            wait = BASE_DELAY * 2 ** attempt * jitter_rng.uniform(0.9, 1.15)
            log.add(f"RETRY ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s error='{e}'")
            time.sleep(wait)


# ---------------------------------------------------------------- task 5: validate the answer
def validate(content):
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return "classified_json", data["category"]
        return "flagged_invalid_output", None              # valid JSON, category outside the list -> never guess
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in ALLOWED if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None


# ---------------------------------------------------------------- pipeline
def load_tickets():
    return pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False, dtype=str)


def run_pipeline(tickets, system_prompt, token_budget=math.inf, label=""):
    mock_llm.reset()                                        # fresh counters before each full run
    jitter_rng = random.Random(42)
    log = RetryLog()
    budget_state = {"used": 0}
    rows, sent = [], 0
    print(f"\n--- run: {label} (token budget {token_budget}) ---")
    for t in tickets.itertuples(index=False):
        text = t.text
        rec = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": None, "category": "", "tokens_used": 0, "raw_reply": ""}
        if not text.strip():                                # task 1a
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:           # task 1b
            rec["status"] = "flagged_long"
        else:
            clean, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean}]     # user content = redacted text only
            resp, fail_status = call_with_retry(messages, t.ticket_id, jitter_rng, log, budget_state, token_budget)
            if resp is None:
                rec["status"] = fail_status
            else:
                sent += 1
                content = resp.choices[0].message.content
                rec["raw_reply"] = content
                rec["tokens_used"] = resp.usage.total_tokens
                rec["status"], cat = validate(content)
                rec["category"] = cat or ""
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"  TransientLLMErrors: {log.errors}; retries logged: {len(log.lines)}; "
          f"tickets answered by the model: {sent}; total tokens: {budget_state['used']}")
    return df, log, budget_state["used"]


def run_summary(df, total_tokens):
    classified = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    acc = correct / len(classified) if len(classified) else 0
    return {"classified": len(classified), "correct": correct, "accuracy_pct": round(acc * 100, 1),
            "total_tokens": total_tokens,
            "tokens_per_classified": round(total_tokens / len(classified), 1) if len(classified) else None,
            "status_counts": df["status"].value_counts().to_dict()}


def route(df, segments):
    seg = segments.set_index("customer_id")
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    premium = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)   # no row -> not premium
    out["route"], out["priority"] = "support_queue", "NORMAL"
    is_prem = premium & ~out["status"].isin(HUMAN_STATUSES)
    out.loc[is_prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    is_fraud = (out["category"] == "fraud") & ~out["status"].isin(HUMAN_STATUSES)
    out.loc[is_fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    human = out["status"].isin(HUMAN_STATUSES)
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    out.loc[human, "category"] = ""                          # no trusted category for human-review tickets
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def main():
    tickets = load_tickets()

    header("PART C - TASK 1: INPUT GUARDRAILS")
    empty = tickets["text"].str.strip() == ""
    tok = tickets["text"].map(est_tokens)
    long_ = ~empty & (tok > MAX_INPUT_TOKENS)
    print(f"tickets: {len(tickets)}; empty/whitespace: {empty.sum()}; long (> {MAX_INPUT_TOKENS} tokens): "
          f"{long_.sum()} with estimates {sorted(tok[long_].unique().tolist())}; eligible: {(~empty & ~long_).sum()}")

    header("PART C - TASK 2: PII REDACTION (email -> phone -> order id)")
    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted = []
    for text in tickets["text"]:
        r, c = redact(text)
        redacted.append(r)
        for k in totals:
            totals[k] += c[k]
    print(f"substitutions: emails {totals['email']}, phones {totals['phone']}, order ids {totals['order_id']}, "
          f"total {sum(totals.values())}")
    left = sum(len(p.findall(r)) for r in redacted for _, p, _ in PII_PATTERNS)
    print(f"matches left after re-applying all three patterns to the redacted texts: {left}")
    assert left == 0
    print("example:", tickets["text"].iloc[0], "\n     -> ", redacted[0])

    header("PART C - TASK 3: PROMPTS")
    print(f"zero-shot prompt: {est_tokens(ZERO_SHOT)} estimated tokens\n{ZERO_SHOT}\n")
    print(f"few-shot prompt:  {est_tokens(FEW_SHOT)} estimated tokens\n{FEW_SHOT}")

    header("PART C - TASKS 4-6: FULL RUNS, ZERO-SHOT vs FEW-SHOT (no token budget)")
    zs, zs_log, zs_tok = run_pipeline(tickets, ZERO_SHOT, label="zero-shot")
    fs, fs_log, fs_tok = run_pipeline(tickets, FEW_SHOT, label="few-shot")
    for name, df, tok_used, log in [("zero-shot", zs, zs_tok, zs_log), ("few-shot", fs, fs_tok, fs_log)]:
        s = run_summary(df, tok_used)
        waits = [float(re.search(r"wait=([\d.]+)s", l).group(1)) for l in log.lines if l.startswith("RETRY")]
        nums = sorted({int(re.search(r"retry_number=(\d+)", l).group(1)) for l in log.lines if l.startswith("RETRY")})
        print(f"\n{name}: statuses {s['status_counts']}")
        print(f"  classified {s['classified']}; accuracy {s['accuracy_pct']}% ({s['correct']}/{s['classified']}); "
              f"total tokens {s['total_tokens']}; tokens per classified ticket {s['tokens_per_classified']}")
        print(f"  TransientLLMErrors {log.errors}, all recovered: {not (df['status'] == 'error_permanent').any()}; "
              f"retry numbers used {nums}; wait range {min(waits):.4f}-{max(waits):.4f}s "
              f"(allowed 0.0450-0.0575s for retry 0)")
        assert all(0.045 <= w <= 0.0575 for w in waits)

    cls = zs[zs["status"].isin(["classified_json", "classified_keyword"])]
    print("\nZero-shot confusion table over classified tickets (rows = true_category, cols = predicted):")
    print(pd.crosstab(cls["true_category"], cls["category"], margins=True).to_string())
    errs = cls[cls["category"] != cls["true_category"]]
    print("zero-shot errors:", errs.groupby(["true_category", "category"]).size().to_dict())

    with open(f"{OUT}/triage_retry_log.txt", "w", encoding="utf-8") as f:
        f.write("# zero-shot full run\n" + "\n".join(zs_log.lines) + "\n# few-shot full run\n"
                + "\n".join(fs_log.lines) + "\n")

    segments = pd.read_csv(f"{OUT}/customer_segments.csv")

    header("PART C - TASK 7a: TOKEN BUDGET DEMO (zero-shot, TOKEN_BUDGET = 6000)")
    demo, _, demo_tok = run_pipeline(tickets, ZERO_SHOT, token_budget=6000, label="zero-shot budget demo")
    demo_out = route(demo, segments)
    demo_out.to_csv(f"{OUT}/triage_budget_demo.csv", index=False)
    n_ex = int((demo["status"] == "budget_exhausted").sum())
    print(f"tokens used {demo_tok}; tickets ending budget_exhausted: {n_ex}")
    print("statuses:", demo["status"].value_counts().to_dict())
    assert n_ex >= 1

    header("PART C - TASK 7b: FINAL TRIAGE (few-shot, TOKEN_BUDGET = 100000) + ROUTING")
    final, _, final_tok = run_pipeline(tickets, FEW_SHOT, token_budget=100000, label="few-shot final")
    final_out = route(final, segments)
    final_out.to_csv(f"{OUT}/triage_results.csv", index=False)
    print(f"rows {len(final_out)}; tokens {final_tok}")
    print("statuses:", final_out["status"].value_counts().to_dict())
    print("routes:", final_out["route"].value_counts().to_dict())
    print(pd.crosstab(final_out["route"], final_out["priority"]).to_string())
    print("human_review reasons:", final_out.loc[final_out["route"] == "human_review", "status"].value_counts().to_dict())
    assert len(final_out) == 164


if __name__ == "__main__":
    main()
