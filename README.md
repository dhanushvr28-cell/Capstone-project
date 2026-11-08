# Instamart Customer Intelligence & Support Copilot

## Overview

This repository is the complete capstone Copilot for a fictional Swiggy Instamart scenario. All data is synthetic, made by `generate_data.py`. All amounts are in INR. It is built in four connected Parts, and each Part reads what the previous ones produced:

| Part | What it does | Main outputs |
|---|---|---|
| A. Data foundation | Audits the dirty raw exports, cleans them with documented rules, stores them in SQLite and answers seven business questions in SQL and pandas | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B. Segments and lapse prediction | RFM table, K-means segments (K = 4), PCA, and a leakage-free lapse model (logistic regression vs random forest) | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C. Ticket triage with guardrails | Empty/long-ticket guardrails, PII redaction, zero-/few-shot classification through `mock_llm.py`, retries with jittered backoff, output validation, token budget, segment-aware routing | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/triage_retry_log.txt` |
| D. Policy RAG and refund agent | TF-IDF retrieval over five policy documents, grounded answers, and a ReAct refund agent with five tools, safety limits and crash-resume checkpoints | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/*.json` |

No API key, account, network call or paid service is used anywhere. Every language-model call goes through the supplied offline `mock_llm.py`. `generate_data.py` and `mock_llm.py` are committed exactly as supplied. Allowed libraries only: the standard library, pandas, numpy, scikit-learn and joblib (pinned in `requirements.txt`).

Layout:

```
generate_data.py, mock_llm.py      supplied files (unmodified)
src/common.py                      shared paths; makes the repo root the working directory
src/part_a_clean.py                Part A tasks 2-5  audit, clean, store (CSV + SQLite)
src/validate.py                    Part A task 5     prints ALL CHECKPOINTS PASSED
src/part_a_sql.py                  Part A task 6     runs sql/queries.sql with pd.read_sql_query
src/part_a_eda.py                  Part A task 7     pandas cross-check + statistics
src/part_b_segments.py             Part B tasks 1-3  RFM, K-means, PCA, segments
src/part_b_lapse.py                Part B tasks 4-8  lapse model
src/part_c_triage.py               Part C            triage pipeline
src/part_d_rag.py                  Part D tasks 1-4  chunking, retrieval, answers, evaluation
src/part_d_agent.py                Part D tasks 5-8  tools, ReAct agent, limits, resume
sql/queries.sql                    the seven SQL queries
```

## Reproduce

Tested with Python 3.13 and SQLite 3.53. Works with any Python 3.11 or newer. Run from the repository root:

```bash
1.  python -m venv .venv
2.  source .venv/bin/activate            # Windows: .venv\Scripts\activate
3.  pip install -r requirements.txt
4.  python generate_data.py              # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
5.  python src/part_a_clean.py           # audit + clean + data/clean/*.csv + data/instamart.db
6.  python src/validate.py               # prints ALL CHECKPOINTS PASSED
7.  python src/part_a_sql.py             # Q1-Q7 from sql/queries.sql
8.  python src/part_a_eda.py             # pandas cross-check and statistics
9.  python src/part_b_segments.py        # outputs/customer_segments.csv
10. python src/part_b_lapse.py           # outputs/lapse_model.joblib, outputs/retention_list.csv
11. python src/part_c_triage.py          # outputs/triage_results.csv, outputs/triage_budget_demo.csv
12. python src/part_d_rag.py             # policy RAG evaluation (overlap 10 and overlap 0)
13. python src/part_d_agent.py           # outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/
```

Every script is deterministic. The seeds are `random.Random(42)` in the generator, `random_state=42` for K-means, splits, forests and resampling, and `random.Random(42)` for the retry jitter. A rerun reproduces every number below. The only things that change are the wall-clock timestamps in the ledger. `src/part_d_agent.py` truncates its two `.jsonl` files and clears `state/` at the start, so reruns never duplicate lines.

## Part A

### Task 1: setup
The repo is on branch `main`. The environment was made with `python -m venv .venv`, and `requirements.txt` came from `pip freeze`. `.gitignore` lists `.env`, `.venv/`, `venv/`, `__pycache__/` and chart files. The supplied files are copied unmodified.

### Task 2: audit of the raw data
The raw CSVs were read with `keep_default_na=False, na_values=[""]`, so only a truly blank cell counts as missing. Text like `N/A` or `error` in `order_amount` stays visible, so it can be counted as non-numeric. Tickets are read with `keep_default_na=False` in Part C.

**customers.csv: 1,200 rows × 5 columns**

| column | dtype | missing | missing % |
|---|---|---|---|
| customer_id | object | 0 | 0.00 |
| city | object | 0 | 0.00 |
| signup_date | object | 0 | 0.00 |
| age | float64 | 61 | 5.08 |
| gender | object | 30 | 2.50 |

**products.csv: 40 rows × 4 columns.** product_id, product_name and category are object; price_inr is int64. No missing values.

**orders.csv: 13,084 rows × 9 columns**

| column | dtype | missing | missing % |
|---|---|---|---|
| order_id, customer_id, product_id, order_ts, status | object | 0 | 0.00 |
| quantity, discount_pct | int64 | 0 | 0.00 |
| order_amount | object (text values mixed in) | 0 | 0.00 |
| delivery_minutes | float64 | 2,541 | 19.42 |

**Audit figures (all equal the reference)**

| check | value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status `pending` | 656 |
| distinct order_id | 12,309 |
| order_amount not convertible to a number | 120 (67 `error`, 53 `N/A`) |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10–100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

### Task 3: clean customers

| rule | values changed |
|---|---|
| (a) `city`: strip leading/trailing spaces, then title case | 476 |
| (b) `gender`: map M/m/Male/MALE/male → male, F/f/Female/FEMALE/female → female, blank → unknown | 950 (raw text differs from the final label) |
| (c1) `age` outside 10–100 → missing | 6 |
| (c2) missing `age` → median age of the customer's city (`groupby("city")["age"].transform("median")`) | 67 (61 blank + 6 impossible) |
| (d) `signup_date` → datetime | 1,200 |

City medians used: Bengaluru 42, Chennai 39, Delhi 43, Hyderabad 37.5, Mumbai 38.5, Pune 39.
Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Female 610, male 560, unknown 30. 0 missing ages, all between 18 and 60.

### Task 4: clean orders (applied in this order)

| # | rule | rows/values changed |
|---|---|---|
| 1 | `order_ts` → datetime | 13,084 converted |
| 2 | remove exact duplicate rows | 119 rows removed |
| 3 | for each `order_id`, keep only the record with the latest `order_ts` (drops the earlier `pending` copies) | 656 rows removed |
| 4 | `order_amount` → numeric (`errors="coerce"`). Each missing amount is rebuilt as `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 amounts rebuilt (8 of the 120 bad values sat in rows already removed by rules 2–3) |
| 5a | negative `delivery_minutes` → missing | 105 values (one of the 106 was in a removed row) |
| 5b | completed orders: missing `delivery_minutes` → median of the product category among completed orders (`groupby` + `transform`). Medians: 25 min for all categories except Personal Care, 25.5 | 774 values filled |
| 5c | cancelled orders: `delivery_minutes` left missing on purpose. A cancelled order was never delivered, so filling in a time would invent data | 1,199 left missing |

Result: 12,309 rows (one per order_id), 11,110 completed and 1,199 cancelled, no pending rows. 0 missing order_amount. 0 missing delivery_minutes among completed and 1,199 among cancelled. Completed order_amount sums to **INR 2,998,668.80**.

### Task 5: store and validate
The clean tables are written to `data/clean/customers.csv`, `products.csv` and `orders.csv`. They are also written to `data/instamart.db` as tables `customers` (1,200), `products` (40) and `orders` (12,309) with `DataFrame.to_sql`. `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS` and `signup_date` as `YYYY-MM-DD`. `src/validate.py` asserts 17 checkpoints: every reference count from tasks 3 and 4, the INR total, the table sizes and the timestamp format. It prints `ALL CHECKPOINTS PASSED`.

### Task 6: SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| query | technique | result |
|---|---|---|
| Q1 | WHERE (completed, amount > 500), then HAVING (total > 8000) | **30 customers** (top: C0693 with INR 13,776.55 over 6 such orders) |
| Q2 | three-table join, revenue by city × category | **30 rows**, summing to **INR 2,998,668.80**. Largest cell **Bengaluru / Electronics INR 233,909.45** |
| Q3 | scalar subquery, price > AVG(price) | **11 products** (8 Electronics, Sunscreen, Body Lotion, Ghee) |
| Q4 | LEFT JOIN … IS NULL | **25 customers** with no completed order |
| Q5 | two CTEs (spend per customer, average spend) | **550 customers** above the average of **INR 2,552.06** |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER over (PARTITION BY city ORDER BY total_spend DESC) | top per city below. **8 rows** have RANK() ≠ ROW_NUMBER() |
| Q7 | SUM(revenue) OVER (ORDER BY day) running total | **183 daily rows** (2025-04-01 … 2025-09-30). Last cumulative value **INR 2,998,668.80** |

Q6 top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

**Why ties cause the difference.** When customers in the same city have exactly the same total spend, RANK() gives them the same rank. ROW_NUMBER() still numbers them one after another. So every tied customer after the first in a tie group has RANK() < ROW_NUMBER(). For example, Pune C0057 and C0936 both spent INR 285.00 and both get rank 189, but row numbers 190 and 191. Eight such rows exist.

### Task 7: pandas cross-check and exploratory analysis

(a) `pd.pivot_table(..., margins=True)` reproduces Q2. Its grand total is INR 2,998,668.80, equal to the SQL total (asserted in code). Completed orders per city, from `groupby` with named aggregation:

| city | sum (INR) | mean (INR) | count |
|---|---|---|---|
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Chennai | 463,629.15 | 272.72 | 1,700 |

(b) Completed order_amount (11,110 orders):

| statistic | value |
|---|---|
| mean / median | INR 269.91 / INR 135.00 |
| skewness | 4.61 (right-skewed) |
| Q1 / Q3 / IQR | INR 64.80 / INR 297.00 / INR 232.20 |
| lower / upper 1.5×IQR fence | INR −283.50 / INR 645.30 |
| outliers below / above | 0 / 897 (635 of them Electronics) |
| min / max | INR 12.00 / INR 3,998.00 |

Correlations over completed orders: order_amount–discount_pct −0.095, order_amount–quantity +0.087, quantity–delivery_minutes −0.013, discount_pct–delivery_minutes −0.010, order_amount–delivery_minutes −0.003, discount_pct–quantity −0.000.

**What is this data about?** Six months (April to September 2025) of synthetic quick-commerce orders: 11,110 completed orders from 1,175 customers in six cities, across 40 products in five categories.
**What do the distributions look like?** Order value is strongly right-skewed. The mean (INR 269.91) is double the median (INR 135.00), and skewness is 4.61. Electronics makes up 40.7% of revenue from relatively few expensive orders.
**Outliers or anomalies?** 897 orders lie above the INR 645.30 upper fence and none below. 635 of them are Electronics, so they are real big-ticket purchases and were kept. The true anomalies (text amounts, negative times, duplicates) were fixed in cleaning.
**Relationships?** All correlations are weak. The largest in absolute value is order_amount with discount_pct (r = −0.095), just ahead of order_amount with quantity (+0.087). Delivery time is unrelated to everything (|r| ≤ 0.013). Heavier discounts go with smaller bills (median INR 165 at 0% vs INR 90 at 40%). But a correlation this small does not establish causation: it cannot tell us whether discounts shrink bills or are simply given on cheaper items.

## Part B

### Task 1: RFM table (snapshot 2025-10-01 00:00:00, completed orders)
1,175 customers. Recency: median 19 days, min 0, max 180. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55.

### Task 2: K-means (StandardScaler, `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | inertia | silhouette | inertia drop vs K−1 |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | – |
| 3 | 1268.2 | 0.445 | 552.6 |
| **4** | **743.1** | **0.526** | **525.1** |
| 5 | 617.4 | 0.480 | 125.7 |
| 6 | 503.2 | 0.482 | 114.2 |
| 7 | 408.5 | 0.416 | 94.7 |
| 8 | 366.5 | 0.412 | 42.0 |

**Chosen K = 4.** The elbow is clearly at 4: each step up to K = 4 removes more than 500 units of inertia, and after it each step removes only about 100–125. K = 4 also has the best silhouette of all K (0.526). And four groups map onto four distinct campaigns the retention team can actually run.

Why inertia alone can't pick K: inertia is the total squared distance of points to their own centroid. Adding a cluster can never increase it, because the extra centroid can always sit at least as close to some points. So K = 8 always beats K = 2, and inertia would reach 0 at K = n. Inertia rewards splitting customers into ever-smaller groups, so K has to come from where the gains flatten (the elbow), from cluster separation (silhouette) and from whether the business can act on the groups.

### Task 3: segment profiles (K = 4)

Cluster numbers are only labels with no order. Segment 3 is not "worse" than segment 1. The names come from the profiles.

| segment_id | name | size | mean recency (days) | mean frequency | mean monetary (INR) | share of revenue | premium | recommended action |
|---|---|---|---|---|---|---|---|---|
| 1 | **Premium Big Spenders** | 138 | 24.1 | 6.77 | 7,129.70 | 32.8% | **True** | Priority support, early access to electronics/personal-care launches, a personal call if they go quiet |
| 2 | Loyal Regulars | 419 | 8.8 | 19.40 | 3,410.02 | 47.6% | False | Grocery/dairy subscription or loyalty points to protect the weekly habit, no blanket discounts |
| 0 | Occasional Shoppers | 436 | 28.9 | 3.77 | 1,057.62 | 15.4% | False | Cross-category nudge plus a free-delivery threshold offer to lift frequency |
| 3 | Drifting Away | 182 | 109.0 | 2.20 | 685.99 | 4.2% | False | Time-limited win-back coupon and a short "what went wrong" survey |

Premium is the segment with the highest mean monetary value (138 customers). PCA(n_components=2) on the scaled RFM gives `explained_variance_ratio_` = **0.634 and 0.218 (total 0.852)**. PC1 loads −0.58 on recency, +0.61 on frequency and +0.53 on monetary, so it is an overall "engagement" axis. PC2 is mostly monetary (+0.82): spend not explained by frequency. Written to `outputs/customer_segments.csv` (customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### Task 4: lapse problem
Cutoff 2025-09-01. The cohort is the **1,156 customers** with at least one completed order before the cutoff. `lapsed = 1` if they have no completed order from 2025-09-01 to 2025-09-30: **443 lapsed (38.32%)**.

Features, built only from orders before the cutoff plus the clean customer table:
- recency (days from cutoff to last completed order), frequency, monetary
- average discount_pct, cancellation rate (cancelled ÷ completed + cancelled)
- average delivery_minutes of completed orders
- favourite category (most frequent; ties broken alphabetically)
- city, age, gender, tenure_days (cutoff − signup_date)

Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` → **924 train / 232 test** (354 / 89 lapsed).
Baseline "always not lapsed" on test: **accuracy 61.6%, lapsed-class precision 0, recall 0, F1 0** (`zero_division=0`).

### Task 5: preprocessing without leakage
A single `ColumnTransformer` does both steps: `StandardScaler` on the 8 numeric features, and `OneHotEncoder(drop="first")` on city, gender and fav_category (k−1 columns each: 5 + 2 + 4 = 11 dummies, reference levels Bengaluru / female / Dairy). It sits inside a `Pipeline`, so it is fitted on the training rows only, including inside every cross-validation fold.

| | frequency mean |
|---|---|
| learned by the fitted scaler (924 training rows) | **8.1147** |
| full cohort (1,156 rows) | **8.0285** |

They differ because the scaler only ever saw the training rows. The 232 test customers were held out, so their (slightly lower) order counts never shaped the scaling. `assert not X.isna().any().any()` confirms no feature column has a missing value.

### Task 6: tuning and test evaluation (GridSearchCV, cv=5, scoring="f1")

| model | grid | best params | best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C ∈ {0.01, 0.1, 1, 10, 100} | **C = 100** (CV F1 by C: 0.562, 0.646, 0.645, 0.649, 0.651) | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth ∈ {3,5,8} × min_samples_leaf ∈ {5,10,20} (9 × 5 = 45 fits) | **max_depth = 8, min_samples_leaf = 10** | 0.681 |

Test set (used only after tuning finished), lapsed class:

| model | TN | FP | FN | TP | precision | recall | F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| baseline | 143 | 0 | 89 | 0 | 0.000 | 0.000 | 0.000 | 0.616 |
| logistic regression (C=100) | 111 | 32 | 31 | 58 | 0.644 | 0.652 | 0.648 | 0.728 |
| **random forest (8, 10)** | 113 | 30 | 31 | 58 | **0.659** | **0.652** | **0.655** | 0.737 |

**Selected model: random forest.** Missing a lapsing customer is the expensive mistake: the average customer is worth about INR 2,552 over six months, while wrongly sending a loyal customer a win-back offer costs only the coupon. So recall on the lapsed class matters most. Both models catch the same 58 of 89 lapsers (recall 0.652), so the tie-breaker is precision. The forest wastes 2 fewer offers (30 vs 32 false positives), giving the higher F1 (0.655 vs 0.648, well above the 0.55 bar). It also captures non-linear effects like "few but expensive orders" without hand-made interaction terms.

The logistic-regression grid picked C = 100, where the reference reports C = 1. Their CV F1 scores differ by only 0.006 (0.651 vs 0.645), so the regularisation choice is nearly flat, and the test F1 (0.648) is of the reference order (about 0.62).

### Task 7: class-balance experiment
Only the training set is over-sampled. All 924 original rows are kept, and 216 randomly chosen lapsed training rows are added as duplicates (`sample(replace=True, random_state=42)`), giving **570 / 570**. The best logistic regression (C = 100) is refit on this and scored on the unchanged test set:

| metric (lapsed class) | unbalanced | balanced | change |
|---|---|---|---|
| precision | 0.644 | 0.589 | −0.055 |
| **recall** | 0.652 | **0.854** | **+0.202** |
| F1 | 0.648 | 0.697 | +0.049 |

Recall moved most. The balanced model catches 76 of 89 lapsers instead of 58, at the cost of 53 instead of 32 false alarms, and precision stays near 0.6 (it does not rise). For the retention team, if the offer is cheap compared with a lost customer, the balanced model (or a lower probability threshold) is the better operating point.

### Task 8: interpret, persist, use

Top 5 random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery_minutes 0.061, cancel_rate 0.052.

| logistic-regression feature (scaled) | coefficient | business reading |
|---|---|---|
| frequency | −1.70 (negative) | customers with a strong ordering habit rarely lapse, so habit is the best protection |
| recency | +0.04 (positive) | the longer since the last order, the higher the lapse risk (small once frequency is known) |
| monetary | +0.28 (positive) | at the same frequency, high spenders are more likely to skip a month: big-ticket, low-rate shoppers need separate retention |

The selected forest pipeline was saved with `joblib.dump` to `outputs/lapse_model.joblib`, then reloaded with `joblib.load`. Its predictions on all 232 test rows are identical (asserted, and the probabilities match too).

**Retention list** (`outputs/retention_list.csv`). There are 40 test-set customers in the Premium segment. The top 10 by predicted lapse probability:

| customer_id | lapse_probability | actually_lapsed |
|---|---|---|
| C0862 | 0.491 | 0 |
| C1112 | 0.484 | 0 |
| C0060 | 0.480 | 0 |
| C0802 | 0.480 | 0 |
| C1008 | 0.478 | 0 |
| C1000 | 0.462 | 0 |
| C0525 | 0.460 | 1 |
| C0133 | 0.456 | 0 |
| C0267 | 0.453 | 0 |
| C0167 | 0.452 | 0 |

**1 of the 10 truly lapsed.** The Premium flag comes from whole-window RFM, which includes the September window that defines the lapse label. It is a post-hoc whole-window label, so this retention list is illustrative and not a leakage-free evaluation.

**Leakage statement.** No encoder, scaler or resampler ever saw test rows: all are fitted on training rows inside a Pipeline, and over-sampling touched only the training set. No order dated 2025-09-01 or later entered any feature (asserted in code). The segment labels from task 3 were not used as model features.

## Part C

### Task 1: input guardrails
164 tickets, processed in file order. **4 empty** tickets get `skipped_empty`. **4 long** tickets (each `est_tokens` = **734** > `MAX_INPUT_TOKENS = 400`) get `flagged_long`. Neither kind is ever sent to the model. 156 tickets are eligible.

### Task 2: PII redaction (order: email → phone → order id)

| type | pattern | placeholder | substitutions |
|---|---|---|---|
| email | `[\w.+-]+@[\w-]+\.[\w.]+` | [EMAIL] | 55 |
| phone | `\b[6-9]\d{9}\b` | [PHONE] | 91 |
| order id | `ORD-\d{6}` | [ORDER_ID] | 132 |
| **total** | | | **278** |

Re-applying all three patterns to the redacted texts finds **0** matches (asserted). The user message is exactly the redacted text. Before every call, an assertion checks that no message contains an email, phone or order id.

### Task 3: two prompts
Both contain CLASSIFY, list billing, delivery, technical and fraud, and demand JSON only, as `{"category": "...", "reason": "..."}`. The zero-shot prompt is 41 estimated tokens and has no `EXAMPLES:`. The few-shot prompt is 123 estimated tokens and adds an `EXAMPLES:` line followed by four labelled examples (fraud, technical, billing, delivery). See `ZERO_SHOT` and `FEW_SHOT` in `src/part_c_triage.py`.

### Task 4: resilient calls
Every call goes through `call_with_retry`: at most 3 retries, with wait = `0.05 × 2**n × uniform(0.9, 1.15)` (seeded `random.Random(42)`). There is one log line per retry, with ticket id, retry number and wait (`outputs/triage_retry_log.txt`). `mock_llm.reset()` runs before each full run. Each full run over the 156 eligible tickets raised **19 TransientLLMErrors, all recovered on retry number 0**. Waits ranged from 0.0453 to 0.0562 s, inside the allowed 0.045–0.0575 s. No ticket ended with a permanent error.

### Task 5: output validation

| status | rule | count (each full run) |
|---|---|---|
| classified_json | valid JSON and category in the allowed four | **135** |
| classified_keyword | not JSON. `re.findall(r"[a-z]+", text.lower())` finds exactly one allowed category word (the chatty "…a billing-related problem.") | **12** |
| flagged_invalid_output | anything else, including valid JSON with category "shipping". Sent to human review, never guessed | **9** |

### Task 6: zero-shot vs few-shot (no budget)

| run | classified | accuracy | total tokens | tokens per classified ticket |
|---|---|---|---|---|
| zero-shot | 147 | **85.7%** (126/147) | 10,804 | 73.5 |
| few-shot | 147 | **100.0%** (147/147) | 23,596 | 160.5 |

Zero-shot confusion table (rows = true, columns = predicted):

| true \ pred | billing | delivery | fraud | technical | total |
|---|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 | 38 |
| delivery | 0 | 36 | 0 | 0 | 36 |
| fraud | **11** | 0 | 23 | 0 | 34 |
| technical | **10** | 0 | 0 | 29 | 39 |
| total | 59 | 36 | 23 | 29 | 147 |

**Cost of few-shot.** The few-shot prompt costs 87 extra tokens per classified ticket (160.5 vs 73.5), so the run uses 2.2× the tokens (+12,792 in total). It buys 21 extra correct tickets (126 → 147), about 609 extra tokens per corrected ticket. All 21 zero-shot errors were fraud or technical tickets mistaken for billing, because they mention "charged" or "payment". So without the examples, 11 fraud cases would sit in the billing queue with a 24-hour SLA instead of reaching security within 2 hours. That risk is worth far more than the tokens, so few-shot is used for the production run. Zero-shot only makes sense when the token budget is the binding limit and fraud can be caught some other way.

### Task 7: budget and routing
**(a) Budget demo.** A zero-shot run with `TOKEN_BUDGET = 6000` (checked before every call) used 6,033 tokens. Then **69 tickets ended `budget_exhausted`** and went to human review (`outputs/triage_budget_demo.csv`). That number reflects the 41-token zero-shot prompt.

**(b) Final triage.** A few-shot run with `TOKEN_BUDGET = 100000` (23,596 used) is routed with `outputs/customer_segments.csv`. A customer with no segment row counts as not premium. Results are in `outputs/triage_results.csv`, 164 rows:

| route | priority | tickets | from |
|---|---|---|---|
| security_team | HIGH | 34 | category fraud |
| premium_desk | HIGH | 11 | non-fraud tickets from Premium customers |
| support_queue | NORMAL | 102 | everything else that was classified |
| human_review | NORMAL | 17 | 4 skipped_empty + 4 flagged_long + 9 flagged_invalid_output |

**What was rule-validated and what went to humans.** 147 tickets were classified by the model and then accepted by deterministic rule checks: 135 JSON replies and 12 single-keyword replies. These were routed automatically. The other 17 went to human review without a guessed category: 8 never reached the model (empty or too long), and 9 replies failed validation.

## Part D

### Task 1: chunking
40-word chunks with a 10-word overlap (step 30), stopping after the first chunk that reaches the last word: **15 chunks**. Refund 4 (119 words), delivery 3 (100), escalation 3 (78), payment 3 (86), privacy 2 (68). Each chunk records its source file.

### Task 2: TF-IDF retrieval
`TfidfVectorizer()` is fitted on the 15 chunk texts (200-term vocabulary). Queries are transformed with the same vectorizer and ranked by `cosine_similarity`. `retrieve(query, k=3)` returns `{source, score, text}`. For "refund policy late delivery" the top three are all `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word appears in a chunk, so a chunk that repeats "refund" scores higher for refund questions. IDF (inverse document frequency) measures how rare the word is across all chunks, so common words like "order" count little and distinctive words like "wallet" or "password" count a lot. One limitation: TF-IDF only matches exact words. "Can I get my money back?" shares almost no terms with "a refund can be requested", and "OTP" never matches "one time password", so a question phrased differently from the policy can miss the right chunk.

### Task 3: augment and answer
`answer(question)` retrieves three chunks. It sends a system message containing "POLICY ASSISTANT", telling the model to answer only from the context or else say "I could not find that information in the policy documents". The user message has the form `CONTEXT:\n[source.txt]\nchunk…\nQUESTION:\nquestion`. It returns the model's text and the sources used.

### Task 4: evaluation (`data/raw/policy_eval.csv`)

| setting | chunks | expected_doc in top 3 (12 answerable) | expected_fact in answer (12 answerable) | fallback on 2 unanswerable |
|---|---|---|---|---|
| overlap 10 | 15 | **12/12** | **12/12** | **2/2** |
| overlap 0 | 13 | 12/12 | **10/12** | 2/2 |

With zero overlap, two questions break, and in both the right document is still retrieved:
- "What delivery time makes a completed order eligible for a full refund?" Refund chunk 0 ends "…If a completed order is delivered in more than 45 minutes, the customer", and "receives a full refund…" starts chunk 1. The model answers only "receives a full refund of the order amount.", which lacks "more than 45 minutes".
- "How soon is money returned when a payment fails but is deducted?" Payment chunk 0 ends "…the amount is returned automatically", and "within 48 hours" is in chunk 1, which was not retrieved, so "48 hours" is missing.

The 10-word overlap keeps such sentences whole in at least one chunk.

### Task 5: tools
There are five tools, each with a name, description and JSON-schema parameters in `TOOLS_SPEC`, and dispatched through `TOOL_MAP`:
- `get_order` reads `data/instamart.db`.
- `search_policy` reuses `retrieve()` from `part_d_rag.py`.
- `calculate_refund` applies the rule: 100% if delivered in more than 45 minutes, otherwise 0.
- `issue_refund` and `escalate_to_human` each append a timestamped JSON line to `outputs/refund_ledger.jsonl`. As an extra safeguard, `issue_refund` refuses anything above INR 1000.

### Task 6: ReAct loop, seven scenarios (max_iterations=6, token_limit=6000, time_limit_seconds=10)

| scenario | expected | status | refund (INR) | model calls | tokens used | tools executed |
|---|---|---|---|---|---|---|
| S1 | REFUND | **REFUND** | 128.25 | 5 | 1,197 | get_order → search_policy → calculate_refund → issue_refund |
| S2 | REFUND | **REFUND** | 135.00 | 5 | 1,197 | get_order → search_policy → calculate_refund → issue_refund |
| S3 | ESCALATE | **ESCALATE** | 1,999.00 | 5 | 1,184 | get_order → search_policy → calculate_refund → escalate_to_human |
| S4 | DENIED | **DENIED** | 0.00 | 4 | 845 | get_order → search_policy → calculate_refund (delivered within 45 min) |
| S5 | DENIED | **DENIED** | 0.00 | 3 | 517 | get_order → search_policy (cancelled order) |
| S6 | NOT_FOUND | **NOT_FOUND** | 0.00 | 2 | 192 | get_order (ORD-999999 not found) |
| S7 | FALLBACK | **FALLBACK_MAX_ITERATIONS** | 0.00 | 6 | 3,360 | search_policy × 6 |

Every refund equals `expected_refund_inr` within 0.01 (asserted). Each run is one line in `outputs/agent_runs.jsonl`, with scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool, arguments and observation. The ledger holds the S1 and S2 refunds, the S3 escalation and the resumed S1 refund.

### Task 7: stop safely

| demonstration | status | detail |
|---|---|---|
| S7 with defaults | **FALLBACK_MAX_ITERATIONS** | all 6 model calls completed, 3,360 tokens (under 6,000) |
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | stopped after model call 2 (222 tokens). The requested `search_policy` was **not** executed |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped before the first model call (`elapsed >= 0` measured with `time.monotonic()`) |

Every fallback returns the steps completed so far, plus a `HANDOFF to human agent: …` summary in `final_answer`. The summary gives the reason, request, model calls, tokens, tools executed and last observation.

### Task 8: crash and resume (S1)
A checkpoint `state/<run_id>.json` is written atomically after every tool observation. It holds the run id, scenario id, full message list, tokens_used, step number, tools executed and the limits. The demo raises a simulated crash right after the second tool, once its checkpoint is saved. `resume_agent("S1-crash-resume")` then loads the checkpoint (step 2, 6 messages, 222 tokens) and continues.

**Tools executed across crash + resume, in order: `get_order`, `search_policy` (crash here), `calculate_refund`, `issue_refund`.** That is four executions with none repeated. The final status is **REFUND, INR 128.25**.

### Task 9: ReAct vs plan-and-execute
My ReAct agent asks the model before every action. It used 5 calls for S1 (1,197 tokens), 3 for S5 (517) and 2 for S6 (192). The cost grows with the number of steps, because each call resends the whole history, including the bulky policy chunks. A plan-and-execute agent would ask once for the full plan (get_order → search_policy → calculate_refund → issue/escalate) and run it with no further model calls. That is about 1–2 calls per request instead of 5, so it is far cheaper on tokens.

The difference shows on unexpected observations. When get_order returns NOT_FOUND (S6) or a cancelled order (S5), ReAct simply reads the observation and stops with the right answer on the next call. A fixed plan would carry on into calculate_refund on a missing or cancelled order unless the executor detects the problem and pays for a replan.

For this refund workflow I would keep ReAct, wrapped in the limits already built (6 iterations, token cap, timeout, checkpoints). The workflow is short (at most 5 calls). The branches (not found, cancelled, on time, above INR 1000) are exactly where a rigid plan goes wrong. And the money-moving step must be decided on the actual observation, not on a plan written before the order was seen. If volume made cost matter, a hybrid could hard-code the fixed prefix (get_order, search_policy) and use the model only for the decision.

## Limitations and Next Steps

- **The retention list is illustrative.** The Premium flag is a post-hoc whole-window label: it comes from RFM over the full window, including the September window that defines the lapse label. So `outputs/retention_list.csv` is illustrative, not a leakage-free evaluation. A production version should compute segments from pre-cutoff data only. (The model's own features use only orders before 2025-09-01.)
- **Lapse threshold.** The forest uses the default 0.5 threshold. The balance experiment shows recall can rise from 0.65 to 0.85, so the next step is to pick the threshold from the real cost of a coupon versus a lost customer. More history (several cutoffs) would also make the model more stable than one 924-row training set allows.
- **Logistic-regression C.** The grid chose C = 100, not the reference C = 1. The CV curve is almost flat from C = 0.1 upward, so this is within noise.
- **The mock LLM is keyword-driven.** The 100% few-shot accuracy and the 12/12 grounding reflect `mock_llm.py`'s rules, not real model quality. With a hosted model, the same guardrails (redaction, validation, retries, budgets, limits) would carry over, but accuracy would need re-measuring.
- **TF-IDF is lexical.** Paraphrases and abbreviations ("money back", "OTP") can miss the right chunk. Next steps: sentence-aware chunking, a synonym list or embeddings, and a minimum-score threshold before answering.
- **Refund idempotency.** The ledger is append-only, and the crash-resume demo re-issues S1's refund as a separate run. A real system needs an idempotency key per order, so the same refund can never be paid twice across runs or retries.
- **Refund window.** All simulated requests are assumed to be inside the 7-day window. The agent does not check request dates.
- **Synthetic data.** Everything is generated by `generate_data.py`. Patterns (for example, uniform delivery times across cities) are simpler than real operations.
