"""Part A, task 5: assert the reference checkpoints of tasks 3 and 4 (and the database contents).
Prints exactly ALL CHECKPOINTS PASSED when every assertion holds.

Run:  python src/validate.py
"""
import sqlite3

import pandas as pd

from common import CLEAN, DB_PATH

c = pd.read_csv(f"{CLEAN}/customers.csv")
o = pd.read_csv(f"{CLEAN}/orders.csv")
comp = o["status"] == "completed"
canc = o["status"] == "cancelled"

checks = [
    # --- task 3: customers
    ("six cities with reference counts",
     c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                             "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("genders female 610 / male 560 / unknown 30",
     c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages", c["age"].isna().sum() == 0),
    ("every age between 18 and 60", c["age"].between(18, 60).all()),
    ("signup_date parses as a date", pd.to_datetime(c["signup_date"], format="%Y-%m-%d").notna().all()),
    # --- task 4: orders
    ("12,309 order rows", len(o) == 12309),
    ("one row per order_id", o["order_id"].is_unique),
    ("11,110 completed orders", comp.sum() == 11110),
    ("1,199 cancelled orders", canc.sum() == 1199),
    ("no pending rows", (o["status"] == "pending").sum() == 0),
    ("0 missing order_amount", o["order_amount"].isna().sum() == 0),
    ("0 missing delivery_minutes among completed", o.loc[comp, "delivery_minutes"].isna().sum() == 0),
    ("1,199 missing delivery_minutes among cancelled", o.loc[canc, "delivery_minutes"].isna().sum() == 1199),
    ("no negative delivery_minutes", (o["delivery_minutes"].dropna() >= 0).all()),
    ("completed order_amount sums to INR 2,998,668.80",
     round(o.loc[comp, "order_amount"].sum(), 2) == 2998668.80),
]

with sqlite3.connect(DB_PATH) as con:
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("customers", "products", "orders")}
    ts_ok = con.execute("SELECT COUNT(*) FROM orders WHERE order_ts NOT GLOB "
                        "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'"
                        ).fetchone()[0] == 0
checks += [
    ("database tables customers 1200 / products 40 / orders 12309",
     counts == {"customers": 1200, "products": 40, "orders": 12309}),
    ("order_ts stored as text YYYY-MM-DD HH:MM:SS", ts_ok),
]

for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, f"checkpoint failed: {name}"
print("ALL CHECKPOINTS PASSED")
