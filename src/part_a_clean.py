"""Part A, tasks 2-5: audit the raw exports, clean customers and orders with documented rules,
then write the clean tables to data/clean/*.csv and to the SQLite database data/instamart.db.

Run:  python src/part_a_clean.py
"""
import sqlite3

import numpy as np
import pandas as pd

from common import CLEAN, DB_PATH, RAW, header, inr

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


# ---------------------------------------------------------------- load
def load_raw():
    # Only a truly blank cell counts as missing. Strings such as "N/A" or "error" stay as text
    # so the audit can count them as non-numeric amounts instead of hiding them as NaN.
    kw = dict(keep_default_na=False, na_values=[""])
    customers = pd.read_csv(f"{RAW}/customers.csv", **kw)
    products = pd.read_csv(f"{RAW}/products.csv", **kw)
    orders = pd.read_csv(f"{RAW}/orders.csv", **kw)
    return customers, products, orders


# ---------------------------------------------------------------- task 2: audit
def missing_report(name, df):
    print(f"\n[{name}] shape = {df.shape[0]} rows x {df.shape[1]} columns")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())


def audit(customers, products, orders):
    header("PART A - TASK 2: AUDIT OF THE RAW DATA")
    for name, df in [("customers.csv", customers), ("products.csv", products), ("orders.csv", orders)]:
        missing_report(name, df)

    amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    figures = {
        "raw orders.csv rows": len(orders),
        "exact duplicate rows": int(orders.duplicated().sum()),
        "rows with status pending": int((orders["status"] == "pending").sum()),
        "distinct order_id": int(orders["order_id"].nunique()),
        "non-numeric order_amount": int((amt_num.isna() & orders["order_amount"].notna()).sum()),
        "negative delivery_minutes": int((deliv < 0).sum()),
        "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
        "customers: blank age": int(customers["age"].isna().sum()),
        "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
        "customers: distinct raw city spellings": int(customers["city"].nunique()),
        "customers: distinct non-blank raw gender spellings": int(customers["gender"].dropna().nunique()),
        "customers: blank gender": int(customers["gender"].isna().sum()),
    }
    print("\nAudit figures:")
    for k, v in figures.items():
        print(f"  {k:<52} {v:>7,}")
    print("\nNon-numeric order_amount values:", orders.loc[amt_num.isna() & orders["order_amount"].notna(),
                                                           "order_amount"].value_counts().to_dict())
    print("Raw city spellings:", sorted(customers["city"].unique(), key=str))
    print("Raw gender spellings:", sorted(customers["gender"].dropna().unique()))
    return figures


# ---------------------------------------------------------------- task 3: clean customers
GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}


def clean_customers(raw):
    header("PART A - TASK 3: CLEAN CUSTOMERS")
    c = raw.copy()
    log = []

    # (a) city: strip spaces + title case
    before = c["city"].copy()
    c["city"] = c["city"].str.strip().str.title()
    log.append(("city: strip + title case", int((before != c["city"]).sum())))

    # (b) gender: map every spelling to male / female / unknown (blank -> unknown)
    before = c["gender"].copy()
    c["gender"] = c["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    log.append(("gender: mapped to male/female/unknown", int((before.fillna("") != c["gender"]).sum())))

    # (c) age: impossible values -> missing, then fill with the median age of the customer's city
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    impossible = (c["age"] < 10) | (c["age"] > 100)
    log.append(("age: outside 10-100 set to missing", int(impossible.sum())))
    c.loc[impossible, "age"] = np.nan
    n_missing = int(c["age"].isna().sum())
    city_median = c.groupby("city")["age"].transform("median")
    c["age"] = c["age"].fillna(city_median)
    log.append(("age: missing filled with city median (groupby + transform)", n_missing))
    print("City median ages used for filling:", c.groupby("city")["age"].median().to_dict())

    # (d) signup_date -> datetime
    c["signup_date"] = pd.to_datetime(c["signup_date"], format="%Y-%m-%d")
    log.append(("signup_date: converted to datetime", len(c)))

    for rule, n in log:
        print(f"  {rule:<62} {n:>6,} values changed")
    print("\nCity counts:", c["city"].value_counts().to_dict())
    print("Gender counts:", c["gender"].value_counts().to_dict())
    print(f"Missing ages: {c['age'].isna().sum()}; age range {c['age'].min()} - {c['age'].max()}")
    return c, log


# ---------------------------------------------------------------- task 4: clean orders
def clean_orders(raw, products):
    header("PART A - TASK 4: CLEAN ORDERS")
    o = raw.copy()
    log = []

    # 1. order_ts -> datetime
    o["order_ts"] = pd.to_datetime(o["order_ts"], format="%Y-%m-%d %H:%M:%S")
    log.append(("1. order_ts converted to datetime", len(o)))

    # 2. remove exact duplicate rows
    n = len(o)
    o = o.drop_duplicates().copy()
    log.append(("2. exact duplicate rows removed", n - len(o)))

    # 3. one record per order_id: keep the record with the latest order_ts
    n = len(o)
    o = (o.sort_values(["order_id", "order_ts"], kind="mergesort")
          .drop_duplicates("order_id", keep="last")
          .sort_values("order_id", kind="mergesort").reset_index(drop=True))
    log.append(("3. earlier records of the same order_id removed (latest kept)", n - len(o)))

    # 4. order_amount -> numeric; unparseable -> missing -> rebuilt from price x qty x (1 - discount)
    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    miss = o["order_amount"].isna()
    log.append(("4a. order_amount unparseable -> missing", int(miss.sum())))
    price = o["product_id"].map(products.set_index("product_id")["price_inr"])
    rebuilt = (price * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
    o.loc[miss, "order_amount"] = rebuilt[miss]
    log.append(("4b. order_amount filled from product price", int(miss.sum())))

    # 5. delivery_minutes: negative -> missing; completed -> category median; cancelled stays missing
    o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
    neg = o["delivery_minutes"] < 0
    log.append(("5a. negative delivery_minutes set to missing", int(neg.sum())))
    o.loc[neg, "delivery_minutes"] = np.nan
    o["category"] = o["product_id"].map(products.set_index("product_id")["category"])
    completed = o["status"] == "completed"
    cat_median = o.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    fill_mask = completed & o["delivery_minutes"].isna()
    log.append(("5b. completed orders: missing delivery filled with category median", int(fill_mask.sum())))
    o.loc[fill_mask, "delivery_minutes"] = cat_median[fill_mask]
    print("Category median delivery minutes (completed orders):",
          o.loc[completed].groupby("category")["delivery_minutes"].median().to_dict())
    cancelled_missing = int((o["status"].eq("cancelled") & o["delivery_minutes"].isna()).sum())
    log.append(("5c. cancelled orders: delivery_minutes deliberately left missing", cancelled_missing))
    o = o.drop(columns="category")

    for rule, n in log:
        print(f"  {rule:<68} {n:>6,}")
    comp = o["status"] == "completed"
    print(f"\nRows: {len(o):,}  distinct order_id: {o['order_id'].nunique():,}")
    print("Status counts:", o["status"].value_counts().to_dict())
    print("Missing order_amount:", int(o["order_amount"].isna().sum()))
    print("Missing delivery_minutes - completed:", int(o.loc[comp, "delivery_minutes"].isna().sum()),
          "| cancelled:", int(o.loc[~comp, "delivery_minutes"].isna().sum()))
    print("Sum of completed order_amount:", inr(o.loc[comp, "order_amount"].sum()))
    return o, log


# ---------------------------------------------------------------- task 5: store
def store(customers, products, orders):
    header("PART A - TASK 5: STORE CLEAN TABLES")
    c = customers.copy()
    c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
    o = orders.copy()
    o["order_ts"] = o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    c.to_csv(f"{CLEAN}/customers.csv", index=False)
    products.to_csv(f"{CLEAN}/products.csv", index=False)
    o.to_csv(f"{CLEAN}/orders.csv", index=False)
    with sqlite3.connect(DB_PATH) as con:
        c.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        o.to_sql("orders", con, if_exists="replace", index=False)
        for t in ("customers", "products", "orders"):
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {DB_PATH} table {t:<10} {n:>6,} rows")
    print(f"  CSV files written to {CLEAN}/")


def main():
    customers, products, orders = load_raw()
    audit(customers, products, orders)
    c, _ = clean_customers(customers)
    o, _ = clean_orders(orders, products)
    store(c, products, o)


if __name__ == "__main__":
    main()
