"""Part A, task 7: pandas cross-check of the SQL results and exploratory statistics.

Run:  python src/part_a_eda.py
"""
import sqlite3

import numpy as np
import pandas as pd

from common import CLEAN, DB_PATH, header, inr
from part_a_sql import load_queries

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main():
    orders = pd.read_csv(f"{CLEAN}/orders.csv")
    customers = pd.read_csv(f"{CLEAN}/customers.csv")
    products = pd.read_csv(f"{CLEAN}/products.csv")
    comp = (orders[orders["status"] == "completed"]
            .merge(customers[["customer_id", "city"]], on="customer_id")
            .merge(products[["product_id", "category"]], on="product_id"))

    header("PART A - TASK 7a: PANDAS CROSS-CHECK OF Q2")
    pivot = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total")
    print(pivot.round(2).to_string())
    with sqlite3.connect(DB_PATH) as con:
        q2 = pd.read_sql_query(load_queries()["Q2"], con)
    pandas_total = round(pivot.loc["Total", "Total"], 2)
    sql_total = round(q2["revenue"].sum(), 2)
    print(f"\npivot_table grand total = {inr(pandas_total)} | SQL Q2 total = {inr(sql_total)} | "
          f"equal: {abs(pandas_total - sql_total) < 0.005}")
    assert abs(pandas_total - sql_total) < 0.005

    by_city = comp.groupby("city").agg(total=("order_amount", "sum"),
                                       mean=("order_amount", "mean"),
                                       orders=("order_amount", "count")).round(2)
    print("\nCompleted order_amount per city (groupby + named aggregation):")
    print(by_city.sort_values("total", ascending=False).to_string())

    header("PART A - TASK 7b: DISTRIBUTION OF COMPLETED order_amount")
    a = comp["order_amount"]
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = {
        "count": len(a), "mean": a.mean(), "median": a.median(), "skewness": a.skew(),
        "min": a.min(), "Q1 (25%)": q1, "Q3 (75%)": q3, "IQR": iqr, "max": a.max(),
        "lower fence (Q1 - 1.5 IQR)": lo, "upper fence (Q3 + 1.5 IQR)": hi,
        "outliers below lower fence": int((a < lo).sum()),
        "outliers above upper fence": int((a > hi).sum()),
    }
    for k, v in stats.items():
        print(f"  {k:<30} {v:>12,.2f}" if isinstance(v, (float, np.floating)) else f"  {k:<30} {v:>12,}")

    above = comp[comp["order_amount"] > hi]
    print("\nCategory mix of the outliers above the upper fence:",
          above["category"].value_counts().to_dict())

    cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = comp[cols].corr()
    print("\nCorrelation matrix (completed orders):")
    print(corr.round(3).to_string())
    pairs = [(corr.loc[x, y], x, y) for i, x in enumerate(cols) for y in cols[i + 1:]]
    pairs.sort(key=lambda t: -abs(t[0]))
    print("\nPairs ranked by |r|:")
    for r, x, y in pairs:
        print(f"  {x:<16} vs {y:<16} r = {r:+.3f}")

    print("\nCompleted revenue by category:",
          comp.groupby("category")["order_amount"].sum().round(2).sort_values(ascending=False).to_dict())
    print("Median order_amount by discount_pct:",
          comp.groupby("discount_pct")["order_amount"].median().round(2).to_dict())


if __name__ == "__main__":
    main()
