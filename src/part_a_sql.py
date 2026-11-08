"""Part A, task 6: run the seven queries in sql/queries.sql against data/instamart.db
with pd.read_sql_query and print the stated result of each.

Run:  python src/part_a_sql.py
"""
import re
import sqlite3

import pandas as pd

from common import DB_PATH, SQL_FILE, header, inr

pd.set_option("display.width", 140)


def load_queries(path=SQL_FILE):
    """Split the .sql file on '-- name: Qn' markers -> {"Q1": "SELECT ...", ...}."""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^--\s*name:\s*(Q\d+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def run_all():
    queries = load_queries()
    with sqlite3.connect(DB_PATH) as con:
        print("SQLite version:", sqlite3.sqlite_version)
        res = {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
    return res


def main():
    r = run_all()
    header("PART A - TASK 6: SQL ANALYSIS")

    q1 = r["Q1"]
    print(f"Q1 WHERE + HAVING: {len(q1)} customers have > INR 8000 in completed orders above INR 500")
    print(q1.head(5).to_string(index=False))

    q2 = r["Q2"]
    top = q2.iloc[0]
    print(f"\nQ2 revenue by city x category: {len(q2)} rows; total {inr(q2['revenue'].sum())}; "
          f"largest cell {top['city']} / {top['category']} = {inr(top['revenue'])}")

    q3 = r["Q3"]
    print(f"\nQ3 products above average price: {len(q3)} products")
    print(q3.to_string(index=False))

    q4 = r["Q4"]
    print(f"\nQ4 customers with no completed order: {len(q4)} customers")

    q5 = r["Q5"]
    print(f"\nQ5 customers above average completed spend: {len(q5)} customers; "
          f"average {inr(q5['avg_total_spend'].iloc[0])}")

    q6 = r["Q6"]
    tops = q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]]
    print("\nQ6 top customer per city:")
    print(tops.to_string(index=False))
    diff = q6[q6["rnk"] != q6["row_num"]]
    print(f"Rows where RANK() != ROW_NUMBER(): {len(diff)}")
    print(diff.to_string(index=False))

    q7 = r["Q7"]
    print(f"\nQ7 daily revenue: {len(q7)} daily rows; first day {q7['day'].iloc[0]}, last day {q7['day'].iloc[-1]}; "
          f"last cumulative value {inr(q7['cumulative_revenue'].iloc[-1])}")
    print(q7.tail(3).to_string(index=False))
    return r


if __name__ == "__main__":
    main()
