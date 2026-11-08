"""Shared paths and helpers for every Part. Importing this module makes the repository root
the working directory and puts it on sys.path, so `import mock_llm` works from any script."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RAW = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
POLICIES = os.path.join("data", "policies")
DB_PATH = os.path.join("data", "instamart.db")
OUT = "outputs"
STATE = "state"
SQL_FILE = os.path.join("sql", "queries.sql")

for _d in (CLEAN, OUT, STATE):
    os.makedirs(_d, exist_ok=True)


def header(title):
    """Print a visible section header in the console output."""
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def inr(x):
    """Format a number as an INR amount with thousands separators."""
    return f"INR {x:,.2f}"
