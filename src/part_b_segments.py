"""Part B, tasks 1-3: RFM table, K-means model selection, segment profiles, names and PCA.
Writes outputs/customer_segments.csv (read later by Part B task 8 and by Part C routing).

Run:  python src/part_b_segments.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import CLEAN, OUT, header

pd.set_option("display.width", 160)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
RFM_COLS = ["recency", "frequency", "monetary"]


def build_rfm():
    o = pd.read_csv(f"{CLEAN}/orders.csv", parse_dates=["order_ts"])
    comp = o[o["status"] == "completed"]
    rfm = comp.groupby("customer_id").agg(last_ts=("order_ts", "max"),
                                          frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum"))
    rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[RFM_COLS].reset_index()


def name_segments(profile):
    """Business names chosen from the profile (cluster numbers themselves carry no order).
    Highest mean monetary -> Premium; of the rest, highest frequency -> Loyal Regulars;
    highest recency -> Drifting Away; the remaining segment -> Occasional Shoppers."""
    names, left = {}, list(profile.index)
    premium = profile["monetary"].idxmax(); names[premium] = "Premium Big Spenders"; left.remove(premium)
    loyal = profile.loc[left, "frequency"].idxmax(); names[loyal] = "Loyal Regulars"; left.remove(loyal)
    drifting = profile.loc[left, "recency"].idxmax(); names[drifting] = "Drifting Away"; left.remove(drifting)
    for s in left:
        names[s] = "Occasional Shoppers"
    return names, premium


ACTIONS = {
    "Premium Big Spenders": "Priority support plus early access to new electronics and personal-care launches; protect "
                            "this revenue with a personal win-back call if they go quiet.",
    "Loyal Regulars": "Grocery/dairy subscription or loyalty points to keep the weekly habit; do not waste "
                      "discounts on them.",
    "Drifting Away": "Time-limited win-back coupon plus a 'what went wrong' survey before they are lost.",
    "Occasional Shoppers": "Nudge to a second category with a small free-delivery threshold offer to raise frequency.",
}


def main():
    header("PART B - TASK 1: RFM TABLE (snapshot 2025-10-01 00:00:00, completed orders only)")
    rfm = build_rfm()
    print(f"customers with >= 1 completed order: {len(rfm):,}")
    print(rfm[RFM_COLS].describe().round(2).to_string())
    print(f"recency median {rfm['recency'].median():.0f}, min {rfm['recency'].min()}, max {rfm['recency'].max()}; "
          f"frequency mean {rfm['frequency'].mean():.2f}, max {rfm['frequency'].max()}; "
          f"monetary mean {rfm['monetary'].mean():,.2f}, max {rfm['monetary'].max():,.2f}")

    header("PART B - TASK 2: K-MEANS, K = 2..8 (StandardScaler, n_init=10, random_state=42)")
    X = StandardScaler().fit_transform(rfm[RFM_COLS])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "inertia": round(km.inertia_, 1),
                     "silhouette": round(silhouette_score(X, km.labels_), 3)})
    sweep = pd.DataFrame(rows)
    sweep["inertia_drop_vs_prev"] = (-sweep["inertia"].diff()).round(1)
    print(sweep.to_string(index=False))
    print(f"chosen K = {FINAL_K}")

    header(f"PART B - TASK 3: SEGMENT PROFILES FOR K = {FINAL_K}")
    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean"))
    names, premium = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)
    profile["is_premium"] = profile.index == premium
    profile["share_of_revenue_pct"] = (rfm.groupby("segment_id")["monetary"].sum()
                                       / rfm["monetary"].sum() * 100).round(1)
    print(profile.round(2).to_string())
    print("\nRecommended actions:")
    for s, n in names.items():
        print(f"  segment {s} - {n}: {ACTIONS[n]}")

    pca = PCA(n_components=2).fit(X)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: PC1 {evr[0]:.3f}, PC2 {evr[1]:.3f}, total {evr.sum():.3f}")
    print("PCA loadings (rows = components, columns = scaled R, F, M):")
    print(pd.DataFrame(pca.components_, columns=RFM_COLS, index=["PC1", "PC2"]).round(3).to_string())

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
        .to_csv(f"{OUT}/customer_segments.csv", index=False)
    print(f"\nwrote {OUT}/customer_segments.csv ({len(rfm)} rows; premium customers = {rfm['is_premium'].sum()})")


if __name__ == "__main__":
    main()
