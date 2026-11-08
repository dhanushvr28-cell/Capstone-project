"""Part B, tasks 4-8: lapse label, leakage-free features, Pipeline preprocessing, tuned models,
class-balance experiment, interpretation, persistence and the illustrative Premium retention list.

Run:  python src/part_b_lapse.py      (needs outputs/customer_segments.csv from part_b_segments.py)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLEAN, OUT, header

pd.set_option("display.width", 160)

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")          # September window: 2025-09-01 .. 2025-09-30
NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate",
           "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]
MODEL_PATH = f"{OUT}/lapse_model.joblib"


# ---------------------------------------------------------------- task 4: cohort, label, features
def build_dataset():
    o = pd.read_csv(f"{CLEAN}/orders.csv", parse_dates=["order_ts"])
    c = pd.read_csv(f"{CLEAN}/customers.csv", parse_dates=["signup_date"])
    p = pd.read_csv(f"{CLEAN}/products.csv")
    o = o.merge(p[["product_id", "category"]], on="product_id", how="left")

    before = o[o["order_ts"] < CUTOFF]                      # the ONLY orders features may see
    sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < LABEL_END)]
    comp_before = before[before["status"] == "completed"]

    f = comp_before.groupby("customer_id").agg(last_ts=("order_ts", "max"),
                                                frequency=("order_id", "nunique"),
                                                monetary=("order_amount", "sum"),
                                                avg_discount_pct=("discount_pct", "mean"),
                                                avg_delivery_minutes=("delivery_minutes", "mean"))
    f["recency"] = (CUTOFF - f["last_ts"]).dt.days
    f = f.drop(columns="last_ts")

    # cancellation rate = cancelled / all orders with a final status (completed or cancelled), before cutoff
    final = before[before["status"].isin(["completed", "cancelled"])]
    f["cancel_rate"] = final.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())

    # favourite category = most frequent category among completed orders (ties -> alphabetical first)
    cat_counts = comp_before.groupby(["customer_id", "category"]).size().reset_index(name="n")
    cat_counts = cat_counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
    f["fav_category"] = cat_counts.drop_duplicates("customer_id").set_index("customer_id")["category"]

    f = f.join(c.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days
    f = f.drop(columns="signup_date")

    active_sept = set(sept.loc[sept["status"] == "completed", "customer_id"])
    f["lapsed"] = (~f.index.isin(active_sept)).astype(int)
    f = f.reset_index()
    assert before["order_ts"].max() < CUTOFF, "an order on/after the cutoff leaked into the features"
    return f


def pos_metrics(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)}


def show_eval(name, y_true, y_pred):
    m = pos_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{name}")
    print(f"  confusion matrix (rows = actual 0/1, cols = predicted 0/1): TN={cm[0,0]} FP={cm[0,1]} "
          f"FN={cm[1,0]} TP={cm[1,1]}")
    print(f"  lapsed class: precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}  "
          f"(accuracy {m['accuracy']:.3f})")
    return m, cm


def make_preprocessor():
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(drop="first"), NOMINAL)])


def main():
    header("PART B - TASK 4: LAPSE PROBLEM (cutoff 2025-09-01)")
    df = build_dataset()
    y = df["lapsed"]
    X = df[NUMERIC + NOMINAL]
    print(f"cohort: {len(df):,} customers with >= 1 completed order before the cutoff")
    print(f"lapsed (no completed order 2025-09-01..2025-09-30): {y.sum()} ({y.mean()*100:.2f}%)")
    print("features:", NUMERIC + NOMINAL)
    print("any missing feature value:", bool(X.isna().any().any()))
    assert not X.isna().any().any(), "a feature column contains a missing value"

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train rows {len(X_tr)}, test rows {len(X_te)}; train lapsed {y_tr.sum()}, test lapsed {y_te.sum()}")
    base = show_eval("Baseline: always predict 'not lapsed' (test set)", y_te, np.zeros(len(y_te), dtype=int))

    header("PART B - TASK 5: PREPROCESSING FITTED ON TRAINING ROWS ONLY")
    pre = make_preprocessor().fit(X_tr)
    scaler = pre.named_transformers_["num"]
    i = NUMERIC.index("frequency")
    print(f"frequency mean learned by the scaler (training rows only): {scaler.mean_[i]:.4f}")
    print(f"frequency mean over the full cohort (train + test):        {X['frequency'].mean():.4f}")
    enc = pre.named_transformers_["cat"]
    print("one-hot columns (k-1 per variable):", list(enc.get_feature_names_out(NOMINAL)))
    print("dropped reference levels:", {v: cats[0] for v, cats in zip(NOMINAL, enc.categories_)})
    print(f"transformed training matrix: {pre.transform(X_tr).shape}; test matrix: {pre.transform(X_te).shape}")

    header("PART B - TASK 6: TUNING WITH GridSearchCV (cv=5, scoring='f1')")
    lr_grid = GridSearchCV(Pipeline([("pre", make_preprocessor()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
                           {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
    lr_grid.fit(X_tr, y_tr)
    print("LogisticRegression CV mean F1 per C:",
          {p["clf__C"]: round(s, 4) for p, s in zip(lr_grid.cv_results_["params"],
                                                     lr_grid.cv_results_["mean_test_score"])})
    print(f"  best params {lr_grid.best_params_}, best CV F1 {lr_grid.best_score_:.4f}")

    rf_grid = GridSearchCV(Pipeline([("pre", make_preprocessor()),
                                     ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                           {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                           cv=5, scoring="f1")
    rf_grid.fit(X_tr, y_tr)
    n_fits = len(rf_grid.cv_results_["params"]) * 5
    print(f"RandomForest grid: {len(rf_grid.cv_results_['params'])} candidates x 5 folds = {n_fits} fits")
    for p, s in zip(rf_grid.cv_results_["params"], rf_grid.cv_results_["mean_test_score"]):
        print(f"  depth {p['clf__max_depth']}, leaf {p['clf__min_samples_leaf']:>2}: CV F1 {s:.4f}")
    print(f"  best params {rf_grid.best_params_}, best CV F1 {rf_grid.best_score_:.4f}")

    print("\n--- test set, used only now that tuning is finished ---")
    lr_best, rf_best = lr_grid.best_estimator_, rf_grid.best_estimator_
    lr_m, _ = show_eval("Tuned LogisticRegression (test set)", y_te, lr_best.predict(X_te))
    rf_m, _ = show_eval("Tuned RandomForest (test set)", y_te, rf_best.predict(X_te))
    selected, selected_name = (rf_best, "RandomForest") if rf_m["f1"] >= lr_m["f1"] else (lr_best, "LogisticRegression")
    print(f"\nselected model: {selected_name}")

    header("PART B - TASK 7: CLASS-BALANCE EXPERIMENT (over-sample lapsed TRAINING rows only)")
    train = X_tr.assign(lapsed=y_tr)
    minority, majority = train[train["lapsed"] == 1], train[train["lapsed"] == 0]
    extra = minority.sample(n=len(majority) - len(minority), replace=True, random_state=42)
    balanced = pd.concat([train, extra])
    print(f"training rows before: {len(train)} ({len(majority)} not lapsed / {len(minority)} lapsed); "
          f"added {len(extra)} duplicates -> {len(balanced)} rows, class counts "
          f"{balanced['lapsed'].value_counts().sort_index().to_dict()}")
    lr_bal = Pipeline([("pre", make_preprocessor()),
                       ("clf", LogisticRegression(max_iter=1000, C=lr_grid.best_params_["clf__C"]))])
    lr_bal.fit(balanced[NUMERIC + NOMINAL], balanced["lapsed"])
    bal_m, _ = show_eval("Balanced LogisticRegression (original, unchanged test set)", y_te, lr_bal.predict(X_te))
    print("\nmetric      unbalanced  balanced  change")
    for k in ("precision", "recall", "f1"):
        print(f"{k:<11} {lr_m[k]:>10.3f} {bal_m[k]:>9.3f} {bal_m[k]-lr_m[k]:>+7.3f}")

    header("PART B - TASK 8a: INTERPRETATION")
    names = rf_best.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    print("Top 5 random-forest feature importances:")
    print(imp.head(5).round(4).to_string())
    coefs = pd.Series(lr_best.named_steps["clf"].coef_[0],
                      index=lr_best.named_steps["pre"].get_feature_names_out()).sort_values()
    print("\nLogistic-regression coefficients (scaled features; + raises lapse odds):")
    print(coefs.round(4).to_string())

    header("PART B - TASK 8b: PERSIST AND RELOAD THE SELECTED MODEL")
    before_pred = selected.predict(X_te)
    joblib.dump(selected, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    after_pred = reloaded.predict(X_te)
    same = bool(np.array_equal(before_pred, after_pred))
    print(f"saved {selected_name} to {MODEL_PATH}; reloaded predictions identical on {len(X_te)} test rows: {same}")
    assert same
    assert np.allclose(selected.predict_proba(X_te), reloaded.predict_proba(X_te))

    header("PART B - TASK 8c: ILLUSTRATIVE RETENTION LIST (Premium customers in the test set)")
    seg = pd.read_csv(f"{OUT}/customer_segments.csv")
    test = df.loc[X_te.index, ["customer_id", "lapsed"]].copy()
    test["lapse_probability"] = reloaded.predict_proba(X_te)[:, 1]
    test = test.merge(seg[["customer_id", "is_premium"]], on="customer_id", how="left")
    prem = test[test["is_premium"] == True]                                     # noqa: E712
    top = (prem.sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
               .rename(columns={"lapsed": "actually_lapsed"}))
    top["lapse_probability"] = top["lapse_probability"].round(4)
    top[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(f"{OUT}/retention_list.csv", index=False)
    print(f"Premium customers in the test set: {len(prem)}; written top {len(top)} to {OUT}/retention_list.csv")
    print(top[["customer_id", "lapse_probability", "actually_lapsed"]].to_string(index=False))
    print(f"truly lapsed among them: {int(top['actually_lapsed'].sum())} of {len(top)}")
    print("NOTE: is_premium comes from whole-window RFM (includes September), so this list is illustrative, "
          "not a leakage-free evaluation.")


if __name__ == "__main__":
    main()
