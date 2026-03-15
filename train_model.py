import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

DATASET_PATH = "ecommerce_fraud.csv"
MODEL_PATH = "fraud_pipeline.joblib"


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    required_cols = ["purchase_value", "age", "signup_time", "purchase_time", "class"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
    df = df.dropna(subset=["signup_time", "purchase_time", "class"])

    df["account_age_hours"] = (
        (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
    )
    df["purchase_hour"] = df["purchase_time"].dt.hour
    df["purchase_day"] = df["purchase_time"].dt.day
    df["purchase_month"] = df["purchase_time"].dt.month
    df["purchase_weekday"] = df["purchase_time"].dt.weekday
    df["is_night_transaction"] = df["purchase_hour"].apply(
        lambda x: 1 if x >= 22 or x <= 5 else 0
    )
    df["is_new_account"] = df["account_age_hours"].apply(
        lambda x: 1 if x < 24 else 0
    )

    median_purchase = df["purchase_value"].median()
    df["high_value_transaction"] = df["purchase_value"].apply(
        lambda x: 1 if x > median_purchase else 0
    )

    df = df[df["purchase_value"] >= 0]
    df = df[df["age"] >= 0]
    df = df[df["account_age_hours"] >= 0]

    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df.dropna(subset=["class"])
    df["class"] = df["class"].astype(int)

    feature_cols = [
        "purchase_value",
        "age",
        "account_age_hours",
        "purchase_hour",
        "purchase_day",
        "purchase_month",
        "purchase_weekday",
        "is_night_transaction",
        "is_new_account",
        "high_value_transaction",
    ]

    X = df[feature_cols]
    y = df["class"]

    return X, y, feature_cols, float(median_purchase)


def main():
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded:", df.shape)

    X, y, feature_cols, median_purchase = prepare_features(df)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    try:
        print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    except Exception:
        print("ROC-AUC could not be calculated")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, y_pred))

    artifacts = {
        "model": model,
        "feature_cols": feature_cols,
        "median_purchase_value": median_purchase,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "RandomForest_v2"
    }

    joblib.dump(artifacts, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
