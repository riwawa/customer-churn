import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def clean_data(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df


def create_features(df):
    df = df.copy()
    df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'TechSupport']
    df['num_services'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['tenure_segment'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 60, float('inf')],
        labels=['new', 'developing', 'established', 'loyal'], include_lowest=True
    )
    return df


def encode_features(df):
    drop_cols = ['customerID', 'Churn']
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c not in drop_cols]
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hiperparâmetros — vêm de fora, via SageMaker SDK
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    # Paths — convenção do SageMaker, injetados via variáveis de ambiente
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "models"))
    parser.add_argument("--train_data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/raw"))

    args = parser.parse_args()

    # Lê o CSV de dentro do canal "train" que o SageMaker montou
    csv_path = os.path.join(args.train_data, "telco_churn.csv")
    df = pd.read_csv(csv_path)
    df = clean_data(df)
    df = create_features(df)

    target = df["Churn"].copy()
    df_encoded = encode_features(df)
    feature_cols = [c for c in df_encoded.columns if c not in ["Churn", "customerID"]]
    X = df_encoded[feature_cols]
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    baseline.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "f1_churn": round(f1_score(y_test, y_pred), 4),
        "baseline_auc": round(baseline_auc, 4)
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Salva no model_dir — o SageMaker faz upload automático pro S3 a partir daqui
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "feature_cols": feature_cols, "metrics": metrics},
        os.path.join(args.model_dir, "churn_model.pkl")
    )
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Modelo salvo")
