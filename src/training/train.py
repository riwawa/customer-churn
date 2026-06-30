import pandas as pd
import joblib
import json
import logging
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from src.etl.feature_engineering import clean_data, create_features, encode_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("churn-prediction")

def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def train_model():
    params = load_params()
    data_params = params["data"]
    train_params = params["train"]

    logger.info(f"Params carregados: {params}")

    # Carrega e processa
    df = pd.read_csv(data_params["raw_path"])
    df = clean_data(df)
    df = create_features(df)

    target = df["Churn"].copy()
    df_encoded = encode_features(df)

    feature_cols = [c for c in df_encoded.columns if c not in ["Churn", "customerID"]]
    X = df_encoded[feature_cols]
    y = target

    # Salva features processadas — DVC rastreia esse output
    Path(data_params["processed_path"]).parent.mkdir(parents=True, exist_ok=True)
    df_encoded.assign(Churn=target).to_csv(data_params["processed_path"], index=False)
    logger.info(f"Features salvas em {data_params['processed_path']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
        stratify=y
    )
    logger.info(f"Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Test:  {y_test.value_counts().to_dict()}")

    # Baseline
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    baseline.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    logger.info(f"Baseline ROC-AUC: {baseline_auc:.4f}")

    # Modelo principal — lê hiperparâmetros do params.yaml
    with mlflow.start_run(run_name="random-forest-v1"):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run: {run_id}")

        mlflow.log_params(train_params)
        mlflow.log_params({
            "test_size": data_params["test_size"],
            "baseline_auc": round(baseline_auc, 4)
        })

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=train_params["n_estimators"],
                class_weight=train_params["class_weight"],
                random_state=data_params["random_state"],
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
        mlflow.log_metrics(metrics)

        importances = pd.Series(
            pipeline.named_steps["model"].feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        logger.info(f"\nTop 10 features:\n{importances.head(10).to_string()}")

        Path("models").mkdir(exist_ok=True)
        importances.to_csv("models/feature_importance.csv")
        mlflow.log_artifact("models/feature_importance.csv")

        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name="churn-predictor",
            skops_trusted_types=["numpy.dtype", "numpy.ndarray"]
        )

        print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
        logger.info(f"ROC-AUC:       {metrics['roc_auc']}")
        logger.info(f"Avg Precision: {metrics['avg_precision']}")
        logger.info(f"F1 Churn:      {metrics['f1_churn']}")

        joblib.dump(
            {"pipeline": pipeline, "feature_cols": feature_cols, "metrics": metrics},
            "models/churn_model.pkl"
        )
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✅ Run ID: {run_id}")

    return pipeline, metrics

if __name__ == "__main__":
    train_model()
