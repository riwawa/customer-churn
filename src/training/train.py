import pandas as pd
import joblib
import json
import logging
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

def train_model(csv_path: str = "data/raw/telco_churn.csv"):
    df = pd.read_csv(csv_path)
    df = clean_data(df)
    df = create_features(df)
    target = df['Churn'].copy()
    df = encode_features(df)
    feature_cols = [c for c in df.columns if c not in ['Churn', 'customerID']]
    X = df[feature_cols]
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Test:  {y_test.value_counts().to_dict()}")

    baseline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    baseline.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    logger.info(f"Baseline (LogReg) ROC-AUC: {baseline_auc:.4f}")

    params = {"n_estimators": 200, "class_weight": "balanced", "random_state": 42, "threshold": 0.4}

    with mlflow.start_run(run_name="random-forest-v1"):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run iniciado: {run_id}")

        mlflow.log_params(params)
        mlflow.log_param("baseline_auc", round(baseline_auc, 4))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=params["n_estimators"],
                class_weight=params["class_weight"],
                random_state=params["random_state"],
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
            pipeline.named_steps['model'].feature_importances_,
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

        print("\n" + classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        logger.info(f"ROC-AUC:       {metrics['roc_auc']}")
        logger.info(f"Avg Precision: {metrics['avg_precision']}")
        logger.info(f"F1 Churn:      {metrics['f1_churn']}")

        joblib.dump({'pipeline': pipeline, 'feature_cols': feature_cols, 'metrics': metrics}, 'models/churn_model.pkl')
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✅ Run ID: {run_id}")

    return pipeline, metrics

if __name__ == "__main__":
    train_model()
