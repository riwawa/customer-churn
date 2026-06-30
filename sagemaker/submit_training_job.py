import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# ===== SETUP DA SESSÃO =====
# Em produção real, isso conecta automaticamente com sua conta AWS
# Aqui mostramos a estrutura completa, mesmo sem executar

boto_session = boto3.Session(region_name="us-east-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# IAM Role — a "permissão" que o SageMaker usa para acessar S3, logs, etc
# Em produção: criada uma vez no console AWS, reutilizada sempre
ROLE = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

BUCKET = "churn-mlops-sagemaker"

# ===== DEFINIÇÃO DO ESTIMATOR =====
estimator = SKLearn(
    entry_point="train.py",
    source_dir="sagemaker",
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,

    # Hiperparâmetros — viram argumentos de linha de comando no train.py
    hyperparameters={
        "n_estimators": 200,
        "test_size": 0.2,
        "random_state": 42
    },

    # Onde salvar métricas customizadas — o SageMaker faz parsing via regex
    metric_definitions=[
        {"Name": "roc_auc", "Regex": '"roc_auc": ([0-9\\.]+)'},
        {"Name": "f1_churn", "Regex": '"f1_churn": ([0-9\\.]+)'}
    ],

    # Output explícito — onde o modelo treinado vai parar no S3
    output_path=f"s3://{BUCKET}/output",

    # Tags — organização de custos, essencial em times grandes
    tags=[
        {"Key": "project", "Value": "churn-prediction"},
        {"Key": "environment", "Value": "dev"}
    ]
)

# ===== SUBMISSÃO DO JOB =====
if __name__ == "__main__":
    estimator.fit(
        inputs={"train": f"s3://{BUCKET}/data/raw"},
        job_name="churn-training-v1",
        wait=True,       # bloqueia até terminar (False = assíncrono)
        logs=True         # mostra os logs do container em tempo real
    )

    print(f"Modelo salvo em: {estimator.model_data}")
