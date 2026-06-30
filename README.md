# Customer Churn Prediction — MLOps System

Sistema completo de previsão de churn de clientes, construído com práticas reais de MLOps: do dado bruto em CSV até deploy em produção na nuvem, passando por versionamento de dados, experiment tracking, containerização e orquestração em Kubernetes.

> Projeto desenvolvido como estudo prático de MLOps, cobrindo o ciclo completo de vida de um modelo de Machine Learning em produção.

## Stack

**Dados & ETL:** Python, Pandas, BigQuery, SQL (Medallion Architecture)
**Modelagem:** Scikit-Learn (Random Forest, Logistic Regression)
**Versionamento:** Git, DVC (dados e modelos), MLflow (experimentos)
**Serving:** FastAPI
**Infraestrutura:** Docker, Kubernetes, Google Cloud Run, Artifact Registry
**Visualização:** Looker Studio

## Arquitetura

```
CSV → BigQuery (raw) → ETL/Feature Engineering → BigQuery (clean/features)
                                    ↓
                          Treino (RandomForest + MLflow)
                                    ↓
                    Docker → Artifact Registry → Cloud Run / Kubernetes
                                    ↓
                          FastAPI (/predict, /health)
                                    ↓
                    BigQuery (predictions) → Looker Studio
```

## O problema

Prever quais clientes têm maior probabilidade de cancelar um serviço, permitindo que o time de Customer Success priorize ações de retenção nos clientes de maior risco — em vez de tratar toda a base igualmente.

## Resultados do modelo

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.824 |
| Avg Precision (PR-AUC) | 0.613 |
| F1-Score (classe Churn) | 0.596 |
| Recall (classe Churn) | 64% |

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.043 clientes, 26% de taxa de churn.

## Estrutura do projeto

```
customer-churn-mlops/
├── data/
│   ├── raw/                  # dataset original
│   └── processed/            # features geradas pelo pipeline
├── sql/
│   └── create_tables.sql     # Medallion Architecture no BigQuery
├── src/
│   ├── etl/
│   │   ├── validate.py            # validação de qualidade de dados
│   │   ├── feature_engineering.py # clean → features → encode
│   │   └── upload_to_bq.py        # ingestão para BigQuery
│   ├── training/
│   │   └── train.py               # treino + MLflow tracking
│   └── api/
│       └── main.py                # FastAPI — /predict, /health
├── sagemaker/
│   ├── train.py                   # script adaptado para AWS SageMaker
│   └── submit_training_job.py     # configuração do training job
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml                   # autoscaling
├── models/                    # artefatos versionados via DVC
├── dvc.yaml                   # pipeline reproduzível
├── params.yaml                 # hiperparâmetros versionados
├── Dockerfile
└── requirements.txt
```

## Como rodar localmente

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduz o pipeline completo (ETL + treino) via DVC
dvc pull        # baixa dataset e modelo versionados
dvc repro        # reproduz o pipeline do zero

# Sobe a API
uvicorn src.api.main:app --reload --port 8000
```

Testa a API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0, "Contract": "Month-to-month"}'
```

## Decisões técnicas relevantes

**Por que Random Forest e não XGBoost de partida** — baseline com Logistic Regression (AUC 0.847) já superava o Random Forest (AUC 0.823). Em datasets tabulares pequenos, modelos simples competem bem com modelos complexos; a prioridade foi validar a pipeline antes de otimizar o algoritmo.

**Por que ROC-AUC e Avg Precision, não Accuracy** — com 26% de taxa de churn, um modelo que nunca prevê churn atinge 74% de accuracy sem nenhuma utilidade prática. Métricas sensíveis a desbalanceamento foram usadas para avaliação real.

**Por que Medallion Architecture no BigQuery** — separar raw / clean / features em camadas (Views, exceto a tabela final de predictions) garante que dados brutos nunca sejam alterados e que o pipeline seja auditável e reprodutível.

**Por que DVC além do Git** — datasets e modelos não pertencem ao controle de versão de código. O DVC versiona esses artefatos via hash, com armazenamento físico no GCS, garantindo reprodutibilidade exata de qualquer experimento.

## Pipeline reproduzível (DVC)

```bash
dvc dag                 # visualiza o grafo de dependências
dvc repro                # reexecuta apenas o que mudou
dvc metrics diff HEAD~1  # compara métricas entre commits
```

## Deploy

A API está containerizada e pronta para deploy em:
- **Google Cloud Run** (serverless, usado neste projeto)
- **Kubernetes** (manifests em `k8s/`, testado localmente via Minikube com HPA)
- **AWS SageMaker** (script de treino adaptado em `sagemaker/`)

---

