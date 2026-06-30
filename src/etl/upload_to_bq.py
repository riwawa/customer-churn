from unittest import result
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import logging
from src.etl.validate import validate_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_raw_data(
    csv_path: str,
    project_id: str,
    dataset: str = "churn_project",
    table: str = "raw_customers"
) -> None:
    """
    Carrega CSV para camada Raw do BigQuery.
    """
    client = bigquery.Client(project=project_id)
    
    # Lê o CSV
    df = pd.read_csv(csv_path)

    result = validate_dataset(df)
    if not result["passed"]:
        raise ValueError(f"Dataset inválido: {result['issues']}")
    
    logger.info(f"Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas")
    
    # Adiciona metadados de ingestão 
    df['_ingestion_ts'] = pd.Timestamp.now(tz='UTC')
    df['_source_file'] = Path(csv_path).name
    
    table_id = f"{project_id}.{dataset}.{table}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  
        autodetect=True,                    
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  
    
    logger.info(f"{len(df)} linhas carregadas em {table_id}")

if __name__ == "__main__":
    upload_raw_data(
        csv_path="data/raw/telco_churn.csv",
        project_id="customer-churn-analysis-500315"
    )