import pandas as pd
import logging

logger = logging.getLogger(__name__)

REQUIRED_COLS = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']

def validate_dataset(df: pd.DataFrame) -> dict:
    issues = []

    for col in REQUIRED_COLS:
        if col not in df.columns:
            issues.append(f"> Coluna ausente: {col}")

    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.2].index.tolist()
    if high_null:
        issues.append(f"> Colunas com >20% nulos: {high_null}")

    if 'tenure' in df.columns:
        neg = (pd.to_numeric(df['tenure'], errors='coerce') < 0).sum()
        if neg > 0:
            issues.append(f"> Tenure negativo: {neg} linhas")

    if 'Churn' in df.columns:
        churn_rate = (df['Churn'] == 'Yes').mean()
        if churn_rate < 0.05 or churn_rate > 0.8:
            issues.append(f"> Churn rate anormal: {churn_rate:.1%}")

    dupes = df.duplicated(subset='customerID').sum()
    if dupes > 0:
        issues.append(f"> customerIDs duplicados: {dupes}")

    if issues:
        for i in issues:
            logger.warning(i)
    else:
        logger.info("{!!} Dataset passou em todas as validações")

    return {"passed": len(issues) == 0, "issues": issues}
