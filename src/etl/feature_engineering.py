import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Pandas 2.0+: sem inplace em colunas
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'TechSupport']
    df['num_services'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    # include_lowest=True garante que tenure=0 entra no bin [0,12]
    df['tenure_segment'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 60, float('inf')],
        labels=['new', 'developing', 'established', 'loyal'],
        include_lowest=True
    )
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ['customerID', 'Churn']
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [c for c in categorical_cols if c not in drop_cols]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df
