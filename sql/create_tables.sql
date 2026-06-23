-- Dataset principal
CREATE SCHEMA IF NOT EXISTS `projeto.churn_project`
OPTIONS (location = 'US');

-- =================== RAW LAYER ===================
-- Nunca altere dados aqui. Só append/replace.
CREATE OR REPLACE TABLE `churn_project.raw_customers` (
  customerID       STRING,
  gender           STRING,
  SeniorCitizen    INT64,
  tenure           INT64,
  PhoneService     STRING,
  Contract         STRING,
  MonthlyCharges   FLOAT64,
  TotalCharges     STRING,  -- STRING pois tem espaços no raw
  Churn            STRING,
  _ingestion_ts    TIMESTAMP,
  _source_file     STRING
);

-- =================== CLEAN LAYER ===================
CREATE OR REPLACE VIEW `churn_project.clean_customers` AS
SELECT
  customerID,
  gender,
  SeniorCitizen,
  CAST(tenure AS INT64) AS tenure,
  SAFE_CAST(TotalCharges AS FLOAT64) AS TotalCharges,
  MonthlyCharges,
  Contract,
  CASE Churn WHEN 'Yes' THEN 1 ELSE 0 END AS churn_flag,
  _ingestion_ts
FROM `churn_project.raw_customers`
WHERE customerID IS NOT NULL
  AND tenure IS NOT NULL;

-- =================== FEATURES LAYER ===================
CREATE OR REPLACE VIEW `churn_project.features_customers` AS
SELECT
  *,
  -- Feature 1: razão de cobrança
  SAFE_DIVIDE(MonthlyCharges, COALESCE(TotalCharges, MonthlyCharges)) 
    AS charge_ratio,
  
  -- Feature 2: segmento de tenure
  CASE
    WHEN tenure <= 12  THEN 'new'
    WHEN tenure <= 24  THEN 'developing'
    WHEN tenure <= 60  THEN 'established'
    ELSE 'loyal'
  END AS tenure_segment,
  
  -- Feature 3: contrato mensal (alto risco)
  IF(Contract = 'Month-to-month', 1, 0) AS is_month_to_month,
  
  -- Partição para performance
  DATE(_ingestion_ts) AS partition_date

FROM `churn_project.clean_customers`;