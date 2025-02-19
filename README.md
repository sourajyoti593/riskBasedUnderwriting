# Risk-Based Underwriting with PySpark, AWS, and MLflow

## Overview
This project builds an automated **underwriting risk assessment system** using **PySpark**, **AWS**, and **MLflow**. The goal is to evaluate customer risk profiles and make underwriting decisions efficiently.

## Features
- **Data Preprocessing:** Clean and transform raw underwriting data
- **Risk Assessment Model:** Train **Random Forest** on historical insurance data
- **PySpark for Big Data Processing:** Handle large-scale datasets
- **MLflow for Model Tracking:** Log models, hyperparameters, and performance metrics
- **AWS for Storage & Deployment:** Store models and predictions in **S3**

## Installation

### **1️⃣ Set Up Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **2️⃣ Install Dependencies**
```sh
pip install pyspark boto3 mlflow pandas scikit-learn
```

## Data Processing
1. Load underwriting data from **S3 or local storage**
2. Perform **feature engineering** (risk scores, claim history, policy details)
3. Encode categorical variables and normalize numeric features

## Model Training (PySpark + MLflow)
```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import mlflow

# Initialize Spark
spark = SparkSession.builder.appName("UnderwritingRisk").getOrCreate()

# Load data
df = spark.read.csv("s3://your-bucket/insurance_data.csv", header=True, inferSchema=True)

# Feature engineering
assembler = VectorAssembler(inputCols=["Premium_Amount", "Claim_Amount", "Risk_Score"], outputCol="features")
df = assembler.transform(df)

# Train model
rf = RandomForestClassifier(featuresCol="features", labelCol="Fraud_Flag", numTrees=100)
with mlflow.start_run():
    model = rf.fit(df)
    mlflow.spark.log_model(model, "risk_underwriting_model")
```

## Model Deployment
- Store trained model in **S3**
- Deploy via **AWS Lambda / SageMaker** for real-time risk assessment

## Evaluation Metrics
- **Accuracy, Precision, Recall**
- **ROC-AUC Score** for model performance tracking

## Monitoring & Logging
- **MLflow** keeps track of model versions and performance
- **AWS CloudWatch** can be used for monitoring real-time predictions

## License
This project is licensed under the MIT License.



