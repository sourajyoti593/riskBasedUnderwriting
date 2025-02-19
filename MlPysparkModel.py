## data Preparation and Processing
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder.appName("UnderwritingRisk").getOrCreate()

# Load data
df = spark.read.csv("https://azmlsa011.blob.core.windows.net/ammlblob01/insurance_data.csv", header=True, inferSchema=True)

# Feature Engineering - Example: Risk Score Calculation
df = df.withColumn("Risk_Score", when(col("Claim_Amount") > 10000, 1).otherwise(0))

# Encode categorical variables (e.g., Insurance Type)
df = df.withColumn("Insurance_Type_Encoded", when(col("Insurance_Type") == "Health", 1)
                    .when(col("Insurance_Type") == "Property", 2)
                    .otherwise(3))


## Train ML Model for risk prediction using PysparkMLlib

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# Prepare feature vector
feature_cols = ["Premium_Amount", "Claim_Amount", "Risk_Score", "Insurance_Type_Encoded"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Define model
rf = RandomForestClassifier(featuresCol="features", labelCol="Fraud_Flag", numTrees=100)

# Enable MLflow tracking
mlflow.set_experiment("Underwriting_Risk_Model")
with mlflow.start_run():
    model = rf.fit(train_df)
    predictions = model.transform(test_df)

    # Log model
    mlflow.spark.log_model(model, "risk_underwriting_model")

    # Evaluate model
    evaluator = BinaryClassificationEvaluator(labelCol="Fraud_Flag", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    mlflow.log_metric("AUC", auc)

    print(f"Model AUC: {auc}")


## Deploy and save
model.save("https://azmlsa011.blob.core.windows.net/ammlblob01/insurance_data.csv")
