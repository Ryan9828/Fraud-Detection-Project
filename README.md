# Fraud-Detection-Project
This project builds and deploys a real-time credit card fraud detection system using machine learning and FastAPI. The goal is to identify potentially fraudulent transactions based on behavioral and transactional features such as amount, time since last transaction, and merchant category.

A Random Forest model (with threshold optimization for imbalanced data) was trained and evaluated on labeled transaction data, achieving strong performance across key metrics including precision, recall, and ROC-AUC.

The trained model was then containerized with Docker and deployed on an AWS EC2 instance using a FastAPI service (service_raw.py). The API exposes two main endpoints — /predict for fraud inference and /health for server status — enabling scalable, production-style inference.

This end-to-end workflow demonstrates not only technical modeling ability but also the ability to operationalize machine learning models in a real-world environment.
Fraud Detection with LSTM — End-to-End AWS Deployment

# Key Features

Deep learning architecture (LSTM) trained on 1.8 M+ transactions

Real-time fraud detection API using FastAPI

Containerised with Docker for reproducibility and portability

Deployed on AWS EC2 (Amazon Linux 2023, t3.micro)

Business-driven threshold tuning — optimised for financial cost, not just statistical accuracy

Model interpretability via engineered temporal features:
amt, trans_hour, time_since_last, last_amt, category

# Technical Stack
Component	Technology
Model	TensorFlow 2.20 / Keras 3.11 (LSTM network)
Framework	FastAPI 0.119
Preprocessing	scikit-learn 1.6 + joblib
Containerisation	Docker
Cloud Deployment	AWS EC2 (Amazon Linux 2023)
Programming Language	Python 3.12
Package Management	pip


# Deployment Workflow

## 1. Model Training & Evaluation

Dataset: Kaggle Fraud Detection

Engineered temporal features such as time_since_last and last_amt

Final LSTM achieved F1 = 0.97, Precision = 0.98, Recall = 0.95 at threshold t = 0.28 

Fraud Detection Report

## 2. Cost-Based Threshold Optimisation

Financial cost considered for both merchants & banks

Optimal threshold t = 0.011 minimised expected loss to $ 0.17 per transaction 

Fraud Detection Report

## 3. Containerisation

FastAPI service wrapped in a Docker image

Dependencies pinned for reproducibility:

tensorflow==2.20.0
keras==3.11.3
fastapi==0.119.0
uvicorn==0.37.0
scikit-learn==1.6.1
numpy==2.1.3
pandas==2.2.3
joblib==1.4.2

## 4. AWS EC2 Deployment

Instance: t3.micro ( Sydney region )

Ports 80 (HTTP) and 22 (SSH) enabled

SSH → transfer project → docker build → docker run -d -p 80:8000 fastapi-demo

Health check:

curl http://<EC2-IP>/health
 → {"status":"ok","timesteps":32,"n_features":17,"threshold":0.011}

## 5. API Usage
Example Request (JSON)
{
  "transactions": [
    {
      "cc_num": "6011477612335392",
      "trans_date_trans_time": "2025-09-22T21:50:00Z",
      "amt": 3.60,
      "trans_hour": 21,
      "time_since_last": 1626,
      "last_amt": 42.24,
      "category": "home"
    }
  ]
}

Example Response
{
  "id": "6011477612335392",
  "proba_fraud": 0.0047,
  "decision": 0,
  "threshold": 0.011
}

## 6. Swagger UI

Access at:

http://<EC2-IP>/docs


Swagger UI provides interactive testing for /predict and /health endpoints, allowing real-time validation of model outputs .

# Local Testing
Note: The public AWS endpoint is only live when the EC2 instance is running.
For local testing, start the FastAPI app with uvicorn service_raw:app --reload
You can test predictions without AWS using:

uvicorn service_raw:app --reload


Then open http://127.0.0.1:8000/docs

Or run:

python test_api.py

# Results Summary
Metric	Score
Precision	0.98
Recall	0.95
F1 Score	0.97
ROC-AUC	0.9993
PR-AUC	0.9865
Optimal Threshold (t★)	0.011
Avg Cost per Tx	$ 0.17
☁️ Screenshots (Portfolio Evidence)

# Conclusion

This project delivers a complete end-to-end MLOps workflow for fraud detection:

From data analysis and temporal feature engineering

To deep learning model development, cost-based optimisation, and cloud deployment

It demonstrates how to bridge data science, software engineering, and business context to build production-ready AI systems that are both accurate and financially effective.

# References

Ali et al. (2022). Financial Fraud Detection Based on Machine Learning: A Systematic Literature Review. Applied Sciences, 12(19), 9637.

LexisNexis Risk Solutions (2024). True Cost of Fraud Study.

J.P. Morgan (2023). False Positives & Fraud Prevention Tools.

Riskified (2025). How Much Does a False Decline Cost Your Business?

Australian Bureau of Statistics (2025). Personal Fraud, 2023-24 Financial Year.
