# Fraud-Detection-Project
This project builds and deploys a real-time credit card fraud detection system using machine learning and FastAPI. The goal is to identify potentially fraudulent transactions based on behavioral and transactional features such as amount, time since last transaction, and merchant category.

A Random Forest model (with threshold optimization for imbalanced data) was trained and evaluated on labeled transaction data, achieving strong performance across key metrics including precision, recall, and ROC-AUC.

The trained model was then containerized with Docker and deployed on an AWS EC2 instance using a FastAPI service (service_raw.py). The API exposes two main endpoints — /predict for fraud inference and /health for server status — enabling scalable, production-style inference.

This end-to-end workflow demonstrates not only technical modeling ability but also the ability to operationalize machine learning models in a real-world environment.
