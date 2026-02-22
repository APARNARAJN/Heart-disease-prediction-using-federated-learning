#Federated Learning for Heart Disease Prediction (Flower + PyTorch)

A privacy-preserving Federated Learning system for cardiovascular disease prediction using Flower and PyTorch under Non-IID client distributions.

📌 Project Overview

This project simulates a real-world healthcare scenario where multiple hospitals collaboratively train a heart disease prediction model without sharing raw patient data.

Each hospital:

Trains locally on private data

Shares only model weights

Receives aggregated global model

A custom Weighted Federated Averaging strategy is implemented to handle Non-IID data distributions.

🏥 Simulation Setup

5 simulated hospital clients

Non-IID data distribution (20%–80% disease ratio)

Custom aggregation strategy

15 federated training rounds

🧠 Model Architecture

Fully Connected Neural Network

Batch Normalization

Dropout Regularization

Binary Classification (Sigmoid Output)

⚙️ Technologies Used

Python

PyTorch

Flower (Federated Learning Framework)

Scikit-learn

NumPy

Pandas

Matplotlib

Seaborn

🔬 Federated Learning Strategy

Custom Weighted FedAvg:

Weight formula:

Weight = dataset_size × (1 − |disease_ratio − 0.5| × 0.5)

This:

Gives higher weight to larger datasets

Penalizes highly imbalanced clients

Improves fairness across hospitals

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

📈 Final Results

Example Output:

F1-Score: ~0.xx

ROC-AUC: ~0.xx

Accuracy: ~0.xx

(Global performance after 15 federated rounds)

🔐 Privacy Features

No raw data sharing

Only model weights transmitted

Simulation of real-world hospital heterogeneity

🚀 How to Run

Install dependencies:

pip install -r requirements.txt

Run the script:

python fl_heart_disease.py
📂 Outputs Generated

global_flower_model.pth

flower_metrics.pkl

scaler.pkl

flower_pytorch_fl_results.png

📚 Future Improvements

Differential Privacy integration

Secure Aggregation

Centralized baseline comparison

Real distributed deployment

👩‍💻 Author

Aparna Raj N
B.Tech Final Year Project
Privacy-Preserving Heart Disease Prediction using Federated Learning
