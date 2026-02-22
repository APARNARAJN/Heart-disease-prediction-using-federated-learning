Federated Learning for Heart Disease Prediction
A robust implementation of Federated Learning (FL) using the Flower framework and PyTorch. This project simulates a decentralized medical environment where five distinct healthcare institutions collaborate to train a global cardiovascular disease prediction model without ever sharing sensitive patient data.
🔬 Project Overview
In traditional Machine Learning, data is centralized. In this project, we utilize Federated Learning to solve privacy and data-silo issues. The simulation focuses on the Non-IID (Independent and Identically Distributed) data challenge, mimicking real-world scenarios where different hospitals have varying patient demographics and disease prevalence.
Key Technical Features
1.Decentralized Training: Clients (hospitals) train locally and only share model parameters (weights).
2.Weighted Aggregation: A custom WeightedFedAvg strategy that balances contributions based on client dataset size and class distribution.
3.Deep Learning Architecture: A PyTorch MLP with Batch Normalization, Dropout, and Adam optimization.
4.Comprehensive Evaluation: Global monitoring of F1-Score, ROC-AUC, and Accuracy across 15 rounds of federation.
🛠️ System Architecture1. 
Model Configuration (HeartDiseaseNet)
The model is a Deep Neural Network designed for binary classification with the following architecture:
Input Layer: 11 Clinical Features (Age, Gender, Cholesterol, etc.)
Hidden Layers: 64 → 32 → 16 units.
Regularization: Dropout (0.3) and BatchNorm1d to ensure generalization across different client data distributions.
Activation: ReLU for hidden layers and Sigmoid for the final output.
2. Federated Strategy
We use a Custom Weighted Federated Averaging strategy. Unlike standard FedAvg, this implementation applies a penalty factor to clients with highly skewed data to prevent the global model from drifting toward a specific hospital's bias.
📊 Client Distribution ProfileThe simulation utilizes 5 heterogeneous clients with specific data profiles
🚀 Getting Started
Prerequisites : Python 3.9+PyTorch 2.1.0+Flower (flwr) 1.6.0+InstallationBash# Clone the repository
git clone https://github.com/your-username/flower-heart-disease.git
cd flower-heart-disease
# Install required packages
pip install torch flwr scikit-learn pandas matplotlib seaborn
ExecutionRun the full federated simulation including data partitioning, training, and visualization:Bashpython fl_heart_disease.py
Data Setup:
Ensure the cardio_train.csv file is present in the root directory.
📈 Results & Visualizations
Upon completion, the system generates flower_pytorch_fl_results.png, which includes:
Global Metric Trends: Tracking the convergence of F1-Score and AUC.
Per-Client Performance: Comparative analysis of how the global model performs on each hospital's local test set.
Confusion Matrix: Final performance breakdown for the aggregated global model.
📂 Repository Structure
fl_heart_disease.py: Core logic for simulation, model definition, and FL strategy.
global_flower_model.pth: Final trained global model weights.
scaler.pkl: The global StandardScaler used for data normalization.
flower_metrics.pkl: Serialized training history for further research.
