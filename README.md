# ❤️ Federated Learning for Heart Disease Prediction

A privacy-preserving machine learning system that predicts cardiovascular disease using **Federated Learning** with a novel **3-Factor Weighted FedAvg** algorithm.

---

## 🎯 Project Overview

This project demonstrates how multiple hospitals can collaborate to build a powerful AI model **without sharing sensitive patient data**. Using federated learning, each hospital trains locally on its own data, and only model weights are shared — ensuring complete privacy preservation.

### Key Features

- ✅ **Privacy-Preserving**: Patient data never leaves hospital premises
- ✅ **Novel Algorithm**: 3-Factor Weighted FedAvg with balance awareness
- ✅ **Real-World Simulation**: 5 hospitals with severe non-IID data (20-80% heterogeneity)
- ✅ **Production-Ready**: Built with Flower framework and PyTorch
- ✅ **Smooth Convergence**: Optimized hyperparameters for stable training
- ✅ **Interactive UI**: Streamlit web interface for predictions

---

## 📊 Results

| Metric | Value | Comparison |
|--------|-------|------------|
| **Accuracy** | 73.0% | Framingham Risk Score: 72% ✅ |
| **Precision** | 74.4% | Good for screening applications |
| **Recall** | 71.1% | Catches 71% of disease cases |
| **F1-Score** | 69.0% | Balanced performance |
| **ROC-AUC** | 80.3% | Good discrimination ability ✅ |

**Privacy Cost**: Only ~7% accuracy reduction compared to centralized training, while achieving **zero data sharing**!

---

## 🔬 Novel Contribution

### 3-Factor Weighted FedAvg Algorithm

Traditional FedAvg uses simple size-based weighting:
```
weight = dataset_size / total_size
```

**Our novel approach** uses three factors:

```python
weight = size × (1 - |ratio - 0.5| × 0.5) × (1 - non_iid_degree)
         ↑            ↑                           ↑
    Factor 1      Factor 2                   Factor 3
    (Size)    (Balance Awareness)    (Non-IID Penalty)
              [NOVEL!]                [NOVEL!]
```

**Factor 1 (Size)**: Standard - larger datasets get more weight  
**Factor 2 (Balance Awareness)**: NEW - rewards clients with balanced data (near 50-50 distribution)  
**Factor 3 (Non-IID Penalty)**: NEW - uses Jensen-Shannon Divergence to penalize extreme heterogeneity  

This approach better handles severe non-IID scenarios common in real-world healthcare data.

---

## 🏥 Dataset

**Source**: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

**Size**: 70,000 patient records  
**Features**: 11 clinical features (age, gender, BP, cholesterol, glucose, lifestyle)  
**Target**: Binary classification (disease / no disease)

### Non-IID Data Distribution Across Hospitals

| Hospital | Disease Ratio | Dataset Size | Non-IID Degree |
|----------|---------------|--------------|----------------|
| **Urban Cardiac Center** | 70% | 17,500 | High |
| **Rural Primary Care** | 20% | 14,000 | Very High |
| **Heart Institute** | 80% | 10,500 | Very High |
| **General Hospital** | 50% | 17,500 | Low (Balanced) |
| **Preventive Clinic** | 30% | 10,500 | High |

This **severe heterogeneity (20-80%)** makes the problem challenging and realistic.

---

## 🛠️ Technical Stack

### Frameworks & Libraries
- **Federated Learning**: Flower 1.26.1
- **Deep Learning**: PyTorch 2.x
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **UI**: Streamlit

### Model Architecture

Simple 4-layer neural network optimized for convergence:


Input (11 features)
    ↓
Layer 1: 11 → 64 (BatchNorm, ReLU, Dropout 0.2)
    ↓
Layer 2: 64 → 32 (BatchNorm, ReLU, Dropout 0.15)
    ↓
Layer 3: 32 → 16 (ReLU)
    ↓
Layer 4: 16 → 1 (Sigmoid)
    ↓
Output (probability)


**Total Parameters**: ~5,000 (lightweight, efficient)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- conda or pip

### Step 1: Install Dependencies
bash
pip install torch flwr numpy pandas scikit-learn matplotlib seaborn scipy streamlit


### Step 2: Download Dataset
1. Download cardio_train.csv from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
2. Place it in the project directory

---
**Output files**:
- final_fl_model.pth` - Trained model
- scaler.pkl` - Feature scaler
- fl_final_results.png` - Visualization

**Training time**: ~8-10 minutes

### Running the Prediction UI
#bash
streamlit run fl_ui_fixed.py


Opens at `http://localhost:8501`

---

## 📈 How It Works

### Federated Learning Process

Hospital 1: [Local Training] → Send Weights ┐
Hospital 2: [Local Training] → Send Weights ├→ Aggregate → Global Model
Hospital 3: [Local Training] → Send Weights ┘

**Privacy Benefit**: Patient data stays local, only 50KB of weights transmitted!

---

## 🎯 Use Cases

### ✅ Suitable For:
- Screening tool for high-risk patients
- Risk stratification
- Resource allocation
- Privacy-preserving research

### ❌ NOT Suitable For:
- Definitive diagnosis (requires >90% accuracy)
- Treatment decisions
- Emergency triage

---

## 🔮 Future Improvements

1. **FedProx Algorithm**: +5-8% accuracy
2. **Differential Privacy**: Formal privacy guarantees
3. **More Clients**: Scale to 10-20 hospitals
4. **Multimodal Data**: ECG + images

**Expected**: 80-85% accuracy with improvements

---

## 📚 Key References

1. **FedAvg** (McMahan et al., 2017) - Foundation
2. **Non-IID Weighted FedAvg** (2024) - Most similar (87.3%)
3. **Jensen-Shannon Divergence** - Non-IID estimation

**Our Contribution**: Novel balance-aware weighting formula

---

## ⚖️ Disclaimer

⚠️ **For educational and research purposes only**  
⚠️ **NOT FDA-approved**  
⚠️ **Requires physician oversight**  

This is a screening tool, not a diagnostic tool.

---

## 👤 Author

**Aparna**  
B.Tech Computer Science, College of Engineering Attingal (2026)

---

## 📊 Quick Start

bash
 1. Install dependencies
pip install torch flwr numpy pandas scikit-learn matplotlib seaborn scipy streamlit

# 2. Train model
python fl_final_converging_3factor.py

# 3. Run UI
streamlit run fl_ui_fixed.py
