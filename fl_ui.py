# ============================================
# FEDERATED LEARNING UI - STREAMLIT
# Heart Disease Prediction Demo
# ============================================
#
# Install: pip install streamlit
# Run:     streamlit run fl_ui.py
#
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import time

# ============================================
# MODEL DEFINITION (same as training)
# ============================================

class HeartDiseaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Federated Learning Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .client-card {
        background: #808080;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown('<div class="main-header">🏥 Federated Learning Demo</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Heart Disease Prediction with Privacy-Preserving FL</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.image("https://raw.githubusercontent.com/adap/flower/main/doc/source/_static/flower-logo.png", width=200)
st.sidebar.markdown("## 🎯 Navigation")

page = st.sidebar.radio(
    "",
    ["🏠 Home", "📊 FL Training Demo", "🔮 Make Prediction", "📈 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**Framework:** Flower (flwr)  
**ML Library:** PyTorch  
**Clients:** 5 Hospitals  
**Data:** Non-IID  
**Privacy:** ✓ Preserved
""")

# ============================================
# LOAD DATA/MODEL IF EXISTS
# ============================================

@st.cache_resource
def load_model():
    try:
        model = HeartDiseaseNet()
        model.load_state_dict(torch.load('global_flower_model.pth', weights_only=True))
        model.eval()
        return model
    except:
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_metrics():
    try:
        with open('flower_metrics.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()
scaler = load_scaler()
metrics = load_metrics()

# ============================================
# PAGE 1: HOME
# ============================================

if page == "🏠 Home":
    st.markdown('<div class="sub-header">Welcome to Federated Learning!</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🏥</h2>
            <h3>5 Hospitals</h3>
            <p>Collaborative Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🔒</h2>
            <h3>Privacy First</h3>
            <p>No Data Sharing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>High Accuracy</h3>
            <p>Better Together</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🤔 What is Federated Learning?")
    
    st.markdown("""
    <div class="info-box">
    Federated Learning (FL) is a machine learning technique that trains models across multiple 
    decentralized devices or servers holding local data samples, <b>without exchanging the raw data</b>.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Benefits")
        st.markdown("""
        - 🔒 **Privacy**: Patient data never leaves hospital
        - 🏥 **Collaboration**: Multiple hospitals work together
        - 📈 **Better Models**: Learn from diverse data
        - ⚡ **Efficient**: No need to centralize data
        - 🌍 **Scalable**: Add more hospitals easily
        """)
    
    with col2:
        st.markdown("### 🔧 How It Works")
        st.markdown("""
        1. **Initialize**: Global model created
        2. **Distribute**: Each hospital gets model copy
        3. **Train Locally**: Hospitals train on their data
        4. **Share Weights**: Only model parameters shared
        5. **Aggregate**: Server combines updates
        6. **Repeat**: Process continues for multiple rounds
        """)
    
    st.markdown("---")
    
    st.markdown("### 🏥 Our 5 Hospital Clients")
    
    clients_info = [
        {"name": "Urban Cardiac Center", "ratio": 0.70, "desc": "Specialized cardiac unit with high disease cases"},
        {"name": "Rural Primary Care", "ratio": 0.20, "desc": "General practice with mostly healthy patients"},
        {"name": "Heart Institute", "ratio": 0.80, "desc": "Heart disease specialists"},
        {"name": "General Hospital", "ratio": 0.50, "desc": "Balanced patient population"},
        {"name": "Preventive Clinic", "ratio": 0.30, "desc": "Prevention-focused healthcare"}
    ]
    
    for i, client in enumerate(clients_info):
        st.markdown(f"""
        <div class="client-card">
            <h4>🏥 {client['name']}</h4>
            <p>{client['desc']}</p>
            <p><b>Disease Ratio:</b> {client['ratio']*100:.0f}% (Non-IID Data)</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE 2: FL TRAINING DEMO
# ============================================

elif page == "📊 FL Training Demo":
    st.markdown('<div class="sub-header">Federated Learning Training Process</div>', unsafe_allow_html=True)
    
    if metrics is None:
        st.warning("⚠️ No training metrics found. Please run `fl_heart_disease.py` first!")
    else:
        # Training progress
        st.markdown("### 📈 Training Progress")
        
        rounds = list(range(1, len(metrics['global']['f1']) + 1))
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Global F1-Score', 'Global ROC-AUC', 
                          'Global Accuracy', 'All Metrics'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # F1 Score
        fig.add_trace(
            go.Scatter(x=rounds, y=metrics['global']['f1'],
                      mode='lines+markers', name='F1-Score',
                      line=dict(color='#e74c3c', width=3)),
            row=1, col=1
        )
        
        # AUC
        fig.add_trace(
            go.Scatter(x=rounds, y=metrics['global']['auc'],
                      mode='lines+markers', name='ROC-AUC',
                      line=dict(color='#2ecc71', width=3)),
            row=1, col=2
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(x=rounds, y=metrics['global']['accuracy'],
                      mode='lines+markers', name='Accuracy',
                      line=dict(color='#3498db', width=3)),
            row=2, col=1
        )
        
        # All metrics
        for metric, color in [('f1', '#e74c3c'), ('auc', '#2ecc71'), ('accuracy', '#3498db')]:
            fig.add_trace(
                go.Scatter(x=rounds, y=metrics['global'][metric],
                          mode='lines+markers', name=metric.upper(),
                          line=dict(color=color, width=2)),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Round", row=2, col=1)
        fig.update_xaxes(title_text="Round", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Final metrics
        st.markdown("---")
        st.markdown("### 🎯 Final Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        final_f1 = metrics['global']['f1'][-1]
        final_auc = metrics['global']['auc'][-1]
        final_acc = metrics['global']['accuracy'][-1]
        final_prec = metrics['global']['precision'][-1]
        
        with col1:
            st.metric("F1-Score", f"{final_f1:.4f}", delta=f"+{final_f1 - metrics['global']['f1'][0]:.4f}")
        with col2:
            st.metric("ROC-AUC", f"{final_auc:.4f}", delta=f"+{final_auc - metrics['global']['auc'][0]:.4f}")
        with col3:
            st.metric("Accuracy", f"{final_acc:.4f}", delta=f"+{final_acc - metrics['global']['accuracy'][0]:.4f}")
        with col4:
            st.metric("Precision", f"{final_prec:.4f}", delta=f"+{final_prec - metrics['global']['precision'][0]:.4f}")
        
        # Per-client performance
        st.markdown("---")
        st.markdown("### 🏥 Per-Client Performance")
        
        client_names = ['Urban Cardiac', 'Rural Primary', 'Heart Institute', 'General Hospital', 'Preventive Clinic']
        client_ids = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4']
        
        client_f1s = [metrics['clients'][cid]['f1'][-1] for cid in client_ids]
        client_aucs = [metrics['clients'][cid]['auc'][-1] for cid in client_ids]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='F1-Score',
            x=client_names,
            y=client_f1s,
            marker_color='#e74c3c'
        ))
        fig2.add_trace(go.Bar(
            name='ROC-AUC',
            x=client_names,
            y=client_aucs,
            marker_color='#2ecc71'
        ))
        
        fig2.update_layout(
            barmode='group',
            title='Final Metrics by Hospital',
            xaxis_title='Hospital',
            yaxis_title='Score',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 3: MAKE PREDICTION
# ============================================

elif page == "🔮 Make Prediction":
    st.markdown('<div class="sub-header">Test the Federated Model</div>', unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.error("❌ Model not found! Please run `fl_heart_disease.py` first to train the model.")
    else:
        st.markdown("""
        <div class="info-box">
        Enter patient information below to predict heart disease risk using our 
        federated learning model trained across 5 hospitals!
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_days = st.number_input("Age (in days)", min_value=10000, max_value=30000, value=20000, step=365)
            gender = st.selectbox("Gender", ["Female (1)", "Male (2)"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            ap_hi = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
            ap_lo = st.number_input("Diastolic BP", min_value=60, max_value=140, value=80)
        
        with col2:
            cholesterol = st.selectbox("Cholesterol", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
            gluc = st.selectbox("Glucose", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
            smoke = st.selectbox("Smoking", ["No (0)", "Yes (1)"])
            alco = st.selectbox("Alcohol", ["No (0)", "Yes (1)"])
            active = st.selectbox("Physical Activity", ["No (0)", "Yes (1)"])
        
        if st.button("🔮 Predict Heart Disease Risk"):
            # Process inputs
            gender_val = int(gender.split("(")[1][0])
            chol_val = int(cholesterol.split("(")[1][0])
            gluc_val = int(gluc.split("(")[1][0])
            smoke_val = int(smoke.split("(")[1][0])
            alco_val = int(alco.split("(")[1][0])
            active_val = int(active.split("(")[1][0])
            
            # Create input array
            input_data = np.array([[
                age_days, gender_val, height, weight, ap_hi, ap_lo,
                chol_val, gluc_val, smoke_val, alco_val, active_val
            ]])
            
            # Scale and predict
            with st.spinner("🔄 Processing with federated model..."):
                time.sleep(1)  # Dramatic effect 😄
                input_scaled = scaler.transform(input_data)
                input_tensor = torch.FloatTensor(input_scaled)
                
                with torch.no_grad():
                    prediction_proba = model(input_tensor).numpy()[0][0]
                    prediction = int(prediction_proba > 0.5)
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h1>⚠️ HIGH RISK</h1>
                        <h2>{prediction_proba*100:.1f}% Probability</h2>
                        <p style="font-size: 1.2rem;">Patient may have cardiovascular disease</p>
                        <p><b>Recommendation:</b> Consult a cardiologist immediately</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h1>✅ LOW RISK</h1>
                        <h2>{(1-prediction_proba)*100:.1f}% Healthy</h2>
                        <p style="font-size: 1.2rem;">Patient appears healthy</p>
                        <p><b>Recommendation:</b> Maintain healthy lifestyle</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show gauge chart
            st.markdown("### 📊 Risk Gauge")
            fig3 = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Disease Risk %"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)

# ============================================
# PAGE 4: ANALYTICS
# ============================================

elif page == "📈 Analytics":
    st.markdown('<div class="sub-header">Federated Learning Analytics</div>', unsafe_allow_html=True)
    
    if metrics is None:
        st.warning("⚠️ No analytics data found. Please run training first!")
    else:
        # System overview
        st.markdown("### 🔍 System Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Rounds", len(metrics['global']['f1']))
        with col2:
            st.metric("Clients", "5")
        with col3:
            st.metric("Framework", "Flower")
        with col4:
            st.metric("ML Library", "PyTorch")
        with col5:
            st.metric("Privacy", "✓ Preserved")
        
        st.markdown("---")
        
        # Client comparison
        st.markdown("### 🏥 Client Performance Comparison")
        
        client_names = ['Urban Cardiac', 'Rural Primary', 'Heart Institute', 'General Hospital', 'Preventive Clinic']
        client_ids = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4']
        disease_ratios = [0.70, 0.20, 0.80, 0.50, 0.30]
        
        rounds = list(range(1, len(metrics['global']['f1']) + 1))
        
        fig4 = go.Figure()
        colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (cid, name, color) in enumerate(zip(client_ids, client_names, colors)):
            fig4.add_trace(go.Scatter(
                x=rounds,
                y=metrics['clients'][cid]['f1'],
                mode='lines+markers',
                name=f"{name} ({disease_ratios[i]*100:.0f}% disease)",
                line=dict(color=color, width=2)
            ))
        
        fig4.update_layout(
            title="F1-Score Evolution by Client",
            xaxis_title="Round",
            yaxis_title="F1-Score",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Data distribution
        st.markdown("---")
        st.markdown("### 📊 Non-IID Data Distribution")
        
        fig5 = go.Figure(data=[
            go.Bar(
                name='Disease Cases',
                x=client_names,
                y=disease_ratios,
                marker_color='#e74c3c'
            ),
            go.Bar(
                name='Healthy Cases',
                x=client_names,
                y=[1-r for r in disease_ratios],
                marker_color='#2ecc71'
            )
        ])
        
        fig5.update_layout(
            barmode='stack',
            title='Data Distribution Across Hospitals (Non-IID)',
            xaxis_title='Hospital',
            yaxis_title='Proportion',
            height=400
        )
        
        st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("""
        <div class="success-box">
        <b>✅ Privacy Preserved:</b> Each hospital's raw patient data never leaves their premises. 
        Only model parameters are shared during federated learning!
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>Federated Learning Demo</b> | Built with Flower + PyTorch + Streamlit</p>
    <p>🔒 Privacy-Preserving Machine Learning for Healthcare</p>
</div>
""", unsafe_allow_html=True)