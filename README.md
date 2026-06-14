# 🛡️ 5G DDoS Intelligence Platform

An AI-powered DDoS detection system for 5G networks, combining classical ML models using Ensemble technique..

## 🚀 Features

| Feature | Description |
|---|---|
| **Multi-Model Ensemble** | RandomForest + XGBoost working in parallel |
| **Interactive Dashboard** | Real-time Streamlit dashboard with 4 analysis modes |
| **Feature Importance** | Explainable AI — see which metrics trigger detections |
| **Model Comparison** | Head-to-head accuracy, F1, ROC AUC benchmarks |
| **Batch Simulation** | Analyze 10–200 flows with timeline visualization |

## 🏗️ Architecture

```
Network Traffic (PCAP/CSV)
        │
        ▼
 Feature Extraction (78 features)
        │
   ┌────┴────┐
   │         │
RandomForest  XGBoost
   │         │
   └────┬────┘
        │  Ensemble Vote
        ▼
  Threat Decision
        │
        ▼
  Streamlit Dashboard (Real-time visualization)
```

## 📦 Setup

```bash
pip install streamlit plotly pandas scikit-learn xgboost joblib requests
```

Download the dataset: [CIC-IDS-2017 Friday DDoS](https://www.unb.ca/cic/datasets/ids-2017.html)
Place `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` in the project root.

## ▶️ Run

**Dashboard (recommended):**
```bash
streamlit run dashboard.py
```

**CLI Simulation:**
```bash
python detector.py
```

## 📊 Dashboard Modes

1. **Single Flow Analysis** — Analyze one random flow with dual-model verdict, and confidence gauges.
2. **Batch Simulation** — Run 10–200 flows, see timeline, distribution charts, and model agreement breakdown
3. **Feature Intelligence** — Top-15 feature importances, class distributions, dataset overview
4. **Model Comparison** — Head-to-head radar chart, metrics table, confidence distribution


## 📁 Project Structure

```
5G-DDOS-detector/
├── dashboard.py          # Streamlit web dashboard (main UI)
├── detector.py           # CLI simulation with LLM analysis
├── RandomForest.pkl       # Trained RF model
├── XGBoost.pkl            # Trained XGBoost model
├── README.md              # This file
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv  # Dataset
```

## 🔬 Dataset

CIC-IDS-2017 (Canadian Institute for Cybersecurity)
- 78 network flow features extracted from PCAP
- Labels: BENIGN / DDoS

## 🎓 Academic Context

This system demonstrates:
- **Network Security**: DDoS detection in 5G infrastructure
- **Ensemble ML**: Combining RandomForest + XGBoost for higher precision
- **Explainable AI (XAI)**: Feature importance for model transparency
- **LLM Integration**: Generative AI for automated threat narration
- **Real-time Systems**: Streaming detection simulation

## Future Scope 
Add LLM integration for report generation
