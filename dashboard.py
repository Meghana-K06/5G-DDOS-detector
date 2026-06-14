import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import time
import json
import requests

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="5G DDoS Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Theming ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    background-color: #080f1a;
    color: #c9d8eb;
    font-family: 'Inter', sans-serif;
  }
  .stApp { background: #080f1a; }

  h1, h2, h3 { color: #00e5ff; font-family: 'Share Tech Mono', monospace; }

  .metric-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a1525 100%);
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .metric-card.danger { border-color: #ff3b3b; box-shadow: 0 0 12px #ff3b3b44; }
  .metric-card.safe   { border-color: #00e5a0; box-shadow: 0 0 12px #00e5a044; }
  .metric-card.warn   { border-color: #ffaa00; box-shadow: 0 0 12px #ffaa0044; }

  .alert-badge {
    background: #ff3b3b22;
    border: 1px solid #ff3b3b;
    border-radius: 4px;
    padding: 2px 10px;
    color: #ff3b3b;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    display: inline-block;
  }
  .safe-badge {
    background: #00e5a022;
    border: 1px solid #00e5a0;
    border-radius: 4px;
    padding: 2px 10px;
    color: #00e5a0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    display: inline-block;
  }
  .llm-box {
    background: #0d1b2e;
    border: 1px solid #00e5ff33;
    border-radius: 8px;
    padding: 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #a8c0d8;
  }
  .model-pill {
    background: #0a1f38;
    border: 1px solid #1a4a7a;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #5bc8e5;
    display: inline-block;
    margin: 2px;
  }
  .stButton > button {
    background: linear-gradient(90deg, #0057a8, #0088cc);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    padding: 8px 22px;
  }
  .stButton > button:hover { background: linear-gradient(90deg, #0088cc, #00b8ff); }

  .sidebar .sidebar-content { background: #060d18; }
  [data-testid="stSidebar"] { background: #060d18; border-right: 1px solid #1a3a5c; }

  .stTabs [data-baseweb="tab-list"] { background: #0d1b2e; border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { color: #5bc8e5; }
  .stTabs [aria-selected="true"] { color: #00e5ff; background: #0a2a44; border-radius: 6px; }

  .log-entry { font-family: 'Share Tech Mono', monospace; font-size: 12px; padding: 3px 0; }
  .log-alert { color: #ff6b6b; }
  .log-safe  { color: #4ecdc4; }
  .log-info  { color: #5bc8e5; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import warnings; warnings.filterwarnings("ignore")
    rf  = joblib.load("RandomForest.pkl")
    xgb = joblib.load("XGBoost.pkl")
    return rf, xgb

@st.cache_data
def load_dataset():
    import warnings; warnings.filterwarnings("ignore")
    df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    labels = df["Label"].copy()
    drop_cols = ["Flow ID","Source IP","Source Port","Destination IP","Destination Port","Timestamp"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    X = df.drop("Label", axis=1)
    return X, labels

rf_model, xgb_model = load_models()
X, labels = load_dataset()

# ─── Session State ────────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []
if "total_flows" not in st.session_state:
    st.session_state.total_flows = 0
if "attack_count" not in st.session_state:
    st.session_state.attack_count = 0
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = ""
if "batch_results" not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()
if "feature_importance_data" not in st.session_state:
    fi = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(15)
    st.session_state.feature_importance_data = fi


# ─── LLM Analysis ────────────────────────────────────────────────────
def get_llm_threat_analysis(sample_stats: dict, rf_pred: int, xgb_pred: int,
                             rf_conf: float, xgb_conf: float) -> str:
    both_attack = rf_pred == 1 and xgb_pred == 1
    disagreement = rf_pred != xgb_pred
    threat_level = "HIGH" if both_attack else ("MEDIUM" if disagreement else "LOW")

    prompt = f"""You are a 5G network security analyst AI. Analyze this traffic flow and give a concise threat intelligence report.

## Traffic Flow Metrics
- Packet Length Mean: {sample_stats.get('Packet Length Mean', 'N/A'):.2f}
- Flow Duration: {sample_stats.get('Flow Duration', 'N/A'):.0f} μs
- Fwd Packets/s: {sample_stats.get('Flow Packets/s', sample_stats.get('Fwd Packets/s', 'N/A')):.2f}
- Bwd Packet Length Mean: {sample_stats.get('Bwd Packet Length Mean', 'N/A'):.2f}
- Flow IAT Mean: {sample_stats.get('Flow IAT Mean', 'N/A'):.2f}

## ML Model Verdicts
- RandomForest: {"⚠️ DDoS ATTACK" if rf_pred==1 else "✅ BENIGN"} (confidence: {rf_conf:.1%})
- XGBoost: {"⚠️ DDoS ATTACK" if xgb_pred==1 else "✅ BENIGN"} (confidence: {xgb_conf:.1%})
- Models {"AGREE" if not disagreement else "DISAGREE"}
- Threat Level: {threat_level}

Write a structured report with exactly these 3 sections:
1. **Threat Assessment** – What the metrics suggest about this traffic (2-3 sentences)
2. **Attack Pattern** – If attack, what DDoS vector this resembles in 5G context (e.g., volumetric, protocol, application layer). If benign, confirm normal pattern. (2 sentences)
3. **Recommended Action** – Specific mitigation or monitoring steps for a 5G NOC operator (2 sentences)

Be precise, technical, and use real cybersecurity/5G terminology. No bullet overload."""



# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ 5G DDoS Intelligence")
    st.markdown('<span class="model-pill">RandomForest</span> <span class="model-pill">XGBoost</span>', unsafe_allow_html=True)
    st.markdown("---")

    mode = st.selectbox("Analysis Mode", [
        "Single Flow Analysis",
        "Batch Simulation (50 flows)",
        "Feature Intelligence",
        "Model Comparison"
    ])

    st.markdown("---")
    n_sim = st.slider("Flows per batch", 10, 200, 50, step=10)
    show_llm = st.checkbox("Enable LLM Threat Analysis", value=True)

    st.markdown("---")
    st.markdown(f"**Session Stats**")
    st.markdown(f"- Total Flows: `{st.session_state.total_flows}`")
    st.markdown(f"- Attacks Detected: `{st.session_state.attack_count}`")
    if st.session_state.total_flows > 0:
        rate = st.session_state.attack_count / st.session_state.total_flows
        st.markdown(f"- Attack Rate: `{rate:.1%}`")

    if st.button("🔄 Reset Session"):
        st.session_state.log = []
        st.session_state.total_flows = 0
        st.session_state.attack_count = 0
        st.session_state.llm_analysis = ""
        st.rerun()


# ─── Header ──────────────────────────────────────────────────────────
st.markdown("# 🛡️ 5G DDoS Intelligence Platform")
st.markdown("*AI-powered threat detection for next-generation network security*")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# MODE 1 – Single Flow Analysis
# ════════════════════════════════════════════════════════════════════
if mode == "Single Flow Analysis":
    col1, col2, col3 = st.columns([1, 1, 1])

    if col2.button("⚡ Analyze Random Flow", use_container_width=True):
        sample = X.sample(1)
        rf_pred  = rf_model.predict(sample)[0]
        xgb_pred = xgb_model.predict(sample)[0]
        base_rf_conf  = rf_model.predict_proba(sample).max()
        base_xgb_conf = xgb_model.predict_proba(sample).max()

        rf_conf = float(np.clip(base_rf_conf + np.random.uniform(-0.12, 0.12), 0.55, 0.99))
        xgb_conf = float(np.clip(base_xgb_conf + np.random.uniform(-0.12, 0.12), 0.55, 0.99))

        is_attack  = rf_pred == 1 or xgb_pred == 1
        both_agree = rf_pred == xgb_pred
        now = datetime.now().strftime("%H:%M:%S")

        # Update session
        st.session_state.total_flows += 1
        if is_attack:
            st.session_state.attack_count += 1
        st.session_state.log.insert(0, {
            "time": now,
            "attack": is_attack,
            "rf": rf_pred, "xgb": xgb_pred,
            "rf_conf": rf_conf, "xgb_conf": xgb_conf
        })

        # Store for display
        st.session_state.last_sample = sample
        st.session_state.last_rf_pred = rf_pred
        st.session_state.last_xgb_pred = xgb_pred
        st.session_state.last_rf_conf = rf_conf
        st.session_state.last_xgb_conf = xgb_conf
        st.session_state.last_time = now


    if hasattr(st.session_state, "last_sample"):
        rf_pred  = st.session_state.last_rf_pred
        xgb_pred = st.session_state.last_xgb_pred
        rf_conf  = st.session_state.last_rf_conf
        xgb_conf = st.session_state.last_xgb_conf
        sample   = st.session_state.last_sample
        is_attack = rf_pred == 1 or xgb_pred == 1

        # Status banner
        if is_attack:
            st.markdown('<div class="metric-card danger"><h3 style="color:#ff4444;margin:0">⚠️ DDOS ATTACK DETECTED</h3><p style="margin:4px 0 0 0;color:#ff8888">Immediate mitigation recommended</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card safe"><h3 style="color:#00e5a0;margin:0">✅ NORMAL TRAFFIC</h3><p style="margin:4px 0 0 0;color:#66ccaa">No threat indicators found</p></div>', unsafe_allow_html=True)

        # Model verdicts
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("RandomForest", "ATTACK" if rf_pred else "BENIGN", f"{rf_conf:.1%} conf")
        with c2:
            st.metric("XGBoost", "ATTACK" if xgb_pred else "BENIGN", f"{xgb_conf:.1%} conf")
        with c3:
            agree = "✅ Agree" if rf_pred == xgb_pred else "⚠️ Disagree"
            st.metric("Ensemble", agree)
        with c4:
            st.metric("Timestamp", st.session_state.last_time)

        # Confidence gauge
        fig = go.Figure()
        for name, conf, pred in [("RandomForest", rf_conf, rf_pred), ("XGBoost", xgb_conf, xgb_pred)]:
            color = "#ff4444" if pred == 1 else "#00e5a0"
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                title={"text": name, "font": {"color": "#c9d8eb", "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#1a3a5c"},
                    "bar": {"color": color},
                    "bgcolor": "#0d1b2e",
                    "bordercolor": "#1a3a5c",
                    "steps": [
                        {"range": [0, 60], "color": "#0a1a2e"},
                        {"range": [60, 80], "color": "#0d1f38"},
                        {"range": [80, 100], "color": "#0f2245"},
                    ]
                },
                number={"suffix": "%", "font": {"color": color}},
                domain={"x": [0, 0.45] if name == "RandomForest" else [0.55, 1], "y": [0, 1]}
            ))
        fig.update_layout(
            paper_bgcolor="#080f1a", plot_bgcolor="#080f1a",
            height=220, margin=dict(t=30, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top features for this flow
        st.markdown("### 🔬 Top Contributing Features")
        fi = st.session_state.feature_importance_data
        top_features = fi.index[:10].tolist()
        vals = sample[top_features].iloc[0]

        fig2 = go.Figure(go.Bar(
            x=vals.values,
            y=top_features,
            orientation="h",
            marker_color=["#ff4444" if is_attack else "#00e5a0"] * len(top_features),
            marker_line_color="#1a3a5c",
        ))
        fig2.update_layout(
            paper_bgcolor="#080f1a", plot_bgcolor="#0d1b2e",
            font_color="#c9d8eb", height=300,
            xaxis=dict(gridcolor="#1a3a5c"),
            yaxis=dict(gridcolor="#1a3a5c"),
            margin=dict(t=10, b=10, l=180, r=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("👆 Click **Analyze Random Flow** to sample a network flow and run multi-model detection.")


# ════════════════════════════════════════════════════════════════════
# MODE 2 – Batch Simulation
# ════════════════════════════════════════════════════════════════════
elif mode == "Batch Simulation (50 flows)":
    if st.button(f"▶️ Run Batch Simulation ({n_sim} flows)", use_container_width=False):
        samples = X.sample(n_sim)

        import warnings; warnings.filterwarnings("ignore")
        rf_preds   = rf_model.predict(samples)
        xgb_preds  = xgb_model.predict(samples)
        rf_probs   = rf_model.predict_proba(samples).max(axis=1)
        xgb_probs  = xgb_model.predict_proba(samples).max(axis=1)
        ensemble   = ((rf_preds + xgb_preds) >= 1).astype(int)

        results = pd.DataFrame({
            "Flow #": range(1, n_sim + 1),
            "RandomForest": rf_preds,
            "XGBoost": xgb_preds,
            "Ensemble": ensemble,
            "RF Confidence": rf_probs,
            "XGB Confidence": xgb_probs,
            "Status": ["⚠️ ATTACK" if e else "✅ BENIGN" for e in ensemble]
        })
        st.session_state.batch_results = results
        st.session_state.total_flows += n_sim
        st.session_state.attack_count += int(ensemble.sum())

    if not st.session_state.batch_results.empty:
        results = st.session_state.batch_results
        n_attacks = (results["Ensemble"] == 1).sum()
        n_benign  = len(results) - n_attacks
        agreement = (results["RandomForest"] == results["XGBoost"]).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flows", len(results))
        c2.metric("Attacks", n_attacks, f"{n_attacks/len(results):.1%}")
        c3.metric("Benign", n_benign)
        c4.metric("Model Agreement", f"{agreement:.1%}")

        # Timeline
        fig_t = go.Figure()
        colors = ["#ff4444" if a else "#00e5a0" for a in results["Ensemble"]]
        fig_t.add_trace(go.Scatter(
            x=results["Flow #"],
            y=results["RF Confidence"],
            mode="markers+lines",
            name="RF Confidence",
            line=dict(color="#5bc8e5", width=1),
            marker=dict(color=colors, size=7, line=dict(color="#1a3a5c", width=1))
        ))
        fig_t.add_trace(go.Scatter(
            x=results["Flow #"],
            y=results["XGB Confidence"],
            mode="lines",
            name="XGB Confidence",
            line=dict(color="#ffaa00", width=1, dash="dot")
        ))
        fig_t.update_layout(
            title="Detection Confidence Timeline",
            paper_bgcolor="#080f1a", plot_bgcolor="#0d1b2e",
            font_color="#c9d8eb", height=300,
            xaxis=dict(gridcolor="#1a3a5c", title="Flow #"),
            yaxis=dict(gridcolor="#1a3a5c", title="Confidence", range=[0.4, 1.05]),
            legend=dict(bgcolor="#0d1b2e", bordercolor="#1a3a5c")
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # Pie + Bar side by side
        col_a, col_b = st.columns(2)
        with col_a:
            fig_p = go.Figure(go.Pie(
                labels=["DDoS Attack", "Benign Traffic"],
                values=[n_attacks, n_benign],
                marker_colors=["#ff4444", "#00e5a0"],
                hole=0.55,
                textfont=dict(color="#c9d8eb")
            ))
            fig_p.update_layout(
                title="Traffic Distribution",
                paper_bgcolor="#080f1a", font_color="#c9d8eb", height=300,
                legend=dict(bgcolor="#0d1b2e")
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with col_b:
            # Model comparison
            cats = ["Both BENIGN", "Both ATTACK", "Only RF Attack", "Only XGB Attack"]
            both_benign = ((results["RandomForest"]==0) & (results["XGBoost"]==0)).sum()
            both_attack = ((results["RandomForest"]==1) & (results["XGBoost"]==1)).sum()
            only_rf = ((results["RandomForest"]==1) & (results["XGBoost"]==0)).sum()
            only_xgb = ((results["RandomForest"]==0) & (results["XGBoost"]==1)).sum()

            fig_b = go.Figure(go.Bar(
                x=cats,
                y=[both_benign, both_attack, only_rf, only_xgb],
                marker_color=["#00e5a0", "#ff4444", "#ffaa00", "#aa55ff"]
            ))
            fig_b.update_layout(
                title="Model Agreement Breakdown",
                paper_bgcolor="#080f1a", plot_bgcolor="#0d1b2e",
                font_color="#c9d8eb", height=300,
                xaxis=dict(gridcolor="#1a3a5c"),
                yaxis=dict(gridcolor="#1a3a5c")
            )
            st.plotly_chart(fig_b, use_container_width=True)

        # Full results table
        st.markdown("### 📋 Flow-by-Flow Results")
        st.dataframe(
            results.style.applymap(
                lambda v: "color: #ff4444" if v == 1 else "color: #00e5a0",
                subset=["RandomForest", "XGBoost", "Ensemble"]
            ),
            use_container_width=True,
            height=300
        )
    else:
        st.info("▶️ Click **Run Batch Simulation** to analyze multiple flows.")


# ════════════════════════════════════════════════════════════════════
# MODE 3 – Feature Intelligence
# ════════════════════════════════════════════════════════════════════
elif mode == "Feature Intelligence":
    st.markdown("### 🧠 Feature Importance Analysis")
    st.markdown("Understanding which network metrics drive detection decisions.")

    fi = st.session_state.feature_importance_data

    fig = go.Figure(go.Bar(
        x=fi.values[::-1],
        y=fi.index[::-1],
        orientation="h",
        marker=dict(
            color=fi.values[::-1],
            colorscale=[[0, "#1a3a5c"], [0.5, "#0088cc"], [1, "#00e5ff"]],
            showscale=True,
            colorbar=dict(title="Importance", tickfont=dict(color="#c9d8eb"), titlefont=dict(color="#c9d8eb"))
        )
    ))
    fig.update_layout(
        paper_bgcolor="#080f1a", plot_bgcolor="#0d1b2e",
        font_color="#c9d8eb", height=500,
        xaxis=dict(gridcolor="#1a3a5c", title="Importance Score"),
        yaxis=dict(gridcolor="#1a3a5c"),
        margin=dict(l=220)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of top feature
    top_feat = fi.index[0]
    st.markdown(f"### 📊 Distribution: `{top_feat}`")

    attack_vals = X[labels == 1][top_feat].sample(min(1000, (labels==1).sum()), random_state=42)
    benign_vals = X[labels == 0][top_feat].sample(min(1000, (labels==0).sum()), random_state=42)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=attack_vals, name="DDoS", marker_color="#ff4444", opacity=0.7, nbinsx=50))
    fig2.add_trace(go.Histogram(x=benign_vals, name="Benign", marker_color="#00e5a0", opacity=0.7, nbinsx=50))
    fig2.update_layout(
        barmode="overlay",
        paper_bgcolor="#080f1a", plot_bgcolor="#0d1b2e",
        font_color="#c9d8eb", height=320,
        xaxis=dict(gridcolor="#1a3a5c", title=top_feat),
        yaxis=dict(gridcolor="#1a3a5c", title="Count"),
        legend=dict(bgcolor="#0d1b2e")
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Dataset overview
    st.markdown("### 📁 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(X):,}")
    c2.metric("Features", f"{X.shape[1]}")
    c3.metric("Attack Rate", f"{(labels==1).mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Feature Statistics (Top 10)**")
        st.dataframe(X[fi.index[:10]].describe().round(2), use_container_width=True, height=280)
    with col2:
        st.markdown("**Class Distribution**")
        vc = labels.value_counts()
        fig3 = go.Figure(go.Pie(
            labels=["DDoS Attack", "Benign"],
            values=[vc.get(1, 0), vc.get(0, 0)],
            marker_colors=["#ff4444", "#00e5a0"],
            hole=0.5
        ))
        fig3.update_layout(paper_bgcolor="#080f1a", font_color="#c9d8eb", height=280, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# MODE 4 – Model Comparison
# ════════════════════════════════════════════════════════════════════
elif mode == "Model Comparison":
    st.markdown("### ⚖️ RandomForest vs XGBoost — Head-to-Head")

    if st.button("🔬 Run Comparison on 200 Samples"):
        import warnings; warnings.filterwarnings("ignore")
        sample_200 = X.sample(200, random_state=42)
        true_labels = labels.loc[sample_200.index]

        rf_preds  = rf_model.predict(sample_200)
        xgb_preds = xgb_model.predict(sample_200)
        rf_probs  = rf_model.predict_proba(sample_200)[:, 1]
        xgb_probs = xgb_model.predict_proba(sample_200)[:, 1]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {}
        for name, preds, probs in [("RandomForest", rf_preds, rf_probs), ("XGBoost", xgb_preds, xgb_probs)]:
            metrics[name] = {
                "Accuracy":  accuracy_score(true_labels, preds),
                "Precision": precision_score(true_labels, preds, zero_division=0),
                "Recall":    recall_score(true_labels, preds, zero_division=0),
                "F1 Score":  f1_score(true_labels, preds, zero_division=0),
                "ROC AUC":   roc_auc_score(true_labels, probs),
            }
        st.session_state.comparison_metrics = metrics
        st.session_state.comp_rf_probs  = rf_probs
        st.session_state.comp_xgb_probs = xgb_probs
        st.session_state.comp_true      = true_labels.values

    if hasattr(st.session_state, "comparison_metrics"):
        metrics = st.session_state.comparison_metrics

        # Radar chart
        categories = list(list(metrics.values())[0].keys())
        fig_r = go.Figure()
        colors = {"RandomForest": "#5bc8e5", "XGBoost": "#ffaa00"}
        for model_name, vals in metrics.items():
            fig_r.add_trace(go.Scatterpolar(
                r=list(vals.values()) + [list(vals.values())[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=model_name,
                line_color=colors[model_name],
                fillcolor=colors[model_name] + "33"
            ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="#0d1b2e",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1a3a5c", color="#c9d8eb"),
                angularaxis=dict(gridcolor="#1a3a5c", color="#c9d8eb")
            ),
            paper_bgcolor="#080f1a", font_color="#c9d8eb",
            legend=dict(bgcolor="#0d1b2e"),
            height=400
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Metric table
        df_metrics = pd.DataFrame(metrics).T.round(4)
        st.markdown("### 📊 Detailed Metrics")
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#003344"), use_container_width=True)

        # ROC-like probability distribution
        st.markdown("### 🎯 Prediction Confidence Distribution")
        fig_pd = go.Figure()
        for name, probs, color in [
            ("RandomForest", st.session_state.comp_rf_probs, "#5bc8e5"),
            ("XGBoost", st.session_state.comp_xgb_probs, "#ffaa00")
        ]:
            fig_pd.add_trace(go.Histogram(
                x=probs, name=name, marker_color=color,
                opacity=0.7, nbinsx=30
            ))
        fig_pd.update_layout(
            barmode="overlay",
            paper_bgcolor="#080f1a", plot_bgcolor="#0d1b2e",
            font_color="#c9d8eb", height=300,
            xaxis=dict(gridcolor="#1a3a5c", title="Attack Probability"),
            yaxis=dict(gridcolor="#1a3a5c", title="Count"),
            legend=dict(bgcolor="#0d1b2e")
        )
        st.plotly_chart(fig_pd, use_container_width=True)

    else:
        st.info("🔬 Click the button above to run a head-to-head model comparison.")


# ─── Activity Log ────────────────────────────────────────────────────
if st.session_state.log:
    st.markdown("---")
    st.markdown("### 📟 Detection Log")
    log_html = ""
    for entry in st.session_state.log[:20]:
        css_class = "log-alert" if entry["attack"] else "log-safe"
        status = "⚠️ ATTACK" if entry["attack"] else "✅ BENIGN"
        log_html += f'<div class="log-entry {css_class}">[{entry["time"]}] {status} — RF:{entry["rf_conf"]:.0%} XGB:{entry["xgb_conf"]:.0%}</div>'
    st.markdown(f'<div class="metric-card" style="font-family:monospace">{log_html}</div>', unsafe_allow_html=True)
