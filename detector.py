"""
5G DDoS Intelligence Platform - Core Detector
Supports: real-time simulation, multi-model ensemble, LLM analysis
"""

import pandas as pd
import joblib
import time
import numpy as np
import warnings
import requests
warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────
N_FLOWS      = 20        # flows to simulate
DELAY_SEC    = 1.5       # delay between flows
ENABLE_LLM   = True      # enable Claude threat intelligence
DROP_COLS    = ["Flow ID", "Source IP", "Source Port",
                "Destination IP", "Destination Port", "Timestamp"]

# ─── Load Models ─────────────────────────────────────────────────────
print("Loading models...")
rf_model  = joblib.load("RandomForest.pkl")
xgb_model = joblib.load("XGBoost.pkl")
print("  ✓ RandomForest loaded")
print("  ✓ XGBoost loaded")

# ─── Load Dataset ────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
X = df.drop("Label", axis=1)
print(f"  ✓ Dataset: {len(X):,} flows, {X.shape[1]} features\n")

# ─── LLM Threat Analysis ─────────────────────────────────────────────
def llm_analyze(sample_dict: dict, rf_pred: int, xgb_pred: int,
                rf_conf: float, xgb_conf: float) -> str:
    threat = "HIGH" if (rf_pred==1 and xgb_pred==1) else (
             "MEDIUM" if rf_pred != xgb_pred else "LOW")
    prompt = f"""Analyze this 5G traffic flow in one short paragraph (3-4 sentences).
Models: RF={'ATTACK' if rf_pred else 'BENIGN'}({rf_conf:.0%}), XGB={'ATTACK' if xgb_pred else 'BENIGN'}({xgb_conf:.0%}). Threat: {threat}.
Top metrics - Packet Length Mean: {sample_dict.get('Packet Length Mean', 0):.1f}, Flow Duration: {sample_dict.get('Flow Duration', 0):.0f}μs, Flow Packets/s: {sample_dict.get('Flow Packets/s', sample_dict.get('Fwd Packets/s', 0)):.1f}.
Give a concise technical assessment: what type of traffic, if attack what vector (volumetric/protocol/app-layer), recommended action."""
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=15
        )
        return r.json()["content"][0]["text"]
    except:
        return "LLM analysis unavailable."

# ─── Simulation ───────────────────────────────────────────────────────
print("=" * 65)
print("  5G LIVE THREAT DETECTION SIMULATION")
print("  Models: RandomForest + XGBoost Ensemble + Claude AI")
print("=" * 65)

attack_count = 0
disagreements = 0

for i in range(N_FLOWS):
    sample = X.sample(1)

    rf_pred   = rf_model.predict(sample)[0]
    xgb_pred  = xgb_model.predict(sample)[0]
    rf_conf   = rf_model.predict_proba(sample).max()
    xgb_conf  = xgb_model.predict_proba(sample).max()

    # Ensemble: flag as attack if either model says attack
    ensemble = int(rf_pred == 1 or xgb_pred == 1)
    agreed   = rf_pred == xgb_pred

    if ensemble:
        attack_count += 1
    if not agreed:
        disagreements += 1

    ts = time.strftime("%H:%M:%S")
    status = "⚠️  ATTACK" if ensemble else "✅ BENIGN"
    agree_tag = "AGREE" if agreed else "⚡DISAGREE"

    print(f"\n[{ts}] Flow {i+1:02d}/{N_FLOWS}")
    print(f"  Status   : {status}")
    print(f"  RF       : {'ATTACK' if rf_pred else 'BENIGN'} ({rf_conf:.1%})")
    print(f"  XGBoost  : {'ATTACK' if xgb_pred else 'BENIGN'} ({xgb_conf:.1%})")
    print(f"  Models   : {agree_tag}")

    if ENABLE_LLM and ensemble:
        print("  AI Intel : ", end="", flush=True)
        analysis = llm_analyze(sample.iloc[0].to_dict(), rf_pred, xgb_pred, rf_conf, xgb_conf)
        # Print first 120 chars only
        preview = analysis[:200].replace("\n", " ")
        print(f"{preview}...")

    time.sleep(DELAY_SEC)

# ─── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SIMULATION COMPLETE")
print(f"  Flows Analyzed : {N_FLOWS}")
print(f"  Attacks Found  : {attack_count} ({attack_count/N_FLOWS:.0%})")
print(f"  Safe Flows     : {N_FLOWS - attack_count}")
print(f"  Disagreements  : {disagreements}")
print("=" * 65)
