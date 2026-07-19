![A.C.H.I.N.T. Cybersecurity](https://img.magnific.com/premium-photo/mysterious-figure-digital-world-representing-themes-cybersecurity-hacking-data-protection-with-vibrant-visuals_1301918-6116.jpg?semt=ais_hybrid&w=740&q=80)

# 🛡️ A.C.H.I.N.T.
> **Adaptive Cyber Heuristics for Intelligent Network Tracking**

Welcome to the frontier of operational network defense. **A.C.H.I.N.T.** is an experimental, near real-time research framework engineered to detect, classify, and track anomalous and malicious network activity. 

By brilliantly fusing **lightning-fast, interpretable heuristics** with **data-driven machine learning**, A.C.H.I.N.T. delivers unparalleled detection coverage while crushing false positives. Designed to thrive under concept drift and adversarial attacks, this is the ultimate toolkit for threat hunting, SOC monitoring, and cutting-edge hybrid detection research.

---

## 🎯 Goals & Design Principles

We built A.C.H.I.N.T. on five uncompromising pillars:

*   🧠 **Adaptive Layering:** Combine the blazing speed of explainable heuristics with the high-fidelity deep reasoning of ML models. Triage alerts by true confidence and operational cost.
*   🧩 **Fierce Modularity:** Ingestion, feature engineering, training, and inference are totally decoupled. Swap, scale, or extend components without breaking a sweat.
*   ♻️ **Absolute Reproducibility:** Deterministic preprocessing, versioned datasets, and clean training artifacts mean your experiments are verifiable every single time.
*   🚀 **Operational Readiness:** Built for the real world. Outputs compact alert artifacts (CSV/JSON) and lightweight inference code ready to plug straight into your SIEM or stream processors.
*   🔒 **Privacy by Design:** We prioritize aggregated metrics and metadata over raw payload hoarding, drastically slashing privacy and compliance risks.

---

## 🏗️ Architecture & Core Components

### 🌊 The High-Level Flow
1.  **📥 Ingest:** Raw PCAPs, NetFlow, or CSV telemetry are ingested and normalized into a pristine, canonical record schema.
2.  **✨ Preprocess:** Cleaning, sessionization, and powerful enrichments (DNS, GeoIP, ASN) forge the foundation.
3.  **📊 Extract:** Time-windowed stats, protocol counters, and engineered indicators (entropy, burstiness, odd ports) map out behavioral baselines.
4.  **⚡ Heuristic Layer:** The domain-knowledge engine. Adaptive thresholds and scoring provide immediate, explainable, zero-day alerts.
5.  **🤖 Model Layer:** Deep detectors (supervised classifiers and unsupervised anomaly scorers) hunt for complex, engineered attack patterns.
6.  **⚖️ Decision Fusion:** The brain of the operation. Heuristic scores and ML outputs fuse into high-confidence alerts loaded with provenance and triage context.
7.  **💾 Storage & Export:** Alerts, traces, and metrics are securely archived to `runs/` and `models/` for seamless audit and retraining.

### 📂 Key Modules
*   `src/ingest` — Battle-tested parsers and normalizers.
*   `src/preprocess` — Sessionization, enrichment, and dataset builders.
*   `src/features` — Advanced feature calculators and temporal windowing utilities.
*   `src/heuristics` — Rule definitions, adaptive thresholding, and the scoring API.
*   `src/models` — ML training, evaluation, and blazing-fast inference wrappers.
*   `src/fusion` — The intelligent logic combining heuristics and ML.

---

## ⚙️ Data Pipeline & ML Engine

### 🔬 Feature Engineering
*   **Temporal Windows:** Sliding and tumbling windows designed to catch everything from micro-burst attacks to low-and-slow campaigns.
*   **Statistical Summaries:** Mean, variance, percentiles, interarrival times, and Shannon entropy.
*   **Categorical Encodings:** Protocol mapping, port clustering, ASN tracking, and domain reputation.
*   **Behavioral Baselines:** Dynamic per-host and per-subnet profiling to ruthlessly expose deviations.

### 🧠 Modeling Approaches
*   **Supervised Classifiers:** Tree ensembles and lightweight neural networks targeting known attack vectors.
*   **Unsupervised Detectors:** Isolation forests, One-Class SVMs, and density estimators seeking out the unknown.
*   **Hybrid Ensembles:** Dynamically reweighted fusion of heuristics and models, adapting to performance in real-time.
*   **Rigorous Evaluation:** Time-aware cross-validation splits, PR curves, ROC/AUC, and strict operational metrics (MTTD, alert volume).

---

## 🎮 Usage Scenarios & Commands

Whether you are defending the perimeter or breaking it, A.C.H.I.N.T. is your weapon of choice.

*   **🚨 Real-Time Monitoring:** Stream telemetry through the pipeline for prioritized, actionable SOC alerts.
*   **🕵️ Threat Hunting:** Leverage Jupyter notebooks and feature sets to pivot on indicators and hunt down suspicious hosts.
*   **🧪 Model Research:** Benchmark experimental feature sets against archived datasets and track victories in your runs directory.
*   **💥 Red Team Validation:** Replay synthetic attack traces or PCAPs to stress-test your detection coverage and tune thresholds.

### 🔥 Quickstart Commands

**Preprocess raw captures:**
```bash
python src/preprocess.py --input data/raw --output data/processed
