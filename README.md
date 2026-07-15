# Achint-Shield
A.C.H.I.N.T. (Adaptive Cyber Heuristics for Intelligent Network Tracking) is an experimental research framework designed to detect, classify, and track anomalous and malicious activity across network telemetry in near real time. It blends adaptive heuristic rules with data-driven machine learning so that lightweight, interpretable heuristics can be combined with statistical and learned detectors to improve detection coverage and reduce false positives. The project targets operational network monitoring, threat hunting, and research into hybrid detection strategies that remain robust under concept drift and adversarial behavior.

Goals and Design Principles
Adaptive layering — combine fast, explainable heuristics with slower, higher‑fidelity ML models so alerts can be triaged by confidence and cost.

Modularity — separate ingestion, feature engineering, model training, and inference so components can be swapped or extended independently.

Reproducibility — provide deterministic preprocessing, versioned datasets, and training artifacts to reproduce experiments.

Operational readiness — produce compact alert artifacts (CSV/JSON) and lightweight inference code suitable for integration with SIEMs or stream processors.

Privacy aware — favor aggregated and metadata features over raw payload storage to reduce privacy and compliance risk.

Architecture and Core Components
High level flow

Ingest — raw PCAPs, NetFlow, or CSV telemetry are normalized into a canonical flow/record schema.

Preprocess — cleaning, sessionization, and enrichment (DNS, GeoIP, ASN) produce consistent inputs for feature extraction.

Feature extraction — time windowed statistics, protocol-specific counters, behavioral baselines, and engineered indicators (e.g., burstiness, entropy, uncommon ports).

Heuristic layer — rule engine with adaptive thresholds and scoring that encodes domain knowledge for immediate, explainable alerts.

Model layer — supervised and unsupervised detectors (classifiers, anomaly scorers) trained on engineered features; supports ensemble scoring.

Decision fusion — combines heuristic scores and model outputs into final alerts with confidence, provenance, and suggested triage actions.

Storage and export — alerts, traces, and metrics saved to runs/ and models/ for audit and retraining.

Key modules

src/ingest — parsers and normalizers for PCAP/CSV/flow formats.

src/preprocess — sessionization, enrichment, and dataset builders.

src/features — feature calculators and windowing utilities.

src/heuristics — rule definitions, adaptive threshold manager, and scoring API.

src/models — training, evaluation, and inference wrappers for ML models.

src/fusion — logic to combine heuristic and model outputs into actionable alerts.

Data Pipeline and Modeling Details
Feature engineering

Temporal windows — sliding and tumbling windows to capture short bursts and long-term trends.

Statistical summaries — mean, variance, percentiles, interarrival times, and entropy measures.

Categorical encodings — protocol, port clusters, ASN, and domain reputation buckets.

Behavioral baselines — per-host and per-subnet baselines to compute deviation scores.

Modeling approaches

Supervised classifiers — tree ensembles and lightweight neural nets for labeled attack classes.

Unsupervised detectors — isolation forests, one-class SVMs, and density estimators for novel anomalies.

Hybrid ensembles — weighted fusion of heuristic scores and model confidences; supports dynamic reweighting based on recent performance.

Evaluation — cross validation with time-aware splits, precision/recall curves, ROC/AUC, and operational metrics like alert volume and mean time to detect.

Usage Scenarios and Examples
Real time monitoring — run the inference pipeline on streaming telemetry to generate prioritized alerts for SOC analysts.

Threat hunting — use notebooks and saved features to explore suspicious hosts, pivot on indicators, and validate hypotheses.

Model research — benchmark new feature sets or detectors against archived datasets in data/ and track results in runs/.

Red team validation — replay PCAPs or synthetic attack traces through the pipeline to measure detection coverage and tuning needs.

Example commands

Preprocess raw captures: python src/preprocess.py --input data/raw --output data/processed

Train baseline model: python src/train.py --data data/processed --out models/

Run inference on a sample: python src/infer.py --model models/best.pt --input data/processed/sample.csv

Evaluation, Limitations, and Future Work
Evaluation focus

Operational metrics — alert precision at fixed recall, false positive rate per host, and analyst workload estimates.

Robustness tests — concept drift simulations, adversarial perturbations, and dataset imbalance handling.

Known limitations

Label scarcity — supervised components depend on labeled attacks which may be limited or biased.

Dataset bias — models trained on specific network environments may not generalize without domain adaptation.

Resource constraints — some detectors require tuning to run at scale in high-throughput environments.

Planned improvements

Online learning — incremental model updates to adapt to drift without full retraining.

Active learning — analyst-in-the-loop labeling to improve supervised detectors efficiently.

Stream-native deployment — connectors for Kafka/Fluentd and lightweight inference containers for edge deployment.

Explainability — richer provenance and per-feature contribution scores for each alert.

Contribution, Citation, and Contact
How to contribute

Fork the repo, create a feature branch, add tests and documentation, and open a PR. Follow PEP8 and include unit tests for new functionality. Use the existing notebooks as templates for experiments and include reproducible requirements.txt updates when adding dependencies.

How to cite

If you use A.C.H.I.N.T. in research, cite the repository and any accompanying paper or technical report included in the repo. Include version or commit hash for reproducibility.

Maintainer and contact

See the repository README header and LICENSE for maintainer contact details and licensing terms. For research collaboration or dataset questions, open an issue or pull request describing your use case.
