"""
pipeline/ — AI-Driven Network Traffic Anomaly Detection Pipeline

Modules:
    config        — Global seeds, hyperparameters, and paths
    data_loader   — Dataset ingestion and leak-proof splitting
    preprocessor  — Imputation, scaling, SMOTE (train-only)
    trainer       — Model training orchestration
    evaluator     — Comprehensive test-set evaluation and reporting
    models/       — Model implementations (RF, XGBoost, DL)
"""
