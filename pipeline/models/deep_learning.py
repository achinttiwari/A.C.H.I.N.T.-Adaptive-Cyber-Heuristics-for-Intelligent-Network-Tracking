"""
=============================================================================
deep_learning.py — Feed-Forward Neural Network (TensorFlow / Keras)
=============================================================================
Purpose:
    Implements a multi-layer perceptron (MLP) as the deep learning model
    in the ensemble.  MLPs have shown strong performance on CICIDS2017
    when supplied with well-scaled features, outperforming shallow models
    on subtle, low-volume attack categories (e.g. Infiltration, Heartbleed).

Architecture (configured via DL_PARAMS in config.py):
    Input → [Dense → BatchNorm → ReLU → Dropout] × N hidden layers
          → Dense(1) → Sigmoid
    Output: P(ATTACK | features)

Design choices:
    - BatchNormalization after each dense layer stabilises gradient flow,
      allowing deeper networks without vanishing gradients.
    - Dropout is applied during training only (Keras handles this).
    - Binary cross-entropy loss with class_weight balancing is used instead
      of SMOTE here, providing an orthogonal view of class imbalance.
    - Early stopping monitors validation AUC (not loss) because AUC is
      the primary ranking metric in IDS evaluation.

References:
    Goodfellow, I., Bengio, Y., & Courville, A. (2016).
    Deep Learning. MIT Press.  Chapter 6 (MLP), Chapter 7 (Regularisation).

    Ring, M. et al. (2019). "Flow-based Network Traffic Generation using
    Deep Learning." Computers & Security, 82, 304–316.
=============================================================================
"""

import logging
import os

import numpy as np

from pipeline.config import ARTIFACT_DIR, DL_PARAMS, GLOBAL_SEED

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "models", "deep_learning.keras")


class DeepLearningModel:
    """
    Feed-forward neural network binary classifier.

    TensorFlow is imported inside methods so that:
      1. GPU initialisation is deferred until actually needed.
      2. The rest of the pipeline works without a GPU / TensorFlow install.
    """

    def __init__(self) -> None:
        self.model = None
        self.history = None
        self.name = "DeepLearning"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "DeepLearningModel":
        """
        Build and train the MLP.

        Parameters
        ----------
        X_train, y_train : np.ndarray  — Pre-scaled training data.
        X_val,   y_val   : np.ndarray  — Validation data for early stopping
                                          and metric logging.
        """
        # Lazy TF import
        import tensorflow as tf
        from tensorflow.keras import layers, callbacks, optimizers

        # -----------------------------------------------------------------
        # Set TensorFlow random seed for reproducibility.
        # Must be set before any TF operations for byte-identical results.
        # -----------------------------------------------------------------
        tf.random.set_seed(GLOBAL_SEED)

        n_features = X_train.shape[1]
        logger.info(
            "[%s] Building model | input_dim=%d | hidden_layers=%s | "
            "dropout=%.2f",
            self.name,
            n_features,
            DL_PARAMS["hidden_units"],
            DL_PARAMS["dropout_rate"],
        )

        # -----------------------------------------------------------------
        # Model construction (Keras functional / sequential API)
        # -----------------------------------------------------------------
        model = tf.keras.Sequential(name="IDS_MLP")

        # First hidden layer (input shape defined here)
        model.add(layers.Input(shape=(n_features,), name="input"))

        for i, units in enumerate(DL_PARAMS["hidden_units"]):
            model.add(
                layers.Dense(
                    units,
                    name=f"dense_{i}",
                    # He uniform initialisation is standard for ReLU networks;
                    # it keeps variance approximately constant across layers.
                    kernel_initializer=tf.keras.initializers.HeUniform(seed=GLOBAL_SEED),
                )
            )
            # Batch Normalisation: normalise layer activations to stabilise
            # training, especially important with large feature ranges.
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
            model.add(layers.Activation("relu", name=f"relu_{i}"))
            # Dropout: randomly zero-out neurons during training to prevent
            # co-adaptation and reduce overfitting.
            model.add(layers.Dropout(DL_PARAMS["dropout_rate"],
                                     seed=GLOBAL_SEED,
                                     name=f"dropout_{i}"))

        # Output layer: single sigmoid neuron → P(ATTACK)
        model.add(layers.Dense(1, activation="sigmoid", name="output"))

        # -----------------------------------------------------------------
        # Compilation
        # -----------------------------------------------------------------
        model.compile(
            # Adam with tuned learning rate from config
            optimizer=optimizers.Adam(learning_rate=DL_PARAMS["learning_rate"]),
            # Binary cross-entropy is the standard loss for binary
            # classification.  It encourages well-calibrated probabilities.
            loss="binary_crossentropy",
            # Track AUC during training — directly interpretable for IDS.
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        model.summary(print_fn=lambda x: logger.debug(x))

        # -----------------------------------------------------------------
        # Class weight computation
        #   Even after SMOTE on the training set we may still want class
        #   weighting for the DL model because SMOTE is applied to the
        #   sklearn-based models; in the DL path we use class_weight instead.
        # -----------------------------------------------------------------
        n_benign = int(np.sum(y_train == 0))
        n_attack = int(np.sum(y_train == 1))
        total    = n_benign + n_attack
        class_weight = {
            0: total / (2 * n_benign),   # Weight for BENIGN class
            1: total / (2 * n_attack),   # Weight for ATTACK class
        }
        logger.info("[%s] Class weights: %s", self.name, class_weight)

        # -----------------------------------------------------------------
        # Callbacks
        # -----------------------------------------------------------------
        cb_list = [
            # Early stopping: halt if validation AUC stops improving.
            # restore_best_weights=True returns the model from the epoch
            # with the highest validation AUC.
            callbacks.EarlyStopping(
                monitor="val_auc",
                patience=DL_PARAMS["patience"],
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            # ReduceLROnPlateau: halve the learning rate if validation AUC
            # plateaus for 5 epochs, helping escape local optima.
            callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=5,
                mode="max",
                verbose=1,
            ),
        ]

        # -----------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------
        logger.info(
            "[%s] Starting training | epochs=%d | batch_size=%d | patience=%d",
            self.name,
            DL_PARAMS["epochs"],
            DL_PARAMS["batch_size"],
            DL_PARAMS["patience"],
        )

        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=DL_PARAMS["epochs"],
            batch_size=DL_PARAMS["batch_size"],
            class_weight=class_weight,
            callbacks=cb_list,
            verbose=1,
        )

        self.model = model
        best_epoch = int(np.argmax(self.history.history["val_auc"])) + 1
        best_auc   = float(np.max(self.history.history["val_auc"]))
        logger.info(
            "[%s] Training complete — best_epoch: %d | best_val_auc: %.4f",
            self.name, best_epoch, best_auc,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(ATTACK | X) — shape (n_samples,).
        Squeeze removes the trailing dimension from the (n, 1) output.
        """
        return self.model.predict(X, verbose=0).squeeze()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard labels using a configurable decision threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence (Keras SavedModel format)
    # ------------------------------------------------------------------

    def save(self, path: str = MODEL_PATH) -> None:
        """
        Save the model in Keras's native format (.keras).
        This is preferred over HDF5 and TF SavedModel for Keras 3+.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info("[%s] Model saved to: %s", self.name, path)

    def load(self, path: str = MODEL_PATH) -> "DeepLearningModel":
        """Load a Keras model from disk."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        logger.info("[%s] Model loaded from: %s", self.name, path)
        return self
