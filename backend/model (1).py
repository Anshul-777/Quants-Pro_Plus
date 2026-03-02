# =============================================================================
# backend/model.py  —  Model Loading, EnsembleModel, and Inference
# Compatible with V1 (plain XGBClassifier) and V2 (EnsembleModel artifact)
# =============================================================================

import os
import json
import logging
import numpy as np
import joblib
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Artifact paths — override via MODEL_DIR env var on Render
BASE_DIR       = os.environ.get("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "model_v2.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "scaler_v2.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "features_v2.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold_v2.json")

# Fallback to v1 artifact names if v2 not found
_V1_MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
_V1_SCALER_PATH    = os.path.join(BASE_DIR, "scaler.pkl")
_V1_FEATURES_PATH  = os.path.join(BASE_DIR, "features.json")
_V1_THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")


# =============================================================================
# ENSEMBLE MODEL CLASS — must exist here for joblib to unpickle V2 artifacts
# This is an exact structural copy of the class used in advanced_model_v2.py
# =============================================================================

class EnsembleModel:
    """
    Rank-average ensemble of XGBoost + LightGBM.
    Rank averaging normalises each model's output distribution before blending,
    making the combination more robust than simple probability averaging.

    This class must be present in any process that loads model_v2.pkl via joblib.
    """

    def __init__(self, xgb_model, lgb_model, weight_xgb=0.5, weight_lgb=0.5):
        self.xgb     = xgb_model
        self.lgb     = lgb_model
        self.w_xgb   = weight_xgb
        self.w_lgb   = weight_lgb

    def predict_proba(self, X):
        from scipy.stats import rankdata

        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]

        # Rank-transform each to [0, 1] then blend
        r_xgb = rankdata(p_xgb) / len(p_xgb)
        r_lgb = rankdata(p_lgb) / len(p_lgb)
        blended = self.w_xgb * r_xgb + self.w_lgb * r_lgb

        # Return (n, 2) array matching sklearn convention: [P(0), P(1)]
        return np.column_stack([1.0 - blended, blended])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        xi = self.xgb.feature_importances_
        li = self.lgb.feature_importances_
        xi_n = xi / (xi.sum() + 1e-12)
        li_n = li / (li.sum() + 1e-12)
        return (xi_n * self.w_xgb + li_n * self.w_lgb)


# =============================================================================
# TRADING MODEL WRAPPER
# =============================================================================

class TradingModel:
    """
    Wraps the pickled model + scaler + feature list + threshold.
    Exposes a single predict(feature_vector_dict) method.
    Thread-safe for concurrent read-only inference.
    """

    def __init__(self):
        self.model      = None
        self.scaler     = None
        self.features:  List[str] = []
        self.threshold: float = 0.5
        self.metadata:  Dict[str, Any] = {}
        self.loaded:    bool = False
        self.model_version: str = "unknown"

    # ── Loading ──────────────────────────────────────────────────────────────

    def _resolve_paths(self):
        """Return (model, scaler, features, threshold) paths — prefer V2."""
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH, SCALER_PATH, FEATURES_PATH, THRESHOLD_PATH, "v2"
        if os.path.exists(_V1_MODEL_PATH):
            logger.warning("V2 artifacts not found — falling back to V1 artifacts")
            return _V1_MODEL_PATH, _V1_SCALER_PATH, _V1_FEATURES_PATH, _V1_THRESHOLD_PATH, "v1"
        raise FileNotFoundError(
            f"No model artifact found. Expected one of:\n"
            f"  V2: {MODEL_PATH}\n  V1: {_V1_MODEL_PATH}\n"
            f"Upload model_v2.pkl (or model.pkl) to the backend directory."
        )

    def load(self) -> "TradingModel":
        m_path, s_path, f_path, t_path, version = self._resolve_paths()
        logger.info(f"Loading {version} model artifacts from {os.path.dirname(m_path)}")

        # Model (EnsembleModel or XGBClassifier — both handled transparently)
        self.model = joblib.load(m_path)
        logger.info(f"  ✓ Model loaded  ({type(self.model).__name__})")

        # Scaler
        if not os.path.exists(s_path):
            raise FileNotFoundError(f"Scaler not found: {s_path}")
        self.scaler = joblib.load(s_path)
        logger.info("  ✓ Scaler loaded")

        # Feature list
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Features JSON not found: {f_path}")
        with open(f_path) as fh:
            feat_data = json.load(fh)
        self.features = feat_data["features"]
        logger.info(f"  ✓ {len(self.features)} features loaded")

        # Threshold + metadata
        if not os.path.exists(t_path):
            raise FileNotFoundError(f"Threshold JSON not found: {t_path}")
        with open(t_path) as fh:
            self.metadata = json.load(fh)
        self.threshold = float(self.metadata["threshold"])
        logger.info(f"  ✓ Threshold τ = {self.threshold:.4f}")

        self.model_version = version
        self.loaded = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, feature_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Accept a dict {feature_name: float_value} and return a prediction dict.
        Only the features listed in self.features are used; extras are ignored.
        Missing features default to 0.0 (safe for scaled inputs).
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded — call load() first.")

        x = np.array(
            [float(feature_vector.get(f, 0.0)) for f in self.features],
            dtype=np.float32,
        ).reshape(1, -1)

        if np.isnan(x).any():
            nan_feats = [f for f, v in zip(self.features, x[0]) if np.isnan(v)]
            return {
                "probability": None, "signal": 0,
                "threshold": self.threshold, "ready": False,
                "error": f"NaN in features: {nan_feats[:5]}",
            }

        x_scaled = self.scaler.transform(x)
        prob      = float(self.model.predict_proba(x_scaled)[0, 1])
        signal    = 1 if prob > self.threshold else 0

        # Classify signal strength
        if prob >= 0.65:
            signal_label = "STRONG BUY"
        elif prob > self.threshold:
            signal_label = "BUY"
        elif prob >= self.threshold - 0.03:
            signal_label = "WATCH"
        else:
            signal_label = "HOLD"

        return {
            "probability":   round(prob, 6),
            "probability_pct": round(prob * 100, 2),
            "signal":        signal,
            "signal_label":  signal_label,
            "threshold":     self.threshold,
            "ready":         True,
            "error":         None,
        }

    def feature_importances(self) -> Dict[str, float]:
        """Return normalized feature importances from the underlying model."""
        if not self.loaded:
            return {}
        try:
            if hasattr(self.model, "feature_importances_"):
                raw = self.model.feature_importances_
                total = raw.sum() + 1e-12
                return {f: float(v / total) for f, v in zip(self.features, raw)}
        except Exception:
            pass
        return {}

    def info(self) -> Dict[str, Any]:
        return {
            "loaded":        self.loaded,
            "model_version": self.model_version,
            "model_type":    type(self.model).__name__ if self.model else None,
            "n_features":    len(self.features),
            "features":      self.features,
            "threshold":     self.threshold,
            "val_sharpe":    self.metadata.get("val_sharpe"),
            "val_metrics":   self.metadata.get("val_metrics"),
            "test_metrics":  self.metadata.get("test_metrics"),
            "tickers":       self.metadata.get("tickers", []),
            "trained_at":    self.metadata.get("trained_at"),
            "shap_top10":    self.metadata.get("shap_top10", []),
            "ensemble_weight_xgb": self.metadata.get("ensemble_weight_xgb"),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_model_instance: Optional[TradingModel] = None


def get_model() -> TradingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = TradingModel()
    if not _model_instance.loaded:
        try:
            _model_instance.load()
        except Exception as e:
            logger.warning(f"Model load failed: {e} — predictions disabled.")
    return _model_instance
