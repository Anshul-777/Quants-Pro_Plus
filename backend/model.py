# =============================================================================
# backend/model.py  —  V2 Model Loading + Inference + Confidence Scoring
# =============================================================================

import os, json, logging, math
import numpy as np
import joblib
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

BASE_DIR       = os.environ.get("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "model_v2.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "scaler_v2.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "features_v2.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold_v2.json")
_V1_MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
_V1_SCALER_PATH    = os.path.join(BASE_DIR, "scaler.pkl")
_V1_FEATURES_PATH  = os.path.join(BASE_DIR, "features.json")
_V1_THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")

# =============================================================================
# ENSEMBLE MODEL CLASS — required for joblib to unpickle V2 artifacts
# =============================================================================

class EnsembleModel:
    import sys
    sys.modules['__main__'].EnsembleModel = EnsembleModel
    """XGBoost + LightGBM rank-average ensemble (must match training class)."""

    def __init__(self, xgb_model, lgb_model, weight_xgb=0.5, weight_lgb=0.5):
        self.xgb   = xgb_model
        self.lgb   = lgb_model
        self.w_xgb = weight_xgb
        self.w_lgb = weight_lgb

    def predict_proba(self, X):
        from scipy.stats import rankdata
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        r_xgb = rankdata(p_xgb) / len(p_xgb)
        r_lgb = rankdata(p_lgb) / len(p_lgb)
        blended = self.w_xgb * r_xgb + self.w_lgb * r_lgb
        return np.column_stack([1.0 - blended, blended])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        xi = self.xgb.feature_importances_ / (self.xgb.feature_importances_.sum() + 1e-12)
        li = self.lgb.feature_importances_ / (self.lgb.feature_importances_.sum() + 1e-12)
        return self.w_xgb * xi + self.w_lgb * li

    def predict_both(self, X):
        """Return both sub-model probabilities for confidence computation."""
        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        p_lgb = float(self.lgb.predict_proba(X)[0, 1])
        return p_xgb, p_lgb


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def compute_confidence(prob: float, threshold: float,
                        p_xgb: Optional[float] = None,
                        p_lgb: Optional[float] = None) -> Dict[str, Any]:
    """
    Multi-factor confidence score in [0, 1]:

    Factor 1 — Margin from threshold (how far above/below the decision boundary)
    Factor 2 — Probability extremity (probs near 0 or 1 = high confidence)
    Factor 3 — Sub-model agreement (XGB and LGB agree = higher confidence)

    Returns: { score: float, grade: str, margin: float, agreement: float }
    """
    # Factor 1: margin from threshold (normalised to 0-1 over 0-0.5 range)
    margin     = abs(prob - threshold)
    f_margin   = min(1.0, margin / 0.25)  # 0.25 from threshold → full score

    # Factor 2: extremity (distance from 0.5)
    extremity  = abs(prob - 0.5) * 2      # 0 at prob=0.5, 1 at prob=0 or 1
    f_extreme  = math.sqrt(extremity)     # sqrt to penalise less harshly

    # Factor 3: sub-model agreement
    if p_xgb is not None and p_lgb is not None:
        agree_gap = abs(p_xgb - p_lgb)   # 0 = perfect agreement
        f_agree   = max(0.0, 1.0 - agree_gap * 4)
    else:
        f_agree   = 0.7  # neutral when we don't have sub-model probs

    score = 0.50 * f_margin + 0.30 * f_extreme + 0.20 * f_agree
    score = round(min(1.0, max(0.0, score)), 4)

    if score >= 0.80:
        grade = "VERY HIGH"
    elif score >= 0.60:
        grade = "HIGH"
    elif score >= 0.40:
        grade = "MODERATE"
    elif score >= 0.20:
        grade = "LOW"
    else:
        grade = "VERY LOW"

    return {
        "score":        score,
        "score_pct":    round(score * 100, 1),
        "grade":        grade,
        "margin":       round(margin, 4),
        "sub_agreement": round(f_agree, 4),
    }


# =============================================================================
# PREDICTION REASONS
# =============================================================================

REASON_TEMPLATES = {
    "vwap_dev":  {"+": "trading above VWAP (bullish intraday momentum)",
                  "-": "trading below VWAP (bearish intraday pressure)"},
    "bb_pos":    {"+": "price in upper Bollinger Band (overbought zone)",
                  "-": "price in lower Bollinger Band (oversold zone)"},
    "rsi_14":    {"H": "RSI overbought ({v:.1f}) — momentum extended",
                  "L": "RSI oversold ({v:.1f}) — potential reversal setup",
                  "N": "RSI neutral ({v:.1f})"},
    "macd_hist": {"+": "MACD histogram positive (bullish crossover)",
                  "-": "MACD histogram negative (bearish crossover)"},
    "vol_ratio": {"+": "volume surge (+{v:.0%} above 20-bar avg) — conviction",
                  "-": "volume below average — weak conviction"},
    "rvol_20":   {"+": "elevated realized volatility — directional opportunity",
                  "-": "low volatility regime — signals less reliable"},
    "amihud_20": {"+": "high illiquidity — wide bid-ask, handle with care",
                  "N": "normal liquidity conditions"},
}

def generate_reason(features: Dict[str, float], signal: int, threshold: float) -> str:
    reasons = []

    # VWAP
    vd = features.get("vwap_dev", 0)
    if abs(vd) > 0.001:
        reasons.append(REASON_TEMPLATES["vwap_dev"]["+" if vd > 0 else "-"])

    # RSI
    rsi = features.get("rsi_14", 50)
    if rsi > 65:
        reasons.append(REASON_TEMPLATES["rsi_14"]["H"].format(v=rsi))
    elif rsi < 35:
        reasons.append(REASON_TEMPLATES["rsi_14"]["L"].format(v=rsi))

    # MACD
    mh = features.get("macd_hist", 0)
    if abs(mh) > 1e-5:
        reasons.append(REASON_TEMPLATES["macd_hist"]["+" if mh > 0 else "-"])

    # Bollinger
    bb = features.get("bb_pos", 0)
    if abs(bb) > 0.3:
        reasons.append(REASON_TEMPLATES["bb_pos"]["+" if bb > 0 else "-"])

    # Volume
    vr = features.get("vol_ratio", 1.0)
    if vr > 1.5:
        reasons.append(REASON_TEMPLATES["vol_ratio"]["+"].format(v=vr - 1))
    elif vr < 0.6:
        reasons.append(REASON_TEMPLATES["vol_ratio"]["-"])

    # Amihud (high illiquidity warning)
    am = features.get("amihud_20", 0)
    if am > 0.01:
        reasons.append(REASON_TEMPLATES["amihud_20"]["+"])

    if not reasons:
        reasons.append(f"Ensemble model consensus (XGBoost × 0.7 + LightGBM × 0.3)")

    action = "BUY signal" if signal == 1 else "HOLD signal"
    return f"{action} — {'; '.join(reasons[:3])}"


# =============================================================================
# TRADING MODEL WRAPPER
# =============================================================================

class TradingModel:
    def __init__(self):
        self.model      = None
        self.scaler     = None
        self.features:  List[str] = []
        self.threshold: float = 0.5
        self.metadata:  Dict[str, Any] = {}
        self.loaded:    bool = False
        self.model_version: str = "unknown"

    def _resolve_paths(self):
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH, SCALER_PATH, FEATURES_PATH, THRESHOLD_PATH, "v2"
        if os.path.exists(_V1_MODEL_PATH):
            logger.warning("V2 not found — falling back to V1")
            return _V1_MODEL_PATH, _V1_SCALER_PATH, _V1_FEATURES_PATH, _V1_THRESHOLD_PATH, "v1"
        raise FileNotFoundError(
            f"No model artifact found. Expected: {MODEL_PATH}\n"
            "Upload model_v2.pkl, scaler_v2.pkl, features_v2.json, threshold_v2.json"
        )

    def load(self) -> "TradingModel":
        m_path, s_path, f_path, t_path, version = self._resolve_paths()
        logger.info(f"Loading {version} model from {os.path.dirname(m_path)}")
        self.model   = joblib.load(m_path)
        self.scaler  = joblib.load(s_path)
        with open(f_path) as fh: feat_data = json.load(fh)
        self.features = feat_data["features"]
        with open(t_path) as fh: self.metadata = json.load(fh)
        self.threshold    = float(self.metadata["threshold"])
        self.model_version = version
        self.loaded = True
        logger.info(f"  ✓ {type(self.model).__name__}  features={len(self.features)}  τ={self.threshold:.4f}")
        return self

    def predict(self, feature_vector: Dict[str, float]) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded.")

        x = np.array(
            [float(feature_vector.get(f, 0.0)) for f in self.features],
            dtype=np.float32,
        ).reshape(1, -1)

        if np.isnan(x).any():
            nan_feats = [f for f, v in zip(self.features, x[0]) if np.isnan(v)]
            return {"probability": None, "signal": 0, "confidence": {"score": 0, "grade": "VERY LOW"},
                    "ready": False, "error": f"NaN in features: {nan_feats[:5]}"}

        x_scaled = self.scaler.transform(x)
        prob = float(self.model.predict_proba(x_scaled)[0, 1])

        # Sub-model probs for confidence
        p_xgb, p_lgb = None, None
        if isinstance(self.model, EnsembleModel):
            try: p_xgb, p_lgb = self.model.predict_both(x_scaled)
            except Exception: pass

        signal = 1 if prob > self.threshold else 0

        if prob >= 0.65:    label = "STRONG BUY"
        elif prob > self.threshold: label = "BUY"
        elif prob >= self.threshold - 0.04: label = "WATCH"
        else: label = "HOLD"

        confidence = compute_confidence(prob, self.threshold, p_xgb, p_lgb)
        reason = generate_reason(feature_vector, signal, self.threshold)

        return {
            "probability":      round(prob, 6),
            "probability_pct":  round(prob * 100, 2),
            "signal":           signal,
            "signal_label":     label,
            "threshold":        self.threshold,
            "confidence":       confidence,
            "reason":           reason,
            "version":          self.model_version,
            "ready":            True,
            "error":            None,
        }

    def feature_importances(self) -> Dict[str, float]:
        if not self.loaded: return {}
        try:
            raw   = self.model.feature_importances_
            total = raw.sum() + 1e-12
            return {f: float(v / total) for f, v in zip(self.features, raw)}
        except Exception: return {}

    def info(self) -> Dict[str, Any]:
        return {
            "loaded":          self.loaded,
            "model_version":   self.model_version,
            "model_type":      type(self.model).__name__ if self.model else None,
            "n_features":      len(self.features),
            "features":        self.features,
            "threshold":       self.threshold,
            "val_sharpe":      self.metadata.get("val_sharpe"),
            "val_metrics":     self.metadata.get("val_metrics"),
            "test_metrics":    self.metadata.get("test_metrics"),
            "shap_top10":      self.metadata.get("shap_top10", []),
            "trained_at":      self.metadata.get("trained_at"),
            "ensemble_weight_xgb": self.metadata.get("ensemble_weight_xgb"),
        }


_model_instance: Optional[TradingModel] = None

def get_model() -> TradingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = TradingModel()
    if not _model_instance.loaded:
        try: _model_instance.load()
        except Exception as e:
            logger.warning(f"Model load failed: {e} — predictions will return placeholder responses")
    return _model_instance
