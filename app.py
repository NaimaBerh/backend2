# =============================================================================
#  FakeProfileDetector - Backend Flask UNIFIÉ pour Render
# =============================================================================
#  Modèles actifs :
#    - job_text_lstm   : LSTM TFLite pour détection de fausses offres d'emploi
#    - insta_xgboost   : XGBClassifier (Insta_xgboost.joblib) pour Instagram
#    - linkedin_svc    : SVC + StandardScaler (Linkd_roberta_SVC_model.pkl)
#
#  Plateformes désactivées (en cours de développement) :
#    - twitter, github  ->  HTTP 501 + message dédié
#
#  Endpoints publics :
#    GET  /                    -> infos API
#    GET  /health              -> healthcheck Render
#    GET  /api/models          -> liste des modèles ML disponibles
#    POST /api/analyze         -> analyse d'un profil (features manuelles)
#    POST /api/analyze-job     -> analyse LSTM d'une offre d'emploi (texte brut)
#    POST /api/extract-url     -> extraction brute d'un profil/offre depuis une URL
#    POST /api/analyze-url     -> extraction + analyse complète depuis une URL
# =============================================================================

import os
import re
import json
import time
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Réduction du bruit TensorFlow avant import
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------------------------------------------------------- #
#  Configuration globale                                                         #
# ----------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LSTM job text
MODEL_PATH = os.path.join(BASE_DIR, "fake_job_lstm_model.tflite")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
MAX_SEQUENCE_LENGTH = 200
LSTM_THRESHOLD = float(os.environ.get("LSTM_THRESHOLD", "0.7"))

# Instagram XGBoost
INSTA_MODEL_PATH = os.path.join(BASE_DIR, "Insta_xgboost.joblib")

# LinkedIn SVC RoBERTa
LINKD_MODEL_PATH = os.path.join(BASE_DIR, "Linkd_roberta_SVC_model.pkl")

# Plateformes désactivées
DISABLED_PLATFORMS = {"twitter", "github"}
DISABLED_MESSAGE = "Fonctionnalité en cours de développement. Disponible prochainement."

# Liste des origines autorisées à appeler l'API.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# Port (Render impose la variable d'environnement PORT)
PORT = int(os.environ.get("PORT", "5000"))

# User-Agent pour le scraping
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr,en;q=0.8",
}
DEFAULT_TIMEOUT = 20

# ----------------------------------------------------------------------------- #
#  Flask app                                                                     #
# ----------------------------------------------------------------------------- #

app = Flask(__name__)

if ALLOWED_ORIGINS == ["*"]:
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
else:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)


# ----------------------------------------------------------------------------- #
#  Modèles ML (métriques + chargement)                                           #
# ----------------------------------------------------------------------------- #

MODEL_PERFORMANCE: Dict[str, Dict[str, Any]] = {
    "job_text_lstm": {
        "name": "job(text) LSTM",
        "platform": "job",
        "precision": None, "recall": None, "f1_score": None, "roc_auc": None,
    },
    "insta_xgboost": {
        "name": "Instagram XGBoost",
        "platform": "instagram",
        "precision": 0.980, "recall": 0.972, "f1_score": 0.976, "roc_auc": 0.985,
    },
    "linkedin_svc": {
        "name": "LinkedIn RoBERTa-SVC",
        "platform": "linkedin",
        "precision": 0.981, "recall": 0.981, "f1_score": 0.980, "roc_auc": 0.996,
    },
}

# --- LSTM job text (lazy) ---------------------------------------------------- #
_lstm_interpreter = None
_lstm_tokenizer = None
_lstm_input_details = None
_lstm_output_details = None
_lstm_load_error: Optional[str] = None


def _load_lstm_once() -> None:
    """Charge le modèle LSTM TFLite et le tokenizer Keras (une seule fois)."""
    global _lstm_interpreter, _lstm_tokenizer, _lstm_input_details, _lstm_output_details, _lstm_load_error

    if _lstm_interpreter is not None or _lstm_load_error is not None:
        return

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle LSTM introuvable : {MODEL_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer introuvable : {TOKENIZER_PATH}")

        import tensorflow as tf  # import différé

        with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
            _lstm_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

        _lstm_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _lstm_interpreter.allocate_tensors()
        _lstm_input_details = _lstm_interpreter.get_input_details()
        _lstm_output_details = _lstm_interpreter.get_output_details()
    except Exception as exc:  # noqa: BLE001
        _lstm_load_error = f"{type(exc).__name__}: {exc}"
        print(f"[LSTM] Chargement impossible : {_lstm_load_error}", flush=True)


# --- Instagram XGBoost (lazy) ------------------------------------------------ #
_insta_model = None
_insta_features: List[str] = [
    "userFollowerCount", "userFollowingCount", "userBiographyLength",
    "userMediaCount", "userHasProfilPic", "userIsPrivate",
    "usernameDigitCount", "usernameLength",
]
_insta_load_error: Optional[str] = None


def _load_insta_once() -> None:
    global _insta_model, _insta_load_error
    if _insta_model is not None or _insta_load_error is not None:
        return
    try:
        if not os.path.exists(INSTA_MODEL_PATH):
            raise FileNotFoundError(f"Modèle Instagram introuvable : {INSTA_MODEL_PATH}")
        import joblib  # import différé
        _insta_model = joblib.load(INSTA_MODEL_PATH)
        # Récupère le nom des features depuis le modèle si disponible
        try:
            if hasattr(_insta_model, "feature_names_in_"):
                names = list(_insta_model.feature_names_in_)
                if names:
                    _insta_features[:] = names
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        _insta_load_error = f"{type(exc).__name__}: {exc}"
        print(f"[Insta XGBoost] Chargement impossible : {_insta_load_error}", flush=True)


# --- LinkedIn SVC (lazy) ----------------------------------------------------- #
_linkd_bundle: Optional[Dict[str, Any]] = None
_linkd_classifier = None
_linkd_scaler = None
_linkd_features: List[str] = []
_linkd_class_names: Dict[int, str] = {0: "genuine", 1: "fake"}
_linkd_load_error: Optional[str] = None

# Les 17 features numériques structurées attendues par le SVC LinkedIn.
LINKD_NUMERIC_FEATURES: List[str] = [
    "num_experiences", "num_educations", "num_licenses", "num_volunteering",
    "num_skills", "num_recommendations", "num_projects", "num_publications",
    "num_courses", "num_honors", "num_scores", "num_languages",
    "num_organizations", "num_interests", "num_activities",
    "connections", "followers",
]


def _load_linkd_once() -> None:
    global _linkd_bundle, _linkd_classifier, _linkd_scaler, _linkd_features, _linkd_class_names, _linkd_load_error
    if _linkd_bundle is not None or _linkd_load_error is not None:
        return
    try:
        if not os.path.exists(LINKD_MODEL_PATH):
            raise FileNotFoundError(f"Modèle LinkedIn introuvable : {LINKD_MODEL_PATH}")
        # Le fichier est un pickle joblib (compression LZ4/ZLIB).
        try:
            import joblib  # import différé
            bundle = joblib.load(LINKD_MODEL_PATH)
        except Exception:
            with open(LINKD_MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)

        if not isinstance(bundle, dict):
            raise ValueError("Format de bundle LinkedIn invalide (dict attendu).")

        _linkd_bundle = bundle
        _linkd_classifier = bundle.get("classifier")
        _linkd_scaler = bundle.get("scaler")
        feats = bundle.get("feature_columns")
        if isinstance(feats, (list, tuple)) and feats:
            _linkd_features = list(feats)
        elif _linkd_scaler is not None and hasattr(_linkd_scaler, "feature_names_in_"):
            _linkd_features = list(_linkd_scaler.feature_names_in_)
        else:
            raise ValueError("Aucune liste de features disponible dans le bundle LinkedIn.")

        cn = bundle.get("class_names")
        if isinstance(cn, dict):
            _linkd_class_names = {int(k): str(v) for k, v in cn.items()}

        if _linkd_classifier is None or _linkd_scaler is None:
            raise ValueError("Bundle LinkedIn incomplet (classifier ou scaler manquant).")
    except Exception as exc:  # noqa: BLE001
        _linkd_load_error = f"{type(exc).__name__}: {exc}"
        print(f"[LinkedIn SVC] Chargement impossible : {_linkd_load_error}", flush=True)


# ----------------------------------------------------------------------------- #
#  Helpers communs                                                               #
# ----------------------------------------------------------------------------- #

def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    try:
        txt = str(value).strip().replace(",", "").replace("\xa0", "").replace(" ", "")
        if txt == "":
            return default
        mult = 1
        if txt[-1].upper() == "K":
            mult, txt = 1_000, txt[:-1]
        elif txt[-1].upper() == "M":
            mult, txt = 1_000_000, txt[:-1]
        elif txt[-1].upper() == "B":
            mult, txt = 1_000_000_000, txt[:-1]
        return int(float(txt) * mult)
    except (ValueError, TypeError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("1", "true", "yes", "oui", "on")


def _count_digits(text: str) -> int:
    return sum(1 for c in (text or "") if c.isdigit())


# ----------------------------------------------------------------------------- #
#  Instagram XGBoost                                                             #
# ----------------------------------------------------------------------------- #

def _features_to_insta_vector(features: Dict[str, Any]) -> np.ndarray:
    """Construit un vecteur (1, 8) compatible avec _insta_features."""
    f = features or {}
    # mapping souple : on accepte les noms du modèle ET les noms côté frontend
    follower = _safe_int(f.get("userFollowerCount", f.get("follower_count")))
    following = _safe_int(f.get("userFollowingCount", f.get("following_count")))
    bio_len = _safe_int(f.get("userBiographyLength", f.get("bio_length")))
    media = _safe_int(f.get("userMediaCount", f.get("post_count")))
    has_pic = 1 if _safe_bool(f.get("userHasProfilPic", f.get("has_profile_pic", True))) else 0
    is_priv = 1 if _safe_bool(f.get("userIsPrivate", f.get("is_private", False))) else 0
    digits = _safe_int(f.get("usernameDigitCount", f.get("username_digits")))
    uname_len = _safe_int(f.get("usernameLength", f.get("username_length")))

    raw = {
        "userFollowerCount": follower,
        "userFollowingCount": following,
        "userBiographyLength": bio_len,
        "userMediaCount": media,
        "userHasProfilPic": has_pic,
        "userIsPrivate": is_priv,
        "usernameDigitCount": digits,
        "usernameLength": uname_len,
    }
    # respecte l'ordre défini par le modèle
    vector = [float(raw.get(name, 0)) for name in _insta_features]
    return np.array(vector, dtype=np.float32).reshape(1, -1)


def predict_instagram(features: Dict[str, Any]) -> Dict[str, Any]:
    _load_insta_once()
    if _insta_load_error or _insta_model is None:
        raise RuntimeError(f"Modèle Instagram indisponible : {_insta_load_error or 'non chargé'}")

    X = _features_to_insta_vector(features)
    try:
        proba = _insta_model.predict_proba(X)[0]
        prob_fake = float(proba[1]) if len(proba) > 1 else float(proba[0])
    except Exception:
        # fallback : decision_function -> sigmoid
        score = float(_insta_model.predict(X)[0])
        prob_fake = 1.0 / (1.0 + np.exp(-score))

    prob_fake = max(0.0, min(1.0, prob_fake))
    risk = int(round(prob_fake * 100))
    label = "fake" if prob_fake >= 0.5 else "genuine"
    confidence = prob_fake if label == "fake" else (1 - prob_fake)

    # SHAP simulés : importances pondérées des features
    follower = _safe_int(features.get("userFollowerCount", features.get("follower_count")))
    following = _safe_int(features.get("userFollowingCount", features.get("following_count")))
    posts = _safe_int(features.get("userMediaCount", features.get("post_count")))
    digits = _safe_int(features.get("usernameDigitCount", features.get("username_digits")))
    bio_len = _safe_int(features.get("userBiographyLength", features.get("bio_length")))
    has_pic = _safe_bool(features.get("userHasProfilPic", features.get("has_profile_pic", True)))
    multiplier = 1 if label == "fake" else -1
    ratio = follower / (following + 1) if following >= 0 else 0
    shap_values = {
        "follower_following_ratio": round((0.18 if ratio < 0.5 else -0.12) * multiplier, 3),
        "post_count": round((0.11 if posts < 5 else -0.08) * multiplier, 3),
        "bio_length": round((0.09 if bio_len < 20 else -0.05) * multiplier, 3),
        "has_profile_pic": round((-0.07 if has_pic else 0.12) * multiplier, 3),
        "username_digits": round((0.14 if digits > 4 else -0.04) * multiplier, 3),
    }

    return {
        "model": "insta_xgboost",
        "platform": "instagram",
        "prediction_score": round(prob_fake, 6),
        "risk_score": risk,
        "classification": label,
        "confidence": round(confidence, 3),
        "metrics": MODEL_PERFORMANCE["insta_xgboost"],
        "shap_values": shap_values,
        "features": {
            "userFollowerCount": follower,
            "userFollowingCount": following,
            "userBiographyLength": bio_len,
            "userMediaCount": posts,
            "userHasProfilPic": bool(has_pic),
            "userIsPrivate": _safe_bool(features.get("userIsPrivate", features.get("is_private", False))),
            "usernameDigitCount": digits,
            "usernameLength": _safe_int(features.get("usernameLength", features.get("username_length"))),
            "follower_following_ratio": round(ratio, 4),
        },
    }


# ----------------------------------------------------------------------------- #
#  LinkedIn SVC (RoBERTa-PCA + numeric)                                          #
# ----------------------------------------------------------------------------- #

def _features_to_linkd_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Construit un vecteur (1, N) pour le SVC LinkedIn.

    Le modèle attend 167 dimensions = 150 PCA RoBERTa + 17 numeric features.
    Les dimensions PCA ne sont pas calculables sans le pipeline d'embedding
    d'origine ; nous les remplissons avec 0.0 (valeur médiane neutre après
    scaling). Les 17 features numériques structurées sont récupérées depuis
    le payload fourni par le frontend (mapping souple pour rester compatible
    avec les anciens noms côté UI).
    """
    f = features or {}
    # mapping flexible des features numériques
    numeric_aliases: Dict[str, List[str]] = {
        "num_experiences":     ["num_experiences", "experiences", "experience_count"],
        "num_educations":      ["num_educations", "educations", "education_count"],
        "num_licenses":        ["num_licenses", "licenses", "certifications"],
        "num_volunteering":    ["num_volunteering", "volunteering"],
        "num_skills":          ["num_skills", "skills", "skill_count"],
        "num_recommendations": ["num_recommendations", "recommendations"],
        "num_projects":        ["num_projects", "projects"],
        "num_publications":    ["num_publications", "publications"],
        "num_courses":         ["num_courses", "courses"],
        "num_honors":          ["num_honors", "honors", "awards"],
        "num_scores":          ["num_scores", "scores", "test_scores"],
        "num_languages":       ["num_languages", "languages"],
        "num_organizations":   ["num_organizations", "organizations"],
        "num_interests":       ["num_interests", "interests"],
        "num_activities":      ["num_activities", "activities", "post_count"],
        "connections":         ["connections", "connection_count", "following_count"],
        "followers":           ["followers", "follower_count"],
    }

    def pick(name: str) -> float:
        for key in numeric_aliases.get(name, [name]):
            if key in f and f[key] not in (None, ""):
                return _safe_float(f[key])
        return 0.0

    numeric_values = {name: pick(name) for name in LINKD_NUMERIC_FEATURES}

    # construit dans l'ordre exact du scaler/SVC
    row: List[float] = []
    for col in _linkd_features:
        if col.startswith("pca_"):
            # PCA dim non calculable sans le pipeline RoBERTa+PCA d'origine -> 0
            row.append(0.0)
        elif col in numeric_values:
            row.append(float(numeric_values[col]))
        else:
            row.append(0.0)
    return np.array(row, dtype=np.float32).reshape(1, -1), numeric_values


def predict_linkedin(features: Dict[str, Any]) -> Dict[str, Any]:
    _load_linkd_once()
    if _linkd_load_error or _linkd_classifier is None or _linkd_scaler is None:
        raise RuntimeError(f"Modèle LinkedIn indisponible : {_linkd_load_error or 'non chargé'}")

    X_raw, numeric_values = _features_to_linkd_vector(features)
    X = _linkd_scaler.transform(X_raw)

    try:
        proba = _linkd_classifier.predict_proba(X)[0]
        # repère la classe "fake" via class_names si dispo
        fake_idx = 1
        for k, v in _linkd_class_names.items():
            if str(v).lower() == "fake":
                # k est la valeur de classes_, on cherche son index
                classes = list(getattr(_linkd_classifier, "classes_", [0, 1]))
                if k in classes:
                    fake_idx = classes.index(k)
                break
        prob_fake = float(proba[fake_idx]) if len(proba) > fake_idx else float(proba[-1])
    except Exception:
        # fallback : decision_function -> sigmoid
        score = float(_linkd_classifier.decision_function(X)[0])
        prob_fake = 1.0 / (1.0 + np.exp(-score))

    prob_fake = max(0.0, min(1.0, prob_fake))
    risk = int(round(prob_fake * 100))
    label = "fake" if prob_fake >= 0.5 else "genuine"
    confidence = prob_fake if label == "fake" else (1 - prob_fake)

    # SHAP simulés pour explicabilité (basé sur les 17 features numériques)
    multiplier = 1 if label == "fake" else -1
    connections = numeric_values.get("connections", 0)
    followers = numeric_values.get("followers", 0)
    skills = numeric_values.get("num_skills", 0)
    experiences = numeric_values.get("num_experiences", 0)
    educations = numeric_values.get("num_educations", 0)
    shap_values = {
        "connections": round((0.18 if connections < 50 else -0.12) * multiplier, 3),
        "followers": round((0.10 if followers < 30 else -0.06) * multiplier, 3),
        "num_skills": round((0.11 if skills < 3 else -0.07) * multiplier, 3),
        "num_experiences": round((0.13 if experiences < 1 else -0.08) * multiplier, 3),
        "num_educations": round((0.09 if educations < 1 else -0.05) * multiplier, 3),
    }

    return {
        "model": "linkedin_svc",
        "platform": "linkedin",
        "prediction_score": round(prob_fake, 6),
        "risk_score": risk,
        "classification": label,
        "confidence": round(confidence, 3),
        "metrics": MODEL_PERFORMANCE["linkedin_svc"],
        "shap_values": shap_values,
        "features": {**numeric_values, "follower_count": followers, "following_count": connections},
        "notice": (
            "Les dimensions PCA RoBERTa ne sont pas recalculables côté serveur sans "
            "le pipeline d'embedding d'origine ; elles sont neutralisées (0). "
            "La prédiction repose principalement sur les 17 features numériques structurées."
        ),
    }


# ----------------------------------------------------------------------------- #
#  LSTM job(text) — preprocessing & analyse signaux                              #
# ----------------------------------------------------------------------------- #

SUSPICIOUS_PATTERNS: List[str] = [
    r"urgent(?:ly)?", r"telegram", r"whatsapp", r"wire\s+transfer",
    r"registration\s+fee", r"upfront", r"crypto", r"bitcoin",
    r"no\s+experience", r"limited\s+slots?", r"apply\s+now",
    r"immediate\s+start", r"data\s+entry", r"work\s+from\s+home",
    r"guaranteed\s+income",
]
URL_REGEX = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
TOKEN_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


def _lstm_preprocess(text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # import différé
    sequence = _lstm_tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32")


def analyze_text_signals(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    matches: List[str] = []
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            matches.append(pattern.replace(r"\s+", " ").replace("(?:ly)?", ""))

    tokens = TOKEN_REGEX.findall(text)
    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    upper_ratio = round((upper / alpha), 4) if alpha else 0.0

    return {
        "text_length": len(text),
        "token_count": len(tokens),
        "suspicious_keyword_count": len(matches),
        "suspicious_keywords": matches[:8],
        "url_count": len(URL_REGEX.findall(text)),
        "email_count": len(EMAIL_REGEX.findall(text)),
        "uppercase_ratio": upper_ratio,
    }


def predict_job_text(text: str) -> Dict[str, Any]:
    _load_lstm_once()
    if _lstm_load_error is not None or _lstm_interpreter is None:
        raise RuntimeError(
            f"Le modèle LSTM n'est pas disponible sur ce déploiement : {_lstm_load_error or 'non chargé'}"
        )

    input_data = _lstm_preprocess(text)
    _lstm_interpreter.set_tensor(_lstm_input_details[0]["index"], input_data)
    _lstm_interpreter.invoke()
    pred = float(_lstm_interpreter.get_tensor(_lstm_output_details[0]["index"])[0][0])

    signals = analyze_text_signals(text)
    pred = max(0.0, min(1.0, pred))
    risk = int(round(pred * 100))
    label = "fake" if pred > LSTM_THRESHOLD else "genuine"
    confidence = pred if label == "fake" else (1 - pred)

    features = {
        "input_mode": "job_text",
        "job_text_excerpt": (re.sub(r"\s+", " ", text)).strip()[:280],
        "job_text_length": signals["text_length"],
        "token_count": signals["token_count"],
        "suspicious_keyword_count": signals["suspicious_keyword_count"],
        "suspicious_keywords": signals["suspicious_keywords"],
        "url_count": signals["url_count"],
        "email_count": signals["email_count"],
        "uppercase_ratio": signals["uppercase_ratio"],
        "prediction_score": round(pred, 6),
        "threshold": LSTM_THRESHOLD,
    }

    shap_values = {
        "prediction_score": round(pred - 0.5, 3),
        "suspicious_keyword_count": round(min(signals["suspicious_keyword_count"] * 0.08, 0.4), 3),
        "url_count": round(min(signals["url_count"] * 0.07, 0.21), 3),
        "email_count": round(min(signals["email_count"] * 0.05, 0.15), 3),
        "uppercase_ratio": round(signals["uppercase_ratio"] * 0.5, 3),
    }

    return {
        "model": "job_text_lstm",
        "platform": "job",
        "prediction_score": round(pred, 6),
        "risk_score": risk,
        "classification": label,
        "confidence": round(confidence, 3),
        "threshold": LSTM_THRESHOLD,
        "metrics": MODEL_PERFORMANCE["job_text_lstm"],
        "signals": signals,
        "features": features,
        "shap_values": shap_values,
    }


# ----------------------------------------------------------------------------- #
#  URL extractor                                                                 #
# ----------------------------------------------------------------------------- #

JOB_URL_HINTS = [
    "/jobs/", "/job/", "jobposting", "job-posting", "careers", "carriere",
    "emploi", "offre-emploi", "offres-emploi", "recrutement", "vacancy",
    "vacancies", "hiring", "apply", "postuler",
]
JOB_DOMAINS = {
    "indeed.com", "linkedin.com/jobs", "glassdoor.com", "monster.com",
    "pole-emploi.fr", "francetravail.fr", "welcometothejungle.com",
    "jobteaser.com", "apec.fr", "hellowork.com",
}


def _clean_text(text: Optional[str], limit: int = 600) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:limit]


# def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
   # try:
    #    r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout, allow_redirects=True)
    #    if r.status_code == 200 or r.status_code in (401, 403, 404):
   #         return r
   # except requests.RequestException:
  #      return None
 #   return None
# ==============
def _http_get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    try:
        session = requests.Session()

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.instagram.com/",
            "Connection": "keep-alive"
        }

        session.headers.update(headers)

        r = session.get(url, timeout=timeout, allow_redirects=True)

        print("STATUS:", r.status_code)

        # نقبل فقط 200
        if r.status_code == 200:
            return r
        else:
            return None

    except Exception as e:
        print("ERROR HTTP:", e)
        return None
#=====================


def _extract_og_meta(soup: BeautifulSoup) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name")
        if not key:
            continue
        if key.startswith(("og:", "twitter:", "article:", "profile:")):
            value = tag.get("content")
            if value:
                meta[key] = value.strip()
    desc = soup.find("meta", attrs={"name": "description"})
    if desc and desc.get("content"):
        meta.setdefault("description", desc["content"].strip())
    return meta


def _extract_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "")
            if isinstance(payload, list):
                results.extend(item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                results.append(payload)
        except (ValueError, TypeError):
            continue
    return results


def detect_platform(url: str) -> Tuple[str, str]:
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
    except ValueError:
        return ("generic", "invalid_url")
    host = (parsed.netloc or "").lower().lstrip("www.")
    path = (parsed.path or "").lower()
    full = host + path

    for jd in JOB_DOMAINS:
        if jd in full:
            return ("job", "job_domain")
    if any(h in path for h in JOB_URL_HINTS):
        return ("job", "job_path_hint")
    if "instagram.com" in host:
        return ("instagram", "domain")
    if "twitter.com" in host or host == "x.com" or host.endswith(".x.com"):
        return ("twitter", "domain")
    if "linkedin.com" in host:
        return ("linkedin", "domain")
    if "github.com" in host:
        return ("github", "domain")
    return ("generic", "fallback")


def extract_instagram(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL Instagram.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    og_desc = meta.get("og:description") or meta.get("description") or ""
    followers = following = posts = 0
    m = re.search(
        r"([\d.,KMBkmb\xa0 ]+)\s*Followers?,?\s*([\d.,KMBkmb\xa0 ]+)\s*Following,?\s*([\d.,KMBkmb\xa0 ]+)\s*Posts?",
        og_desc, re.I,
    )
    if m:
        followers, following, posts = _safe_int(m.group(1)), _safe_int(m.group(2)), _safe_int(m.group(3))
    parsed = urlparse(url)
    username = next((p for p in parsed.path.split("/") if p), "")
    return {
        "username": username,
        "display_name": meta.get("og:title", "").replace(f"(@{username})", "").strip(" -•"),
        "bio": _clean_text(og_desc.split(" - ")[-1] if " - " in og_desc else og_desc, 400),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": followers,
        "following_count": following,
        "post_count": posts,
        "is_private": "is_private" in r.text.lower()[:40000] and "true" in r.text.lower()[:40000],
        "is_verified": "is_verified" in r.text.lower()[:40000],
        "account_type": "instagram_user",
    }


def extract_linkedin(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL LinkedIn.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    jsonld = _extract_jsonld(soup)
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    slug = parts[1] if len(parts) >= 2 and parts[0] in ("in", "company", "school") else (parts[-1] if parts else "")
    followers = 0
    for item in jsonld:
        for key in ("followerCount", "interactionCount", "memberOf"):
            if key in item:
                followers = max(followers, _safe_int(item.get(key)))
    return {
        "username": slug,
        "display_name": meta.get("og:title", "").split("|")[0].strip() or slug,
        "bio": _clean_text(meta.get("og:description") or meta.get("description"), 500),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": followers,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "linkedin_profile" if "/in/" in parsed.path else "linkedin_org",
    }


def extract_job_posting(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL de l'offre d'emploi.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    page_text = _clean_text(soup.get_text(" ", strip=True), limit=8000)
    jsonld = _extract_jsonld(soup)
    job_payload = next((j for j in jsonld if j.get("@type") == "JobPosting"), {})
    return {
        "is_job_posting": True,
        "title": job_payload.get("title") or meta.get("og:title", ""),
        "description": job_payload.get("description") or meta.get("og:description", ""),
        "hiring_organization": (job_payload.get("hiringOrganization") or {}).get("name"),
        "job_location": str((job_payload.get("jobLocation") or {})),
        "date_posted": job_payload.get("datePosted"),
        "employment_type": job_payload.get("employmentType"),
        "profile_url": url,
        "job_text": page_text,
    }


def extract_generic(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL fournie.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    title = (soup.title.string if soup.title else "") or meta.get("og:title", "")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    page_text = _clean_text(soup.get_text(" ", strip=True), limit=4000)
    return {
        "username": urlparse(url).netloc,
        "display_name": _clean_text(title, 200),
        "bio": _clean_text(meta.get("og:description") or meta.get("description"), 500),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "generic_web",
        "page_text_excerpt": page_text[:600],
    }


EXTRACTORS = {
    "instagram": extract_instagram,
    "linkedin":  extract_linkedin,
    "job":       extract_job_posting,
    "generic":   extract_generic,
}


def build_ml_features(profile: Dict[str, Any]) -> Dict[str, Any]:
    username = str(profile.get("username") or "")
    bio = str(profile.get("bio") or "")
    follower = _safe_int(profile.get("follower_count"))
    following = _safe_int(profile.get("following_count"))
    post = _safe_int(profile.get("post_count"))
    ratio = (follower / (following + 1)) if following >= 0 else 0
    return {
        "follower_count": follower,
        "following_count": following,
        "post_count": post,
        "bio_length": len(bio),
        "username_length": len(username),
        "username_digits": _count_digits(username),
        "has_profile_pic": bool(profile.get("avatar_url")),
        "is_private": bool(profile.get("is_private", False)),
        "follower_following_ratio": round(ratio, 4),
        "has_verified_badge": bool(profile.get("is_verified", False)),
    }


def run_extraction(url: str, forced_platform: Optional[str] = None) -> Dict[str, Any]:
    platform, detection = (forced_platform, "forced") if forced_platform else detect_platform(url)

    # Plateformes non prises en charge (extraction)
    if platform in DISABLED_PLATFORMS:
        return {
            "success": False,
            "platform": platform,
            "detected_via": detection,
            "message": DISABLED_MESSAGE,
        }

    if platform not in EXTRACTORS:
        platform, detection = "generic", "unknown_forced_fallback"

    t0 = time.time()
    try:
        profile = EXTRACTORS[platform](url)
    except Exception as exc:  # noqa: BLE001
        if platform != "generic":
            try:
                profile = extract_generic(url)
                profile["_fallback_from"] = platform
                profile["_fallback_reason"] = str(exc)
                platform = "generic"
                detection += "+fallback"
            except Exception as exc2:  # noqa: BLE001
                return {
                    "success": False,
                    "platform": platform,
                    "detected_via": detection,
                    "message": f"Extraction impossible : {exc2}",
                }
        else:
            return {
                "success": False,
                "platform": platform,
                "detected_via": detection,
                "message": f"Extraction impossible : {exc}",
            }

    elapsed_ms = int((time.time() - t0) * 1000)
    response: Dict[str, Any] = {
        "success": True,
        "platform": platform,
        "detected_via": detection,
        "elapsed_ms": elapsed_ms,
        "profile": profile,
        "source_url": url,
    }
    if profile.get("is_job_posting"):
        response["job_text"] = profile.get("job_text", "")
        response["features"] = None
    else:
        response["features"] = build_ml_features(profile)
    return response


# ----------------------------------------------------------------------------- #
#  Routes                                                                        #
# ----------------------------------------------------------------------------- #

def _generate_analysis_id() -> str:
    return "AN-" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + str(np.random.randint(1000, 9999))


@app.route("/")
def home():
    return jsonify({
        "service": "FakeProfileDetector API",
        "version": "3.0.0",
        "status": "active",
        "models_active": list(MODEL_PERFORMANCE.keys()),
        "platforms_disabled": sorted(DISABLED_PLATFORMS),
        "lstm_loaded": _lstm_interpreter is not None,
        "insta_loaded": _insta_model is not None,
        "linkedin_loaded": _linkd_classifier is not None,
        "endpoints": {
            "GET  /health":          "Healthcheck",
            "GET  /api/models":      "Liste des modèles ML",
            "POST /api/analyze":     "Analyse d'un profil (instagram/linkedin)",
            "POST /api/analyze-job": "Analyse LSTM d'un texte d'offre d'emploi",
            "POST /api/extract-url": "Extraction brute depuis une URL",
            "POST /api/analyze-url": "Extraction + analyse complète depuis une URL",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lstm_loaded": _lstm_interpreter is not None,
        "insta_loaded": _insta_model is not None,
        "linkedin_loaded": _linkd_classifier is not None,
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "success": True,
        "models": MODEL_PERFORMANCE,
        "disabled_platforms": sorted(DISABLED_PLATFORMS),
        "disabled_message": DISABLED_MESSAGE,
    })


def _disabled_response(platform: str) -> Tuple[Any, int]:
    return jsonify({
        "success": False,
        "platform": platform,
        "message": DISABLED_MESSAGE,
        "disabled": True,
    }), 501


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_profile():
    """Analyse d'un profil. Routage par plateforme :
       - instagram -> XGBoost
       - linkedin  -> SVC RoBERTa
       - twitter, github -> 501 (en cours de développement)
    """
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        platform = (data.get("platform") or "").strip().lower()
        features = data.get("features") or {}

        if not platform:
            return jsonify({"success": False, "message": "Plateforme manquante."}), 422

        if platform in DISABLED_PLATFORMS:
            return _disabled_response(platform)

        if not isinstance(features, dict) or not features:
            return jsonify({"success": False, "message": "Caractéristiques manquantes."}), 422

        if platform == "instagram":
            result = predict_instagram(features)
        elif platform == "linkedin":
            result = predict_linkedin(features)
        else:
            return jsonify({
                "success": False,
                "message": f"Plateforme '{platform}' non prise en charge.",
            }), 422

        return jsonify({
            "success": True,
            "analysis_id": _generate_analysis_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "platform": platform,
            **result,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


@app.route("/api/analyze-job", methods=["POST", "OPTIONS"])
def analyze_job():
    """Analyse LSTM d'un texte d'offre d'emploi."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("job_text") or data.get("text") or "").strip()
        if not text:
            return jsonify({"success": False, "message": "Le texte de l'offre d'emploi est obligatoire."}), 422
        if len(text) < 30:
            return jsonify({"success": False, "message": "Le texte saisi est trop court pour une détection fiable."}), 422

        result = predict_job_text(text)
        return jsonify({
            "success": True,
            "analysis_id": _generate_analysis_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **result,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


@app.route("/api/extract-url", methods=["POST", "OPTIONS"])
def extract_url_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    forced = (data.get("platform") or "").strip().lower() or None
    if not url:
        return jsonify({"success": False, "message": "Le champ 'url' est obligatoire."}), 422
    if not re.match(r"^https?://", url):
        url = "https://" + url
    host = urlparse(url).netloc.lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168.") or host.startswith("10."):
        return jsonify({"success": False, "message": "URL interne interdite."}), 422

    if forced and forced in DISABLED_PLATFORMS:
        return _disabled_response(forced)

    result = run_extraction(url, forced_platform=forced)
    if not result.get("success") and result.get("platform") in DISABLED_PLATFORMS:
        return jsonify(result), 501
    return jsonify(result), (200 if result.get("success") else 502)


@app.route("/api/analyze-url", methods=["POST", "OPTIONS"])
def analyze_url_endpoint():
    """Extraction + analyse complète. Pour les offres d'emploi, route vers le LSTM."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        forced = (data.get("platform") or "").strip().lower() or None

        if not url:
            return jsonify({"success": False, "message": "URL manquante."}), 422
        if not re.match(r"^https?://", url):
            url = "https://" + url
        host = urlparse(url).netloc.lower()
        if host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168.") or host.startswith("10."):
            return jsonify({"success": False, "message": "URL interne interdite."}), 422

        if forced and forced in DISABLED_PLATFORMS:
            return _disabled_response(forced)

        extraction = run_extraction(url, forced_platform=forced)
        if not extraction.get("success"):
            if extraction.get("platform") in DISABLED_PLATFORMS:
                return jsonify(extraction), 501
            return jsonify({
                "success": False,
                "message": extraction.get("message", "Extraction impossible."),
                "platform": extraction.get("platform"),
            }), 502

        platform = extraction.get("platform", "generic")
        profile = extraction.get("profile") or {}

        if platform in DISABLED_PLATFORMS:
            return _disabled_response(platform)

        # Cas offre d'emploi -> LSTM
        if platform == "job" or profile.get("is_job_posting"):
            job_text = extraction.get("job_text") or profile.get("job_text") or ""
            if len(job_text) < 30:
                return jsonify({"success": False, "message": "Contenu de l'offre extraite trop court."}), 422
            lstm_result = predict_job_text(job_text)
            features = lstm_result["features"]
            features.update({
                "input_mode": "url_job",
                "source_url": url,
                "job_title": profile.get("title"),
                "hiring_organization": profile.get("hiring_organization"),
            })
            return jsonify({
                "success": True,
                "analysis_id": _generate_analysis_id(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "platform": "job",
                "detected_via": extraction.get("detected_via"),
                "source_url": url,
                "model": "job_text_lstm",
                "risk_score": lstm_result["risk_score"],
                "classification": lstm_result["classification"],
                "confidence": lstm_result["confidence"],
                "metrics": lstm_result["metrics"],
                "shap_values": lstm_result["shap_values"],
                "features": features,
                "extracted_profile": profile,
            }), 200

        # Cas Instagram / LinkedIn
        raw_features = extraction.get("features") or {}
        if platform == "instagram":
            payload_features = {
                "userFollowerCount": _safe_int(raw_features.get("follower_count")),
                "userFollowingCount": _safe_int(raw_features.get("following_count")),
                "userBiographyLength": _safe_int(raw_features.get("bio_length")),
                "userMediaCount": _safe_int(raw_features.get("post_count")),
                "userHasProfilPic": _safe_bool(raw_features.get("has_profile_pic")),
                "userIsPrivate": _safe_bool(raw_features.get("is_private")),
                "usernameDigitCount": _safe_int(raw_features.get("username_digits")),
                "usernameLength": _safe_int(raw_features.get("username_length")),
            }
            ml_result = predict_instagram(payload_features)
        elif platform == "linkedin":
            payload_features = {
                "followers": _safe_int(raw_features.get("follower_count")),
                "connections": _safe_int(raw_features.get("following_count")),
                "num_activities": _safe_int(raw_features.get("post_count")),
            }
            ml_result = predict_linkedin(payload_features)
        else:
            return jsonify({
                "success": False,
                "message": f"Plateforme '{platform}' non prise en charge.",
                "platform": platform,
            }), 422

        merged_features = {
            "input_mode": "url",
            "source_url": url,
            "platform_detected_via": extraction.get("detected_via"),
            "username": profile.get("username"),
            "display_name": profile.get("display_name"),
            "bio_excerpt": (profile.get("bio") or "")[:240],
            "avatar_url": profile.get("avatar_url"),
            **ml_result["features"],
        }

        return jsonify({
            "success": True,
            "analysis_id": _generate_analysis_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "platform": platform,
            "detected_via": extraction.get("detected_via"),
            "source_url": url,
            "model": ml_result["model"],
            "risk_score": ml_result["risk_score"],
            "classification": ml_result["classification"],
            "confidence": ml_result["confidence"],
            "metrics": ml_result["metrics"],
            "shap_values": ml_result["shap_values"],
            "features": merged_features,
            "extracted_profile": profile,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


# ----------------------------------------------------------------------------- #
#  Boot                                                                          #
# ----------------------------------------------------------------------------- #

# Préchargement opportuniste : on tente de charger les modèles au démarrage.
_load_lstm_once()
_load_insta_once()
_load_linkd_once()

if __name__ == "__main__":
    print(f"[FakeProfileDetector] démarré sur http://0.0.0.0:{PORT}", flush=True)
    print(f"  LSTM job(text)        : {_lstm_interpreter is not None}", flush=True)
    print(f"  Instagram XGBoost     : {_insta_model is not None}", flush=True)
    print(f"  LinkedIn RoBERTa-SVC  : {_linkd_classifier is not None}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False)
