# =============================================================================
#  FakeProfileDetector - Backend Flask UNIFIÉ pour Render
# =============================================================================
#  Ce fichier fusionne les 3 microservices d'origine :
#    - app.py principal (analyse heuristique + SHAP)
#    - job_text_service (LSTM TFLite pour détection de fausses offres d'emploi)
#    - url_extractor_service (extraction de profils sociaux + offres d'emploi)
#
#  Endpoints publics :
#    GET  /                    -> infos API
#    GET  /health              -> healthcheck Render
#    GET  /api/models          -> liste des modèles ML disponibles
#    POST /api/analyze         -> analyse standard (features manuelles)
#    POST /api/analyze-job     -> analyse LSTM d'une offre d'emploi (texte brut)
#    POST /api/extract-url     -> extraction brute d'un profil/offre depuis une URL
#    POST /api/analyze-url     -> extraction + analyse complète depuis une URL
#
#  IMPORTANT :
#    - CORS activé pour permettre l'appel depuis le frontend InfinityFree.
#    - Aucune base de données ici (l'historique est géré côté PHP/MySQL).
#    - Le LSTM (.tflite + tokenizer.json) est chargé une seule fois au boot.
# =============================================================================

import os
import re
import json
import time
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
MODEL_PATH = os.path.join(BASE_DIR, "fake_job_lstm_model.tflite")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
MAX_SEQUENCE_LENGTH = 200
LSTM_THRESHOLD = float(os.environ.get("LSTM_THRESHOLD", "0.7"))

# Liste des origines autorisées à appeler l'API.
# En prod, mettez la variable ALLOWED_ORIGINS sur Render avec votre domaine
# InfinityFree, ex : "https://monsite.infinityfreeapp.com,https://monsite.rf.gd"
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
DEFAULT_TIMEOUT = 10

# ----------------------------------------------------------------------------- #
#  Flask app                                                                     #
# ----------------------------------------------------------------------------- #

app = Flask(__name__)

# CORS : si "*" -> tous, sinon liste blanche
if ALLOWED_ORIGINS == ["*"]:
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
else:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)


# ----------------------------------------------------------------------------- #
#  Modèles ML (métriques + chargement LSTM)                                      #
# ----------------------------------------------------------------------------- #

MODEL_PERFORMANCE: Dict[str, Dict[str, Any]] = {
    "logistic": {
        "name": "Logistic Regression",
        "precision": 0.918, "recall": 0.895, "f1_score": 0.906, "roc_auc": 0.923,
    },
    "random_forest": {
        "name": "Random Forest",
        "precision": 0.962, "recall": 0.949, "f1_score": 0.955, "roc_auc": 0.971,
    },
    "xgboost": {
        "name": "XGBoost",
        "precision": 0.980, "recall": 0.972, "f1_score": 0.976, "roc_auc": 0.985,
    },
    "neural_network": {
        "name": "Neural Network",
        "precision": 0.958, "recall": 0.937, "f1_score": 0.947, "roc_auc": 0.963,
    },
    "job_text_lstm": {
        "name": "job(text) LSTM",
        "precision": None, "recall": None, "f1_score": None, "roc_auc": None,
    },
}

# Lazy-loading du LSTM : on ne le charge que si tensorflow est dispo et le fichier existe.
# Cela permet à l'API de booter même si tensorflow n'est pas installé (mode dégradé).
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


# ----------------------------------------------------------------------------- #
#  Heuristiques communes                                                         #
# ----------------------------------------------------------------------------- #

def compute_risk_score(features: Dict[str, Any]) -> int:
    """Calcule un score de risque (0-100) à partir des features de profil."""
    score = 0
    follower = int(features.get("follower_count", 0) or 0)
    following = max(int(features.get("following_count", 0) or 0), 0)
    posts = int(features.get("post_count", 0) or 0)
    bio_length = int(features.get("bio_length", 0) or 0)
    digits = int(features.get("username_digits", 0) or 0)
    has_pic = bool(features.get("has_profile_pic", True))

    if not has_pic:
        score += 15
    if bio_length < 20:
        score += 10
    if digits > 4:
        score += 15
    if following > 0 and (follower / (following + 1)) < 0.5:
        score += 20
    if posts < 5:
        score += 15
    if following > 1000 and follower < 100:
        score += 25

    return min(score, 100)


def compute_shap_values(features: Dict[str, Any], classification: str) -> Dict[str, float]:
    """Valeurs SHAP simulées pour l'explicabilité (cohérentes avec PHP)."""
    multiplier = 1 if classification == "fake" else -1
    follower = int(features.get("follower_count", 0) or 0)
    following = max(int(features.get("following_count", 0) or 0), 0)
    ratio = follower / (following + 1)
    posts = int(features.get("post_count", 0) or 0)
    bio_length = int(features.get("bio_length", 0) or 0)
    digits = int(features.get("username_digits", 0) or 0)
    has_pic = bool(features.get("has_profile_pic", True))

    return {
        "follower_following_ratio": round((0.18 if ratio < 0.5 else -0.12) * multiplier, 3),
        "post_count": round((0.11 if posts < 5 else -0.08) * multiplier, 3),
        "bio_length": round((0.09 if bio_length < 20 else -0.05) * multiplier, 3),
        "has_profile_pic": round((-0.07 if has_pic else 0.12) * multiplier, 3),
        "username_digits": round((0.14 if digits > 4 else -0.04) * multiplier, 3),
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
    """Tokenise + pad le texte pour le LSTM."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # import différé
    sequence = _lstm_tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32")


def analyze_text_signals(text: str) -> Dict[str, Any]:
    """Extrait des signaux interprétables (mots-clés, ratios) en complément du LSTM."""
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
    """Lance le LSTM sur un texte d'offre d'emploi et retourne le verdict structuré."""
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
#  URL extractor (toutes plateformes)                                            #
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


def _count_digits(text: str) -> int:
    return sum(1 for c in (text or "") if c.isdigit())


def _clean_text(text: Optional[str], limit: int = 600) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 or r.status_code in (401, 403, 404):
            return r
    except requests.RequestException:
        return None
    return None


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


def _epoch_to_iso(epoch: Any) -> Optional[str]:
    try:
        if epoch in (None, "", 0):
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))
    except (TypeError, ValueError):
        return None


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
    if "tiktok.com" in host:
        return ("tiktok", "domain")
    if "youtube.com" in host or "youtu.be" in host:
        return ("youtube", "domain")
    if "reddit.com" in host:
        return ("reddit", "domain")
    if "facebook.com" in host or "fb.com" in host:
        return ("facebook", "domain")
    return ("generic", "fallback")


def extract_github(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise ValueError("URL GitHub invalide.")
    username = parts[0]
    r = _http_get(f"https://api.github.com/users/{username}")
    if r is None or r.status_code != 200:
        raise ValueError(f"Utilisateur GitHub introuvable : {username}")
    data = r.json()
    return {
        "username": data.get("login") or username,
        "display_name": data.get("name") or data.get("login"),
        "bio": data.get("bio") or "",
        "avatar_url": data.get("avatar_url"),
        "profile_url": data.get("html_url") or url,
        "follower_count": _safe_int(data.get("followers")),
        "following_count": _safe_int(data.get("following")),
        "post_count": _safe_int(data.get("public_repos")),
        "public_gists": _safe_int(data.get("public_gists")),
        "is_private": False,
        "is_verified": False,
        "account_type": (data.get("type") or "User").lower(),
        "created_at": data.get("created_at"),
        "location": data.get("location"),
        "company": data.get("company"),
        "blog": data.get("blog"),
    }


def extract_reddit(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("user", "u"):
        username = parts[1]
        r = _http_get(f"https://www.reddit.com/user/{username}/about.json")
        if r is None or r.status_code != 200:
            raise ValueError(f"Profil Reddit introuvable : {username}")
        d = r.json().get("data", {})
        return {
            "username": d.get("name") or username,
            "display_name": d.get("subreddit", {}).get("title") or d.get("name"),
            "bio": d.get("subreddit", {}).get("public_description", ""),
            "avatar_url": (d.get("icon_img") or "").split("?")[0],
            "profile_url": url,
            "follower_count": _safe_int(d.get("subreddit", {}).get("subscribers")),
            "following_count": 0,
            "post_count": _safe_int(d.get("link_karma", 0)) + _safe_int(d.get("comment_karma", 0)),
            "is_private": False,
            "is_verified": bool(d.get("verified", False)),
            "account_type": "reddit_user",
            "created_at": _epoch_to_iso(d.get("created_utc")),
        }
    raise ValueError("URL Reddit non reconnue : seul /user/<name> est pris en charge.")


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


def extract_twitter(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    username = parts[0] if parts else ""
    oembed = _http_get(f"https://publish.twitter.com/oembed?url={url}")
    oembed_data: Dict[str, Any] = {}
    if oembed is not None and oembed.status_code == 200:
        try:
            oembed_data = oembed.json()
        except ValueError:
            oembed_data = {}
    bio, avatar = "", None
    r = _http_get(url)
    if r is not None:
        soup = BeautifulSoup(r.text, "html.parser")
        meta = _extract_og_meta(soup)
        bio = _clean_text(meta.get("og:description") or meta.get("description"), 400)
        avatar = meta.get("og:image")
    return {
        "username": username,
        "display_name": oembed_data.get("author_name") or username,
        "bio": bio,
        "avatar_url": avatar,
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "twitter_user",
        "notice": "X/Twitter ne publie plus les compteurs publics. Valeurs = 0.",
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


def extract_tiktok(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL TikTok.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    username = parts[0].lstrip("@") if parts else ""
    desc = meta.get("og:description") or ""
    followers = following = posts = 0
    m = re.search(
        r"([\d.,KMBkmb ]+)\s*Followers?.*?([\d.,KMBkmb ]+)\s*Following.*?([\d.,KMBkmb ]+)\s*Likes?",
        desc, re.I | re.S,
    )
    if m:
        followers, following, posts = _safe_int(m.group(1)), _safe_int(m.group(2)), _safe_int(m.group(3))
    return {
        "username": username,
        "display_name": meta.get("og:title", "").strip(" |TikTok"),
        "bio": _clean_text(desc, 400),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": followers,
        "following_count": following,
        "post_count": posts,
        "is_private": False,
        "is_verified": False,
        "account_type": "tiktok_user",
    }


def extract_youtube(url: str) -> Dict[str, Any]:
    oembed = _http_get(f"https://www.youtube.com/oembed?url={url}&format=json")
    oembed_data: Dict[str, Any] = {}
    if oembed is not None and oembed.status_code == 200:
        try:
            oembed_data = oembed.json()
        except ValueError:
            oembed_data = {}
    r = _http_get(url)
    meta: Dict[str, str] = {}
    if r is not None:
        soup = BeautifulSoup(r.text, "html.parser")
        meta = _extract_og_meta(soup)
    return {
        "username": oembed_data.get("author_name", ""),
        "display_name": oembed_data.get("author_name", ""),
        "bio": _clean_text(meta.get("og:description") or meta.get("description"), 500),
        "avatar_url": meta.get("og:image") or oembed_data.get("thumbnail_url"),
        "profile_url": oembed_data.get("author_url") or url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "youtube_channel",
    }


def extract_facebook(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL Facebook.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    return {
        "username": parts[0] if parts else "",
        "display_name": meta.get("og:title", "").strip(),
        "bio": _clean_text(meta.get("og:description") or meta.get("description"), 500),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "facebook_page",
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
    "github":    extract_github,
    "reddit":    extract_reddit,
    "instagram": extract_instagram,
    "twitter":   extract_twitter,
    "linkedin":  extract_linkedin,
    "tiktok":    extract_tiktok,
    "youtube":   extract_youtube,
    "facebook":  extract_facebook,
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
        "version": "2.0.0",
        "status": "active",
        "lstm_loaded": _lstm_interpreter is not None,
        "lstm_error": _lstm_load_error,
        "endpoints": {
            "GET  /health":             "Healthcheck",
            "GET  /api/models":         "Liste des modèles ML",
            "POST /api/analyze":        "Analyse de profil avec features manuelles",
            "POST /api/analyze-job":    "Analyse LSTM d'un texte d'offre d'emploi",
            "POST /api/extract-url":    "Extraction brute depuis une URL",
            "POST /api/analyze-url":    "Extraction + analyse complète depuis une URL",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lstm_loaded": _lstm_interpreter is not None,
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"success": True, "models": MODEL_PERFORMANCE})


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_profile():
    """Analyse standard avec features fournies manuellement par le client."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        platform = (data.get("platform") or "").strip()
        model = (data.get("model") or "xgboost").strip()
        features = data.get("features") or {}

        if not platform or not isinstance(features, dict) or not features:
            return jsonify({"success": False, "message": "Plateforme ou caractéristiques manquantes."}), 422
        if model not in MODEL_PERFORMANCE or model == "job_text_lstm":
            return jsonify({"success": False, "message": "Modèle invalide."}), 422

        risk = compute_risk_score(features)
        classification = "fake" if risk > 50 else "genuine"
        confidence = round(risk / 100, 3) if classification == "fake" else round(1 - (risk / 100), 3)
        shap_values = compute_shap_values(features, classification)
        metrics = MODEL_PERFORMANCE[model]

        return jsonify({
            "success": True,
            "analysis_id": _generate_analysis_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "platform": platform,
            "model": model,
            "risk_score": risk,
            "classification": classification,
            "confidence": confidence,
            "metrics": metrics,
            "shap_values": shap_values,
            "features": features,
        }), 200

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
    """Extraction brute (ne fait pas d'analyse ML)."""
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    forced = data.get("platform")
    if not url:
        return jsonify({"success": False, "message": "Le champ 'url' est obligatoire."}), 422
    if not re.match(r"^https?://", url):
        url = "https://" + url
    host = urlparse(url).netloc.lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168.") or host.startswith("10."):
        return jsonify({"success": False, "message": "URL interne interdite."}), 422

    result = run_extraction(url, forced_platform=forced)
    return jsonify(result), (200 if result.get("success") else 502)


@app.route("/api/analyze-url", methods=["POST", "OPTIONS"])
def analyze_url_endpoint():
    """Extraction + analyse complète. Pour les offres d'emploi, route vers le LSTM."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        model = (data.get("model") or "xgboost").strip()
        forced = data.get("platform")

        if not url:
            return jsonify({"success": False, "message": "URL manquante."}), 422
        if not re.match(r"^https?://", url):
            url = "https://" + url
        host = urlparse(url).netloc.lower()
        if host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168.") or host.startswith("10."):
            return jsonify({"success": False, "message": "URL interne interdite."}), 422
        if model not in MODEL_PERFORMANCE or model == "job_text_lstm":
            model = "xgboost"

        extraction = run_extraction(url, forced_platform=forced)
        if not extraction.get("success"):
            return jsonify({
                "success": False,
                "message": extraction.get("message", "Extraction impossible."),
                "platform": extraction.get("platform"),
            }), 502

        platform = extraction.get("platform", "generic")
        profile = extraction.get("profile") or {}

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

        # Cas profils classiques
        raw_features = extraction.get("features") or {}
        features = {
            "input_mode": "url",
            "source_url": url,
            "platform_detected_via": extraction.get("detected_via"),
            "username": profile.get("username"),
            "display_name": profile.get("display_name"),
            "bio_excerpt": (profile.get("bio") or "")[:240],
            "avatar_url": profile.get("avatar_url"),
            "follower_count": _safe_int(raw_features.get("follower_count")),
            "following_count": _safe_int(raw_features.get("following_count")),
            "post_count": _safe_int(raw_features.get("post_count")),
            "bio_length": _safe_int(raw_features.get("bio_length")),
            "username_length": _safe_int(raw_features.get("username_length")),
            "username_digits": _safe_int(raw_features.get("username_digits")),
            "has_profile_pic": bool(raw_features.get("has_profile_pic")),
            "is_private": bool(raw_features.get("is_private")),
            "follower_following_ratio": float(raw_features.get("follower_following_ratio") or 0),
            "has_verified_badge": bool(raw_features.get("has_verified_badge")),
        }
        risk = compute_risk_score(features)
        classification = "fake" if risk > 50 else "genuine"
        confidence = round(risk / 100, 3) if classification == "fake" else round(1 - (risk / 100), 3)
        shap_values = compute_shap_values(features, classification)
        metrics = MODEL_PERFORMANCE[model]

        return jsonify({
            "success": True,
            "analysis_id": _generate_analysis_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "platform": platform,
            "detected_via": extraction.get("detected_via"),
            "source_url": url,
            "model": model,
            "risk_score": risk,
            "classification": classification,
            "confidence": confidence,
            "metrics": metrics,
            "shap_values": shap_values,
            "features": features,
            "extracted_profile": profile,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


# ----------------------------------------------------------------------------- #
#  Boot                                                                          #
# ----------------------------------------------------------------------------- #

# Tente de charger le LSTM au démarrage (mais ne casse pas le boot si absent)
_load_lstm_once()

if __name__ == "__main__":
    print(f"[FakeProfileDetector] démarré sur http://0.0.0.0:{PORT}", flush=True)
    print(f"[FakeProfileDetector] LSTM chargé : {_lstm_interpreter is not None}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False)
