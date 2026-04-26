# 🟢 FakeProfileDetector — Backend Flask (Render)

API REST qui héberge **toute la logique Machine Learning** de FakeProfileDetector.
Conçu pour être déployé sur **Render** (free tier compatible) et appelé par le
frontend PHP hébergé sur InfinityFree.

---

## 📦 Contenu

| Fichier                          | Rôle                                                     |
| -------------------------------- | -------------------------------------------------------- |
| `app.py`                         | API Flask unifiée (analyse + LSTM + extraction d'URL)    |
| `requirements.txt`               | Toutes les dépendances Python                            |
| `Procfile`                       | Commande de démarrage Gunicorn                           |
| `render.yaml`                    | Blueprint Render (déploiement automatique)               |
| `runtime.txt`                    | Version Python ciblée                                    |
| `fake_job_lstm_model.tflite`     | Modèle LSTM (à copier depuis le projet d'origine)        |
| `tokenizer.json`                 | Tokenizer Keras (à copier depuis le projet d'origine)    |

> ⚠️ Avant de déployer, vous DEVEZ ajouter **manuellement** les fichiers
> `fake_job_lstm_model.tflite` et `tokenizer.json` à la racine de ce dossier.
> Ils sont fournis dans le projet original `FPD/backend/job_text_service/`.

---

## 🛣️ Endpoints exposés

| Méthode | Route               | Description                                              |
| ------- | ------------------- | -------------------------------------------------------- |
| GET     | `/`                 | Page d'accueil JSON (liste les routes)                   |
| GET     | `/health`           | Healthcheck (utilisé par Render)                         |
| GET     | `/api/models`       | Liste des modèles ML & métriques                         |
| POST    | `/api/analyze`      | Analyse d'un profil avec features manuelles              |
| POST    | `/api/analyze-job`  | Analyse LSTM d'un texte d'offre d'emploi                 |
| POST    | `/api/extract-url`  | Extraction brute d'une URL (sans calcul ML)              |
| POST    | `/api/analyze-url`  | Extraction + analyse complète d'une URL                  |

### Exemple `POST /api/analyze`

```json
{
  "platform": "instagram",
  "model": "xgboost",
  "features": {
    "follower_count": 1500,
    "following_count": 300,
    "post_count": 45,
    "bio_length": 120,
    "has_profile_pic": true,
    "username_digits": 2
  }
}
```

Réponse :

```json
{
  "success": true,
  "analysis_id": "AN-20260425...",
  "platform": "instagram",
  "model": "xgboost",
  "risk_score": 25,
  "classification": "genuine",
  "confidence": 0.75,
  "metrics": { "name": "XGBoost", "precision": 0.98, ... },
  "shap_values": { ... },
  "features": { ... }
}
```

### Exemple `POST /api/analyze-url`

```json
{ "url": "https://github.com/torvalds", "model": "xgboost" }
```

### Exemple `POST /api/analyze-job`

```json
{ "job_text": "Urgent! Work from home, $500/day, contact via Telegram..." }
```

---

## 🚀 Déploiement sur Render (3 méthodes)

### Méthode 1 — Avec render.yaml (recommandée)

1. Créez un dépôt Git contenant **tous les fichiers de ce dossier**, y compris
   les deux fichiers du modèle LSTM.
2. Connectez-vous sur https://dashboard.render.com → **New +** → **Blueprint**.
3. Pointez vers votre repo. Render lit `render.yaml` et configure tout.
4. Une fois déployé, copiez l'URL publique (ex : `https://fake-profile-detector-api.onrender.com`).
5. Mettez à jour la variable `ALLOWED_ORIGINS` dans le dashboard Render avec
   l'URL exacte de votre site InfinityFree.

### Méthode 2 — Manuelle

1. **New +** → **Web Service** → connectez votre repo.
2. Paramètres :
   - **Environment** : `Python 3`
   - **Build Command** : `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Health Check Path** : `/health`
3. Variables d'environnement :
   - `PYTHON_VERSION` = `3.11.9`
   - `ALLOWED_ORIGINS` = URL de votre frontend (ou `*` pour tous, déconseillé en prod)
   - `LSTM_THRESHOLD` = `0.7` (optionnel)

### Méthode 3 — Test local

```bash
pip install -r requirements.txt
python app.py
# ou en prod :
gunicorn app:app --bind 0.0.0.0:5000
```

---

## ⚙️ Variables d'environnement

| Variable          | Défaut | Description                                                   |
| ----------------- | ------ | ------------------------------------------------------------- |
| `PORT`            | `5000` | Port d'écoute (Render le fixe automatiquement)                |
| `ALLOWED_ORIGINS` | `*`    | Liste CSV des origines CORS autorisées (URL InfinityFree)     |
| `LSTM_THRESHOLD`  | `0.7`  | Seuil de classification LSTM (>seuil = fake)                  |

---

## 🧠 Notes importantes

- **Le free tier Render** met le service en veille après 15 min d'inactivité.
  Le premier appel après veille met ~30-60s à répondre (cold start).
- Le LSTM TensorFlow consomme ~300-500 MB de RAM. Le free tier (512 MB) est
  juste suffisant. Si vous avez des erreurs `Out Of Memory`, passez au plan
  starter (~7$/mois) ou supprimez `tensorflow-cpu` et désactivez `/api/analyze-job`.
- L'API est **totalement stateless** (pas de DB, pas de session). Toute la
  persistance (utilisateurs, historique) est gérée côté PHP/MySQL InfinityFree.

---

## 🔐 Sécurité

- En production, **ne laissez jamais `ALLOWED_ORIGINS=*`**. Mettez l'URL exacte
  de votre frontend.
- Aucune authentification n'est faite ici : c'est le rôle du PHP/InfinityFree
  (qui agit comme proxy authentifié + CSRF + sessions). L'utilisateur ne parle
  jamais directement à Render.
- Pour ajouter une couche supplémentaire, vous pouvez exiger un header secret
  partagé entre PHP et Flask (voir variable `API_SECRET` à ajouter facilement).
