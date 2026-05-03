# FakeProfileDetector — Backend Flask (Render)

Backend Python/Flask unifié pour la détection de faux profils et de fausses
offres d'emploi. Conçu pour un déploiement direct sur **Render** (free plan
compatible) et appelé depuis le frontend PHP hébergé sur **InfinityFree**.

---

## Modèles ML actifs

| Clé interne       | Plateforme | Modèle                                     | Fichier                          |
|-------------------|------------|--------------------------------------------|----------------------------------|
| `job_text_lstm`   | `job`      | LSTM TFLite (texte d'offre)                | `fake_job_lstm_model.tflite` + `tokenizer.json` |
| `insta_xgboost`   | `instagram`| XGBClassifier (8 features)                 | `Insta_xgboost.joblib`           |
| `linkedin_svc`    | `linkedin` | SVC RoBERTa (167 features = 150 PCA + 17 num.) | `Linkd_roberta_SVC_model.pkl`    |

### Plateformes désactivées

`twitter` et `github` renvoient HTTP **501** + le message :

> **« Fonctionnalité en cours de développement. Disponible prochainement. »**

---

## Endpoints publics

| Méthode | Chemin               | Description                                                                 |
|---------|----------------------|-----------------------------------------------------------------------------|
| GET     | `/`                  | Infos API + état des modèles                                                |
| GET     | `/health`            | Healthcheck (utilisé par Render)                                            |
| GET     | `/api/models`        | Liste des modèles ML actifs et plateformes désactivées                      |
| POST    | `/api/analyze`       | Analyse d'un profil (instagram/linkedin) à partir de features manuelles     |
| POST    | `/api/analyze-job`   | Analyse LSTM d'un texte d'offre d'emploi                                    |
| POST    | `/api/extract-url`   | Extraction brute des données d'une URL (sans ML)                            |
| POST    | `/api/analyze-url`   | Extraction + analyse complète d'une URL                                     |

### Exemple — Instagram

```http
POST /api/analyze
Content-Type: application/json

{
  "platform": "instagram",
  "features": {
    "userFollowerCount": 1500,
    "userFollowingCount": 300,
    "userBiographyLength": 120,
    "userMediaCount": 45,
    "userHasProfilPic": true,
    "userIsPrivate": false,
    "usernameDigitCount": 3,
    "usernameLength": 12
  }
}
```

### Exemple — LinkedIn

```http
POST /api/analyze
Content-Type: application/json

{
  "platform": "linkedin",
  "features": {
    "followers": 220,
    "connections": 180,
    "num_skills": 12,
    "num_experiences": 3,
    "num_educations": 2,
    "num_recommendations": 5,
    "num_activities": 30
  }
}
```

> ⚠️ Les 150 dimensions PCA RoBERTa du modèle LinkedIn ne peuvent pas être
> recalculées côté serveur sans le pipeline d'embedding d'origine. Elles sont
> neutralisées (0.0) après scaling. La prédiction s'appuie principalement sur
> les **17 features numériques structurées**.

### Exemple — Plateforme désactivée

```http
POST /api/analyze
{ "platform": "twitter", "features": {...} }
```

Réponse `HTTP 501` :

```json
{
  "success": false,
  "platform": "twitter",
  "message": "Fonctionnalité en cours de développement. Disponible prochainement.",
  "disabled": true
}
```

---

## Déploiement sur Render

1. Créez un nouveau service **Web Service** Python sur Render.
2. Connectez ce dépôt (ou uploadez le ZIP).
3. Render détectera `render.yaml` automatiquement.
4. Définissez la variable d'environnement `ALLOWED_ORIGINS` avec votre URL
   InfinityFree (ex. `https://fakeprofiledetector.wuaze.com`).
5. Premier déploiement : le `buildCommand` installera TF-CPU + scikit-learn
   + xgboost (~3-5 min sur free).

### Variables d'environnement

| Variable          | Défaut | Rôle                                                |
|-------------------|--------|-----------------------------------------------------|
| `PORT`            | 5000   | Imposé par Render                                   |
| `ALLOWED_ORIGINS` | `*`    | Liste blanche CORS, séparée par virgules            |
| `LSTM_THRESHOLD`  | `0.7`  | Seuil de classification du LSTM job posting         |

---

## Test local

```bash
pip install -r requirements.txt
python app.py
# -> http://0.0.0.0:5000
```

```bash
curl http://localhost:5000/health
curl http://localhost:5000/api/models
```
