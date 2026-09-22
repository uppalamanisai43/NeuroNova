# NeuroNova — Brain MRI Classification System

> **Educational / Research Demonstration Only.**
> NeuroNova is not a medical diagnostic device and must not be used to diagnose, treat, or make clinical decisions.

---

## Overview

NeuroNova is a full-stack web application that classifies brain MRI images into four categories using a **weighted ensemble** of three fine-tuned convolutional neural networks:

| Class | Description |
|-------|-------------|
| **Glioma** | Tumor arising from glial cells |
| **Meningioma** | Tumor on the brain membranes |
| **No Tumor** | No detectable tumor |
| **Pituitary** | Tumor in the pituitary gland |

---

## Architecture

```
Browser (Next.js)
      │
      │  POST /predict  (multipart/form-data)
      ▼
Flask Backend
      │
      ├─► Image Validation (extension + Pillow verify)
      │
      ├─► RGB conversion + 224×224 resize
      │
      ├─► MobileNetV2 inference  (20% weight)
      ├─► ResNet50 inference     (60% weight)
      └─► VGG16 inference        (20% weight)
             │
             ▼
    Weighted Soft-Voting Ensemble
    0.20 × P(MobileNet) + 0.60 × P(ResNet50) + 0.20 × P(VGG16)
             │
             ▼
    JSON Response → Browser renders result dashboard
```

---

## Models

Three `.keras` model files trained on the Brain Tumor MRI dataset:

| Model | File | Ensemble Weight |
|-------|------|-----------------|
| MobileNetV2 | `mobilenet_finetuned.keras` | 0.2 (20%) |
| ResNet50 | `resnet50_finetuned.keras` | 0.6 (60%) |
| VGG16 | `vgg16_finetuned.keras` | 0.2 (20%) |

Models are loaded **once at application startup**. No model loading occurs on each request.

---

## Ensemble

Weighted soft-voting over four classes:

```
P_ensemble(class) = 0.20 × P_MobileNetV2(class)
                  + 0.60 × P_ResNet50(class)
                  + 0.20 × P_VGG16(class)
```

The class with the highest ensemble probability is returned as the prediction.

---

## Project Structure

```
NeuroNova/
├── backend/
│   ├── app.py              # Flask application & endpoints
│   ├── model_service.py    # Model loading & inference
│   ├── preprocessing.py    # Per-model image preprocessing
│   ├── ensemble.py         # Weighted soft-voting
│   ├── config.py           # Configuration loader & validator
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       ├── conftest.py
│       ├── test_ensemble.py
│       └── test_api.py
├── config/
│   ├── ensemble_config.json
│   └── class_names.json
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx        # Main application page
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── Navbar.tsx
│   │   │   ├── HeroSection.tsx
│   │   │   ├── UploadZone.tsx
│   │   │   ├── AnalyzeButton.tsx
│   │   │   ├── LoadingState.tsx
│   │   │   ├── PredictionCard.tsx
│   │   │   ├── ProbabilityChart.tsx
│   │   │   ├── ModelPredictionCards.tsx
│   │   │   ├── EnsembleExplanation.tsx
│   │   │   └── Disclaimer.tsx
│   │   └── types/index.ts
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   └── Dockerfile
├── uploads/
├── mobilenet_finetuned.keras   ← existing trained model
├── resnet50_finetuned.keras    ← existing trained model
├── vgg16_finetuned.keras       ← existing trained model
├── ensemble_config.json
├── class_names.json
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 20+
- TensorFlow 2.15+ (install separately if no GPU: `pip install tensorflow-cpu`)

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running

### 1. Start the backend

```bash
cd backend
python app.py
```

The server starts on **http://localhost:5000**.
Model loading (all three `.keras` files) takes 30–90 seconds on first start.

Watch for:
```
All three models loaded successfully.
```

### 2. Start the frontend

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Environment Variables

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:5000` | Flask backend URL |

Set in `frontend/.env.local` (already present).

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins |
| `NEURONOVA_MODELS_DIR` | Project root | Directory containing `.keras` files |
| `NEURONOVA_CONFIG_DIR` | `config/` | Directory containing JSON config files |

---

## API Reference

### `GET /health`

```json
{
  "status": "healthy",
  "models_loaded": true,
  "num_classes": 4,
  "class_names": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

### `GET /model-info`

Returns model metadata (input/output shapes, preprocessing flags, ensemble weights).
Does **not** expose filesystem paths.

### `POST /predict`

**Request:** `multipart/form-data` with field `image` (JPG/JPEG/PNG, max 10 MB)

**Success response:**
```json
{
  "success": true,
  "prediction": {
    "class": "glioma",
    "confidence": 0.9234,
    "confidence_percent": 92.34
  },
  "probabilities": {
    "glioma": 0.9234,
    "meningioma": 0.0312,
    "notumor": 0.0189,
    "pituitary": 0.0265
  },
  "models": {
    "mobilenet": { "prediction": "glioma", "confidence": 0.88, "confidence_percent": 88.0, "probabilities": {} },
    "resnet50":  { "prediction": "glioma", "confidence": 0.95, "confidence_percent": 95.0, "probabilities": {} },
    "vgg16":     { "prediction": "glioma", "confidence": 0.90, "confidence_percent": 90.0, "probabilities": {} }
  },
  "ensemble": {
    "weights": { "mobilenet": 0.2, "resnet50": 0.6, "vgg16": 0.2 }
  }
}
```

**Error response:**
```json
{
  "success": false,
  "error": "Human-readable error message"
}
```

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## Docker

```bash
docker-compose up --build
```

Backend: http://localhost:5000
Frontend: http://localhost:3000

---

## Disclaimer

NeuroNova is an AI-based research and educational demonstration.

- It is **not a medical diagnostic device**.
- It must **not** be used to diagnose, treat, or make clinical decisions.
- Results must **not** replace evaluation by a qualified healthcare professional or radiologist.
- The models are fine-tuned CNNs trained for research purposes.
- No clinical validation has been performed.

---

*NeuroNova — Brain MRI Classification Research Demo*
