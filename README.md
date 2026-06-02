# Diabetes
## DiabetesIQ — Full-Stack Prediction Platform

## Project Structure
```
diabetes_app/
├── app.py              ← Flask backend (API + auth + DB)
├── requirements.txt    ← Python dependencies
├── diabetes.db         ← SQLite database (auto-created on first run)
└── templates/
    └── index.html      ← Full SPA frontend
```

## Quick Start

### 1. Install dependencies
```bash
pip install flask
```

### 2. Run the server
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

---

## Demo Accounts (auto-seeded)

| Username  | Password   | Role    | Access                        |
|-----------|-----------|---------|-------------------------------|
| admin     | admin123  | admin   | All predictions history       |
| doctor1   | doctor123 | doctor  | All predictions history       |
| patient1  | patient123| patient | Own predictions only          |

You can also register a new account from the Sign In screen.

---

## Database Schema (SQLite — `diabetes.db`)

### `users` table
| Column        | Type    | Description                |
|---------------|---------|----------------------------|
| id            | INTEGER | Primary key                |
| username      | TEXT    | Unique login name          |
| password_hash | TEXT    | SHA-256 hashed password    |
| full_name     | TEXT    | Display name               |
| email         | TEXT    | Contact email              |
| role          | TEXT    | patient / doctor / admin   |
| created_at    | TEXT    | ISO timestamp              |

### `predictions` table
| Column         | Type  | Description                           |
|----------------|-------|---------------------------------------|
| id             | INT   | Primary key                           |
| user_id        | INT   | FK → users.id                         |
| gender         | TEXT  | male / female                         |
| chol           | REAL  | Total cholesterol value               |
| stab_glu       | REAL  | Stabilised glucose value              |
| hdl            | REAL  | HDL cholesterol                       |
| ratio          | REAL  | Chol/HDL ratio                        |
| age            | REAL  | Patient age                           |
| height         | REAL  | Height in inches                      |
| weight         | REAL  | Weight in lbs                         |
| waist          | REAL  | Waist circumference inches            |
| hip            | REAL  | Hip circumference inches              |
| bp1s           | REAL  | Systolic BP                           |
| bp1d           | REAL  | Diastolic BP                          |
| time_ppn       | REAL  | Post-prandial time minutes            |
| *_label        | TEXT  | Selected option label for each field  |
| risk_score     | REAL  | 0–100 risk percentage                 |
| probability    | REAL  | 0–1 model probability                 |
| prediction     | TEXT  | "Diabetic" or "Non-Diabetic"          |
| flagged_count  | INT   | Number of high-risk attributes        |
| created_at     | TEXT  | ISO timestamp                         |

---

## API Endpoints

| Method | Endpoint         | Auth Required | Description              |
|--------|-----------------|---------------|--------------------------|
| POST   | /api/login       | No            | Returns session cookie   |
| POST   | /api/logout      | No            | Clears session           |
| POST   | /api/register    | No            | Create new account       |
| GET    | /api/me          | No            | Check session status     |
| POST   | /api/predict     | Yes           | Run prediction + save DB |
| GET    | /api/history     | Yes           | Fetch prediction records |
| GET    | /api/stats       | Yes           | Aggregated stats          |

---

## Connecting the Real Trained Model

Replace `compute_probability()` in `app.py` with your actual model:

```python
import pickle
import numpy as np

with open('ann_model.pkl', 'rb') as f:
    saved = pickle.load(f)
    ann    = saved['model']
    scaler = saved['scaler']

def compute_probability(v, gender):
    feature_order = ['chol','stab_glu','hdl','ratio','age',
                     'height','weight','waist','hip','bp1s','bp1d','time_ppn']
    X = np.array([[v[k] for k in feature_order]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    return float(ann.predict_proba(X_scaled)[0])
```

Save the model from notebook with:
```python
import pickle
with open('ann_model.pkl', 'wb') as f:
    pickle.dump({'model': ann, 'scaler': scaler}, f)
```

---

## Features
- ✅ Session-based authentication (login / register / logout)
- ✅ Role-based access (admin & doctor see all records; patients see own)
- ✅ SQLite database capturing all predictions with full audit trail
- ✅ Click-to-select attribute ranges (no sliders) with colour-coded risk levels
- ✅ Gender field (♀ Female / ♂ Male) from original dataset
- ✅ Progress bar — predict button locked until all 13 fields complete
- ✅ Deep ANN scoring (mirrors notebook StandardScaler + logistic layers)
- ✅ Dashboard with live stats (total, diabetic count, avg risk)
- ✅ Prediction history table (with role-filtered views)
- ✅ Single-file deployment — no npm, no build step required
