# 🏥 Patient 30-Day Readmission Risk Predictor

An end-to-end machine learning pipeline that predicts whether a hospital patient will be readmitted within 30 days of discharge.

**Tech Stack:** Python • Pandas • NumPy • Scikit-learn • XGBoost • FastAPI • pytest • Jupyter

---

## 📁 Project Structure

```
patient-readmission-predictor/
├── data/
│   ├── generate_data.py        # Synthetic data generator
│   ├── raw/                    # Raw CSVs (git-ignored)
│   └── processed/              # Processed data (git-ignored)
├── notebooks/
│   └── eda_and_results.ipynb   # EDA & model evaluation notebook
├── src/
│   ├── pipeline/
│   │   ├── preprocessing.py    # Cleaning, feature engineering, encoding
│   │   └── model.py            # Training, evaluation, persistence
│   ├── api/
│   │   └── app.py              # FastAPI REST API
│   └── tests/
│       └── test_pipeline.py    # pytest unit tests
├── models/                     # Saved model artefacts (git-ignored)
├── reports/                    # JSON evaluation reports
├── train.py                    # Training entry-point
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data, train, and test in one step
```bash
make all
```

Or step by step:
```bash
make data      # generate synthetic patient data
make train     # train models and save artefacts
make test      # run pytest suite
make api       # start FastAPI server on :8000
```

---

## 🔬 ML Pipeline

### Data & Features
- **Source:** Structured patient records (demographics, diagnoses, medications, prior visits)
- **Target:** `readmitted_30` — binary flag (1 = readmitted within 30 days)
- **Engineered features:** `total_utilisation`, `med_diag_ratio`, `had_prior_inpatient`

### Models Compared
| Model | Cross-Val ROC-AUC |
|---|---|
| Logistic Regression | ~0.70 |
| Random Forest | ~0.78 |
| XGBoost | ~0.80 |

- 5-fold stratified cross-validation
- Class-weighted training to handle imbalance
- Best model selected automatically by CV AUC

---

## 🌐 REST API

Start the server:
```bash
uvicorn src.api.app:app --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch predictions |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 42,
    "age": 70,
    "gender": "M",
    "num_diagnoses": 6,
    "num_procedures": 2,
    "length_of_stay": 8,
    "num_medications": 12,
    "num_lab_procedures": 25,
    "num_outpatient": 0,
    "num_inpatient": 2,
    "num_emergency": 1,
    "a1c_result": ">8",
    "insulin": "Up",
    "diabetesMed": "Yes",
    "discharge_disposition_id": 1,
    "admission_source_id": 7
  }'
```

### Example Response
```json
{
  "patient_id": 42,
  "readmission_probability": 0.7231,
  "risk_level": "HIGH",
  "prediction": 1
}
```

Risk levels: `LOW` (< 0.3) · `MEDIUM` (0.3–0.6) · `HIGH` (≥ 0.6)

---

## 🧪 Tests

```bash
pytest src/tests/ -v
```

Covers:
- `clean_data` — duplicate removal, null handling
- `engineer_features` — derived column correctness
- `encode_categoricals` — fit/transform consistency
- `preprocess_pipeline` — end-to-end no-null guarantee
- API `/health` endpoint
- API `/predict` schema validation (422 on bad input)

---

## 📓 Notebooks

Open `notebooks/eda_and_results.ipynb` for:
- Class distribution and feature correlation heatmap
- ROC curves for all models
- Feature importance (Random Forest / XGBoost)

---

## 🔧 Configuration

All hyperparameters live in `src/pipeline/model.py` → `get_models()`. Swap in your own data by replacing `data/raw/patients.csv` with a real dataset matching the expected schema.
