# Chest X-Ray Pneumonia Detection (CNN vs ResNet18 Transfer Learning)

Capstone project of a 2-month remote Machine Learning & Deep Learning internship at GetAlphaBit. The system classifies pediatric chest X-rays as **NORMAL** or **PNEUMONIA** and serves the best model through a small web API.

## What this project does

- Uses the public Kaggle dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Analyzes the data and finds a class imbalance of about 3:1 (PNEUMONIA : NORMAL)
- Handles the imbalance with a `WeightedRandomSampler` and a class-weighted loss
- Trains two models in PyTorch under the same preprocessing:
  - a baseline CNN built from scratch (4 convolutional blocks)
  - ResNet18 with transfer learning (ImageNet weights, last block and classifier fine-tuned)
- Compares them with precision, recall, F1, ROC-AUC, confusion matrices and ROC curves, not accuracy alone
- Serves the best model with **FastAPI** (`POST /predict`)

## Results (test set, 624 images)

| Metric | Baseline CNN | ResNet18 |
|--------|--------------|----------|
| Accuracy | 0.8317 | **0.9071** |
| Precision | 0.8740 | 0.8915 |
| Recall | 0.8538 | **0.9692** |
| F1 | 0.8638 | **0.9287** |
| ROC-AUC | 0.8930 | **0.9573** |

ResNet18 missed 12 of 390 real pneumonia cases, against 57 for the baseline. It was selected as the final model. Its main weakness is false alarms on normal X-rays, which is a safer error than a missed case in a screening setting. Full details are in `Capstone_Report.pdf`.

## The API

`app.py` loads the saved model once at startup and exposes:

- `GET /` : health check
- `POST /predict` : upload a JPEG or PNG X-ray, get back `label`, `confidence` and `model_type`
  - rejects wrong file types, empty files and unreadable images with a 400 error
  - returns `UNCERTAIN` when confidence is below 0.60

## Tech stack

Python, PyTorch, torchvision, scikit-learn, FastAPI, pandas, NumPy, matplotlib, seaborn, Google Colab

## Files

| File | Description |
|------|-------------|
| `chest_xray_pneumonia_detection.py` | Full pipeline exported from Google Colab (EDA, training, evaluation, API) |
| `app.py` | FastAPI service |
| `Capstone_Report.pdf` | Project report |

## Run it

The notebook was written for Google Colab with a GPU. It downloads the dataset from Kaggle, so you need your **own** Kaggle API token (`kaggle.json`). Never commit that file; it is listed in `.gitignore`.

```bash
pip install torch torchvision scikit-learn seaborn fastapi uvicorn python-multipart pillow
# after training, which saves model_artifacts/best_model.pth:
uvicorn app:app --reload
```

## Disclaimer

Educational project only. It is not a medical device and must not be used for real diagnosis.
