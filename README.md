# Skin Lesion Classification

A Streamlit web application for educational/research image classification:

- Benign = 0
- Melanoma = 1

## Files required

Keep these files in the same folder:

- `app.py`
- `requirements.txt`
- `best_transformer_model.keras`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

**Important:** This project is for education/research only and is not a medical diagnostic tool.
