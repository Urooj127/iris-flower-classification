# Deployment & Publishing Guide

This project was built and tested in a local sandbox, so it isn't pushed to
GitHub or deployed for you automatically. Follow the steps below to publish it
yourself — it only takes a few minutes.

## 1. Push to GitHub

```bash
cd iris-flower-classification
git init
git add .
git commit -m "Initial commit: Iris flower classification project"

# Create a new empty repo on GitHub first (github.com/new), then:
git branch -M main
git remote add origin https://github.com/<your-username>/iris-flower-classification.git
git push -u origin main
```

## 2. Deploy the live app (Streamlit Community Cloud — free)

1. Make sure `models/best_model.pkl`, `models/scaler.pkl`, and
   `models/metadata.json` are committed to the repo (or add a small
   `setup step` — see note below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repo, branch `main`, and set the main file
   path to `app/streamlit_app.py`.
4. Click **Deploy**. You'll get a live URL like
   `https://<your-app-name>.streamlit.app`.

**Note on the model file:** if you'd rather not commit binary `.pkl` files,
add a `packages.txt`/startup hook that runs
`python src/train_model.py` once on deploy, or simply keep the small model
files in the repo — for this dataset they are only a few KB each, so
committing them is simplest and recommended for a beginner project.

## 3. Alternative: record a short demo video

If you prefer not to deploy a live app, record a 2–3 minute screen capture
showing:
1. Running `python src/train_model.py` in a terminal (shows training + accuracy).
2. Launching `streamlit run app/streamlit_app.py`.
3. Using the sliders to predict a species and viewing the probability chart.
4. Browsing the "Explore Data" and "Model Comparison" tabs.

Upload it to YouTube (unlisted is fine) or Loom, and link it in your README
under a "Demo Video" section.

## 4. Other deployment options
- **Hugging Face Spaces** (supports Streamlit natively, also free).
- **Render / Railway** — deploy as a general Python web service.
- **Docker** — containerize with a simple Dockerfile:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install --no-cache-dir -r requirements.txt
  RUN python src/train_model.py
  EXPOSE 8501
  CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
  ```
