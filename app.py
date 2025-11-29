from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import requests
import os
import csv
def check_phishtank(url):
    # Real PhishTank API integration (public API, no key required for basic checks)
    # See https://phishtank.org/api_info.php
    try:
        resp = requests.post(
            "https://checkurl.phishtank.com/checkurl/",
            data={"url": url, "format": "json"},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            # If in_database and valid, then it's a known phishing site
            if data.get("results", {}).get("in_database") and data["results"].get("valid"):
                return True
        return False
    except Exception:
        return None  # Could not check
try:
    app
except NameError:
    from fastapi import FastAPI
    app = FastAPI()

@app.get("/dashboard_stats")
def dashboard_stats():
    # Return real stats for dashboard: phishing detections (last 7 days), recent feedback
    # 1. Phishing detections: count from feedback.csv (if exists)
    # 2. Recent feedback: last 3 rows from feedback.csv
    feedback_path = os.path.join("outputs", "feedback.csv")
    detections = [0]*7
    today = None
    feedback_rows = []
    if os.path.exists(feedback_path):
        with open(feedback_path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            feedback_rows = reader[-3:] if len(reader) >= 3 else reader
            # Count phishing detections per day (last 7 days)
            from datetime import datetime, timedelta
            today = datetime.now().date()
            for row in reader:
                try:
                    ts = row.get("timestamp")
                    if ts:
                        dt = datetime.strptime(ts[:10], "%Y-%m-%d").date()
                        days_ago = (today - dt).days
                        if 0 <= days_ago < 7 and row.get("prediction") == "1":
                            detections[6-days_ago] += 1
                except Exception:
                    pass
    # Format feedback for frontend
    feedback_list = []
    for row in feedback_rows:
        url = row.get("url", "?")
        pred = row.get("prediction")
        user = row.get("user_label")
        if pred == user:
            verdict = "True Positive" if pred == "1" else "True Negative"
        elif pred == "1" and user == "0":
            verdict = "False Positive (user reported legitimate)"
        elif pred == "0" and user == "1":
            verdict = "False Negative (user reported phishing)"
        else:
            verdict = "Feedback"
        feedback_list.append(f"{url}: {verdict}")
    # Dates for last 7 days
    from datetime import datetime, timedelta
    days = [(today - timedelta(days=6-i)).strftime("%Y-%m-%d") for i in range(7)] if today else []
    return JSONResponse({
        "days": days,
        "detections": detections,
        "recent_feedback": feedback_list
    })
# User Feedback Endpoint
import csv
from datetime import datetime

@app.post("/feedback")
async def feedback(url: str = Form(...), prediction: int = Form(...), user_label: int = Form(...)):
    """
    Accepts user feedback for a URL prediction.
    prediction: 0=Legitimate, 1=Phishing (what the model predicted)
    user_label: 0=Legitimate, 1=Phishing (what the user says)
    """
    feedback_file = "user_feedback.csv"
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "url": url,
        "prediction": prediction,
        "user_label": user_label
    }
    try:
        file_exists = os.path.isfile(feedback_file)
        with open(feedback_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# FastAPI version of the backend
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd

app = FastAPI()

# Online prediction endpoint for single URL
@app.post("/predict_url")
async def predict_url(url: str = Form(...)):
    try:
        # Minimal DataFrame for pipeline
        df = pd.DataFrame({"url": [url]})
        results = run_ml_pipeline_on_df(df)
        # Use best model's prediction (simulate for now)
        # In a real system, you'd load the trained model and call model.predict([features])
        # Here, we just return the processed features and a dummy prediction
        # For demonstration, return the feature-engineered row
        preprocessor = EnhancedPhishingPreprocessor()
        df_clean = preprocessor.clean_and_convert_data(df)
        df_enhanced = preprocessor.feature_engineering(df_clean)
        features = df_enhanced.iloc[0].to_dict()

        # Threat intelligence check (PhishTank demo)
        phishtank_result = check_phishtank(url)
        threat_intel = {"phishtank": phishtank_result}

        # Simulate prediction (replace with real model.predict in production)
        prediction = 1 if "login" in url or "secure" in url else 0

        # Explainability: generate reasons for the prediction
        reasons = []
        if "login" in url:
            reasons.append("Contains suspicious keyword: 'login'")
        if "secure" in url:
            reasons.append("Contains suspicious keyword: 'secure'")
        if features.get('ip_usage', 0):
            reasons.append("Uses IP address instead of domain")
        if features.get('shortener', 0):
            reasons.append("Uses URL shortener service")
        if features.get('https_flag', 0) == 0:
            reasons.append("Does not use HTTPS")
        if features.get('cert_age', 365) < 30:
            reasons.append("SSL certificate is very new")
        if not reasons and prediction == 1:
            reasons.append("Other suspicious patterns detected")
        if not reasons and prediction == 0:
            reasons.append("No suspicious patterns detected")

        # (Optional) SHAP or LIME placeholder
        shap_values = {
            k: round(0.2, 2) if k in ['url_length', 'dot_count', 'suspicious_words'] else 0.0
            for k in features.keys()
        }

        return {
            "url": url,
            "features": features,
            "prediction": prediction,
            "prediction_label": "Phishing" if prediction else "Legitimate",
            "reasons": reasons,
            "shap_values": shap_values,
            "threat_intel": threat_intel
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# --- ML Pipeline Backend Function ---
from main import EnhancedPhishingPreprocessor, ComprehensiveModelTrainer

def run_ml_pipeline_on_df(df):
    """
    Run the full ML pipeline on a given DataFrame and return results for API.
    Returns:
        dict: { 'models': [...], 'best_model': {...}, 'data_stats': {...} }
    """
    preprocessor = EnhancedPhishingPreprocessor()
    trainer = ComprehensiveModelTrainer()
    # Clean and feature engineer
    df_clean = preprocessor.clean_and_convert_data(df)
    df_enhanced = preprocessor.feature_engineering(df_clean)
    X, y = trainer.prepare_data(df_enhanced)
    X_test, y_test = trainer.train_and_evaluate_models(X, y)
    # Compose results for API
    models = []
    for name, result in trainer.results.items():
        if result.get('status') == 'SUCCESS':
            models.append({
                'name': name,
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'auc': result['roc_auc']
            })
    # Sort by accuracy
    models.sort(key=lambda m: m['accuracy'], reverse=True)
    best_model = models[0] if models else None
    data_stats = {
        'totalRows': len(df),
        'features': len(df.columns),
        'phishingCount': int(y.sum()),
        'legitimateCount': int((1-y).sum()),
        'featureNames': list(df.columns)
    }
    return {
        'models': models,
        'bestModel': best_model,
        'dataStats': data_stats
    }


# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main HTML at root
@app.get("/")
async def serve_root():
    html_path = os.path.join(os.path.dirname(__file__), "project.html")
    return FileResponse(html_path, media_type="text/html")

# Endpoint for file upload (optional, for storage)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        contents = await file.read()
        # Save file to disk (optional, for demonstration)
        with open(f"uploads/{file.filename}", "wb") as f:
            f.write(contents)
        return {"filename": file.filename, "message": "File uploaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Endpoint for prediction (ML pipeline)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        contents = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        results = run_ml_pipeline_on_df(df)
        if not results['models']:
            return JSONResponse(status_code=500, content={"error": "No successful model"})
        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})