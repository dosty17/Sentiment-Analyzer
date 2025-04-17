from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import nltk
import io
from preprocess import preprocess

nltk.download('wordnet', quiet=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Sentiment Analyzer</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @keyframes fadeUp {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            .fade-up {
                animation: fadeUp 1s ease-out;
            }
        </style>
    </head>
    <body class="min-h-screen bg-gradient-to-br from-blue-700 via-blue-500 to-blue-300 text-white flex items-center justify-center p-6 font-sans">
        <div class="bg-white/10 p-10 rounded-3xl shadow-2xl max-w-xl w-full backdrop-blur-md text-center fade-up">
            <h1 class="text-4xl font-extrabold mb-4 text-white drop-shadow-lg">💬 Sentiment Analyzer API</h1>
            <p class="text-lg text-blue-100 mb-6">Analyze text or CSV content with logistic regression model.</p>
            <div class="text-sm text-blue-200 space-y-1 mb-6">
                <p>🧠 Model: Logistic Regression</p>
                <p>🔗 Endpoints: <code class="bg-white/20 px-2 py-1 rounded">/predict-text</code>, <code class="bg-white/20 px-2 py-1 rounded">/predict-csv</code></p>
            </div>
            <div class="bg-white/20 border border-white/30 rounded-2xl p-6 mt-6 hover:scale-105 hover:shadow-xl transition-all duration-500 ease-in-out backdrop-blur-sm">
                <p class="text-xs uppercase tracking-widest text-white/80 mb-2">Created by</p>
                <h2 class="text-2xl font-bold text-white drop-shadow">Dosty Pshtiwan & Bander Sidiq</h2>
                <p class="text-sm text-blue-100 mt-1">Crafted with 💙 for developers and learners</p>
            </div>
        </div>
    </body>
    </html>
    """

# Load traditional model
traditional_model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class TextRequest(BaseModel):
    text: str

@app.post("/predict-text/")
async def predict_text(data: TextRequest):
    cleaned = preprocess([data.text])
    vect = vectorizer.transform(cleaned)
    pred = traditional_model.predict(vect)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    return {"sentiment": sentiment, "confidence": "-"}

@app.post("/predict-csv/")
async def predict_csv(
    file: UploadFile = File(...),
    column: str = Query("text")
):
    contents = await file.read()

    try:
        text_data = contents.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text_data = contents.decode("cp1252")
        except Exception:
            return JSONResponse(content={"error": "Unable to decode file."}, status_code=400)

    df = pd.read_csv(io.StringIO(text_data), on_bad_lines='skip')

    if column not in df.columns:
        return JSONResponse(content={"error": f"CSV must contain a '{column}' column."}, status_code=400)

    original_text = df[column].fillna("").astype(str).tolist()
    cleaned = preprocess(original_text)
    df["text"] = cleaned
    vect = vectorizer.transform(cleaned)
    preds = traditional_model.predict(vect)
    df["sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    total = len(df)
    summary = {
        "total": total,
        "positive": sentiment_counts.get("Positive", 0),
        "negative": sentiment_counts.get("Negative", 0),
        "positive_percent": round(sentiment_counts.get("Positive", 0) / total * 100, 2),
        "negative_percent": round(sentiment_counts.get("Negative", 0) / total * 100, 2),
    }

    return {
        "data": df[["text", "sentiment"]].to_dict(orient="records"),
        "summary": summary
    }
