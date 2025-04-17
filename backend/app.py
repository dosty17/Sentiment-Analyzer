from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import nltk
import io
from transformers import pipeline
from preprocess import preprocess

from fastapi.responses import HTMLResponse

nltk.download('wordnet', quiet=True)

app = FastAPI()


from fastapi.responses import HTMLResponse

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
            <p class="text-lg text-blue-100 mb-6">Analyze text or CSV content with AI-powered sentiment models.</p>
            <div class="text-sm text-blue-200 space-y-1 mb-6">
                <p>🧠 Models: Logistic Regression | RoBERTa</p>
                <p>🔗 Endpoints: <code class="bg-white/20 px-2 py-1 rounded">/predict-text</code>, <code class="bg-white/20 px-2 py-1 rounded">/predict-csv</code></p>
            </div>
            <!-- Creator Card -->
            <div class="bg-white/20 border border-white/30 rounded-2xl p-6 mt-6 hover:scale-105 hover:shadow-xl transition-all duration-500 ease-in-out backdrop-blur-sm">
                <p class="text-xs uppercase tracking-widest text-white/80 mb-2">Created by</p>
                <h2 class="text-2xl font-bold text-white drop-shadow">Dosty Pshtiwan & Bander Sidiq</h2>
                <p class="text-sm text-blue-100 mt-1">Crafted with 💙 for developers and learners</p>
            </div>
        </div>
    </body>
    </html>
    """




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
traditional_model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
transformer_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

label_map = {
    "LABEL_0": "🔴 Negative",
    "LABEL_1": "🟡 Neutral",
    "LABEL_2": "🟢 Positive"
}

class TextRequest(BaseModel):
    text: str
    model: str  # 'traditional' or 'transformer'

@app.post("/predict-text/")
async def predict_text(data: TextRequest):
    if data.model == "transformer":
        result = transformer_classifier(data.text)[0]
        sentiment = label_map[result['label']]
        confidence = round(result['score'] * 100, 2)
        return {"sentiment": sentiment, "confidence": f"{confidence}%"}
    else:
        cleaned = preprocess([data.text])
        vect = vectorizer.transform(cleaned)
        pred = traditional_model.predict(vect)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        return {"sentiment": sentiment, "confidence": "-"}


@app.post("/predict-csv/")
async def predict_csv(
    file: UploadFile = File(...),
    model: str = Query("traditional", enum=["traditional", "transformer"]),
    column: str = Query("text")  # 👈 user-selected column from dropdown
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

    if model == "transformer":
        results = transformer_classifier(original_text)
        df["sentiment"] = [label_map[r["label"]] for r in results]
        df["text"] = original_text  # store as-is
    else:
        cleaned = preprocess(original_text)
        df["text"] = cleaned  # update or add cleaned version
        vect = vectorizer.transform(cleaned)
        preds = traditional_model.predict(vect)
        df["sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

    df = df.fillna("")  # prevent NaN JSON error

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    total = len(df)
    summary = {
        "total": total,
        "positive": sentiment_counts.get("Positive", 0) + sentiment_counts.get("🟢 Positive", 0),
        "negative": sentiment_counts.get("Negative", 0) + sentiment_counts.get("🔴 Negative", 0),
        "positive_percent": round((sentiment_counts.get("Positive", 0) + sentiment_counts.get("🟢 Positive", 0)) / total * 100, 2),
        "negative_percent": round((sentiment_counts.get("Negative", 0) + sentiment_counts.get("🔴 Negative", 0)) / total * 100, 2),
    }

    return {
        "data": df[["text", "sentiment"]].to_dict(orient="records"),
        "summary": summary
    }







# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import pandas as pd
# import joblib
# import nltk
# import io
# from preprocess import preprocess

# nltk.download('wordnet', quiet=True)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load ML model and vectorizer
# model = joblib.load("logistic_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

# class TextRequest(BaseModel):
#     text: str

# @app.post("/predict-text/")
# async def predict_text(data: TextRequest):
#     cleaned = preprocess([data.text])
#     vect = vectorizer.transform(cleaned)
#     pred = model.predict(vect)[0]
#     sentiment = "Positive" if pred == 1 else "Negative"
#     return {"sentiment": sentiment}

# @app.post("/predict-csv/")
# async def predict_csv(file: UploadFile = File(...)):
#     contents = await file.read()

#     try:
#         # Try UTF-8 first
#         text_data = contents.decode("utf-8")
#     except UnicodeDecodeError:
#         try:
#             # Fallback to common Windows encoding
#             text_data = contents.decode("cp1252")
#         except Exception:
#             return JSONResponse(content={"error": "Unable to decode file. Try saving as UTF-8 or CSV UTF-8 format."}, status_code=400)

#     df = pd.read_csv(io.StringIO(text_data))

#     if "text" not in df.columns:
#         return JSONResponse(content={"error": "CSV must contain a 'text' column."}, status_code=400)

#     cleaned = preprocess(df["text"].tolist())
#     vect = vectorizer.transform(cleaned)
#     preds = model.predict(vect)
#     df["sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

#     summary = {
#         "positive": int(sum(preds == 1)),
#         "negative": int(sum(preds == 0)),
#         "total": len(preds),
#         "positive_percent": round(sum(preds == 1) / len(preds) * 100, 2),
#         "negative_percent": round(sum(preds == 0) / len(preds) * 100, 2)
#     }

#     return {"data": df.to_dict(orient="records"), "summary": summary}
