from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

model = joblib.load('model/logistic_regression_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

app = FastAPI()

class TextInput(BaseModel):
    text: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    user_input = input_data.text
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)
    print(f"Prediction: {prediction}")
    sentiment = "Positive" if prediction == 4 else "Negative"
    return JSONResponse(content={"sentiment": sentiment}, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)