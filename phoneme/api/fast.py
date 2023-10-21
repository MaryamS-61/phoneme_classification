from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/predict")
def predict():
    return {"greeting":"Hello World!"}
