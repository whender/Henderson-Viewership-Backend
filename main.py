from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_viewership

app = FastAPI(
    title="Henderson Viewership Model API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # During development, allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Input schema
# -------------------------------
class GameInput(BaseModel):
    team1: str
    team2: str
    rank1: int
    rank2: int
    spread: float
    network: str
    time_slot: str
    comp_tier1: int = 0


# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"status": "running", "message": "Henderson Viewership Model API"}


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict_game(game: GameInput):
    result = predict_viewership(game.dict())
    return {
        "prediction_raw": result["raw"],
        "prediction_formatted": result["formatted"]
    }