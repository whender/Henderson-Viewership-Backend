from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_viewership
from predict import teams_list

app = FastAPI(
    title="Henderson Viewership Model API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameInput(BaseModel):
    team1: str
    team2: str
    rank1: int
    rank2: int
    spread: float
    network: str
    time_slot: str
    comp_tier1: int = 0

@app.get("/")
def root():
    return {"status": "running", "message": "Henderson Viewership Model API"}

@app.get("/teams")
def get_teams():
    return {
        "teams": [
            {"label": frontend_name, "value": backend_name}
            for frontend_name, backend_name in teams_list.items()
        ]
    }

@app.post("/predict")
def predict_game(game: GameInput):
    result = predict_viewership(game.dict())
    return {
        "prediction_raw": result["raw"],
        "prediction_formatted": result["formatted"]
    }


# =====================================================
# 🚀 NEW ENDPOINTS
# =====================================================

@app.get("/brand_rankings")
def brand_rankings():
    # Replace this with real rankings later
    return {
        "rankings": [
            {"team": "Ohio St.", "brand_score": 97},
            {"team": "Michigan", "brand_score": 95},
            {"team": "Alabama", "brand_score": 94},
        ]
    }

@app.get("/weekly_predictions")
def weekly_predictions():
    # Replace with real CSV-driven predictions
    return {
        "predictions": [
            {"matchup": "Michigan vs USC", "predicted_viewers": "5.2M"},
            {"matchup": "Georgia vs Texas", "predicted_viewers": "4.8M"},
        ]
    }

@app.get("/model_explanation")
def model_explanation():
    # Replace with your actual regression explanation
    return {
        "model": "Log-transformed OLS regression with smearing adjustment.",
        "features": [
            "Top 10 Rankings",
            "Network Dummies",
            "Time Slot",
            "Conference Strength",
            "Rivalry Flags",
            "Spread",
        ]
    }