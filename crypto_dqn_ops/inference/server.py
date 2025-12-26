"""FastAPI inference server for crypto trading predictions."""

from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from crypto_dqn_ops.inference.predictor import CryptoPredictor


class PredictionRequest(BaseModel):
    observations: List[List[float]]


class PredictionResponse(BaseModel):
    actions: List[int]
    q_values: List[List[float]]


app = FastAPI(title="Crypto DQN Inference Server", version="0.1.0")

predictor = None


@app.on_event("startup")
async def load_model():
    global predictor
    model_path = Path("trained_models/BTC_777/model_final.pth")
    if not model_path.exists():
        model_path = Path("trained_models/model.onnx")

    predictor = CryptoPredictor(
        model_path=model_path, obs_dim=32, action_dim=2, v_min=0.0, v_max=20.0, atom_size=51
    )
    print(f"Model loaded from {model_path}")


@app.get("/")
async def root():
    return {"message": "Crypto DQN Inference Server", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        observations = np.array(request.observations, dtype=np.float32)

        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)

        actions = predictor.predict_batch(observations)
        q_values = [predictor.get_q_values(obs).tolist()[0] for obs in observations]

        return PredictionResponse(actions=actions.tolist(), q_values=q_values)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_path": str(predictor.model_path),
        "obs_dim": predictor.obs_dim,
        "action_dim": predictor.action_dim,
        "v_min": predictor.v_min,
        "v_max": predictor.v_max,
        "atom_size": predictor.atom_size,
    }
