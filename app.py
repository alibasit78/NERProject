from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from uvicorn import run as app_run

from ner.constants import (
    APP_HOST,
    APP_PORT,
)
from ner.pipline.prediction_pipeline import ModelPredictor
from train import training as start_training

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResponseModel(BaseModel):
    sentence: str
    entity_group: str
    score: float
    word: str
    start: int
    end: int


@app.get("/train")
async def training():
    try:
        start_training()

        return Response("Training successful !!")

    except Exception as e:
        raise Response(f"Error Occurred! {e}")


@app.post("/predict", response_model=ResponseModel)
async def predict_route(text: str):
    try:
        prediction_pipeline = ModelPredictor()
        sentence, labels = prediction_pipeline.initiate_model_predictor(input_sentence=text)
        outputs = labels[0]
        return {
            "sentence": sentence,
            "entity_group": outputs["entity_group"],
            "score": outputs["score"],
            "word": outputs["word"],
            "start": outputs["start"],
            "end": outputs["end"],
        }

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
