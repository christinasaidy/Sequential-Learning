from fastapi import APIRouter
from app.domain.entities import TextRequest, PredictionResponse
from app.application.services import TextClassificationService
from app.infrastructure.model_classifiers import LSTMTextClassifier, GRUTextClassifier

router = APIRouter()
predictions = {}

lstm_service = TextClassificationService(LSTMTextClassifier())
gru_service = TextClassificationService(GRUTextClassifier())

@router.get("/")
def root():
    return {"Hello": "World"}

@router.post("/predict_lstm", response_model=PredictionResponse)
def predict_lstm(request: TextRequest):
    label = lstm_service.classify(request.text)
    predictions[request.text] = label
    return PredictionResponse(label=label)

@router.post("/predict_gru", response_model=PredictionResponse)
def predict_gru(request: TextRequest):
    label = gru_service.classify(request.text)
    predictions[request.text] = label
    return PredictionResponse(label=label)

@router.get("/predictions")
def get_predictions():
    return predictions