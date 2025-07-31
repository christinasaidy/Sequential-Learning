# from fastapi import FastAPI
# from pydantic import BaseModel
# import torch
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import re
# from utils.preprocess import preprocess_text, vectorize
# from models.lstm import LSTMClassifier
# from models.gru import GRUClassifier
# import pickle

# app = FastAPI()

# class TextRequest_LSTM(BaseModel):
#     text: str

# class TextRequest_GRU(BaseModel):
#     text: str


# class PredictionResponse_LSTM(BaseModel):
#     label: str

# class PredictionResponse_GRU(BaseModel):
#     label: str



# with open("assets/max_length.pkl", "rb") as f:
#     max_length = pickle.load(f)

# with open("assets/word_to_idx.pkl", "rb") as f:
#     word_to_idx = pickle.load(f)


# categories = ["World", "Sports", "Business", "Sci/Tech"]

# with open("assets/vocab_size.pkl", "rb") as f:
#     vocab_size = pickle.load(f)


# print(vocab_size)

# with open("assets/LSTM_best_params.pkl", "rb") as f:
#     best_params_lstm = pickle.load(f)

# with open("assets/GRU_best_params.pkl", "rb") as f:
#     best_params_gru = pickle.load(f)

# embedding_dim_lstm = best_params_lstm['embed_dim']
# hidden_dim_lstm = best_params_lstm['hidden_dim']

# embedding_dim_gru = best_params_gru['embed_dim']
# hidden_dim_gru = best_params_gru['hidden_dim']


# LSTM_model = LSTMClassifier(vocab_size, embedding_dim_lstm, hidden_dim_lstm, 4)
# LSTM_model.load_state_dict(torch.load("assets/LSTM_Classifier.pt"))
# LSTM_model.eval()

# GRU_model = GRUClassifier(vocab_size, embedding_dim_gru, hidden_dim_gru, 4)
# GRU_model.load_state_dict(torch.load("assets/GRU_Classifier.pt"))
# GRU_model.eval()

# predictions = {}

# tokes = preprocess_text("Test 123, this is a test! and ain't an actual run")
# encoded = vectorize(tokes,max_length, word_to_idx)

# @app.get("/")
# def root():
#     return {"Hello": "World"}

# @app.post("/predict_lstm", response_model=PredictionResponse_LSTM)
# def predict_lstm(request: TextRequest_LSTM):
#     text = request.text
#     processed_text = preprocess_text(text)
#     encoded_text = vectorize(processed_text, max_length, word_to_idx)
#     input_tensor = torch.tensor(encoded_text).unsqueeze(0)
#     with torch.no_grad():
#         output = LSTM_model(input_tensor)
#         pred_idx = int(torch.argmax(output, dim=1).item())

#     predictions[text] = categories[pred_idx]
#     return PredictionResponse_LSTM(label=categories[pred_idx])

# @app.post("/predict_gru", response_model=PredictionResponse_GRU)
# def predict_gru(request: TextRequest_GRU):
#     text = request.text
#     processed_text = preprocess_text(text)
#     encoded_text = vectorize(processed_text, max_length, word_to_idx)
#     input_tensor = torch.tensor(encoded_text).unsqueeze(0)
#     with torch.no_grad():
#         output = GRU_model(input_tensor)
#         pred_idx = int(torch.argmax(output, dim=1).item())

#     predictions[text] = categories[pred_idx]
#     return PredictionResponse_GRU(label=categories[pred_idx])
    
# @app.get("/predictions")
# def get_predictions():
#     return predictions

