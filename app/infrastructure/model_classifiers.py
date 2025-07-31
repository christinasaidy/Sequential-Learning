import torch
import pickle
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier  

class GRUTextClassifier():
    def __init__(self):
        self.categories = ["World", "Sports", "Business", "Sci/Tech"]

        with open("assets/max_length.pkl", "rb") as f:
            self.max_length = pickle.load(f)

        with open("assets/word_to_idx.pkl", "rb") as f:
            self.word_to_idx = pickle.load(f)

        with open("assets/vocab_size.pkl", "rb") as f:
            vocab_size = pickle.load(f)

        with open("assets/GRU_best_params.pkl", "rb") as f:
            params = pickle.load(f)

        self.model = GRUClassifier(vocab_size, params['embed_dim'], params['hidden_dim'], 4)
        self.model.load_state_dict(torch.load("assets/GRU_Classifier.pt"))
        self.model.eval()

class LSTMTextClassifier():
    def __init__(self):
        self.categories = ["World", "Sports", "Business", "Sci/Tech"]
        with open("assets/max_length.pkl", "rb") as f:
            self.max_length = pickle.load(f)
        with open("assets/word_to_idx.pkl", "rb") as f:
            self.word_to_idx = pickle.load(f)
        with open("assets/vocab_size.pkl", "rb") as f:
            vocab_size = pickle.load(f)
        with open("assets/LSTM_best_params.pkl", "rb") as f:
            params = pickle.load(f)

        self.model = LSTMClassifier(vocab_size, params['embed_dim'], params['hidden_dim'], 4)
        self.model.load_state_dict(torch.load("assets/LSTM_Classifier.pt"))
        self.model.eval()
