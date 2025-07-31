from utils.preprocess import preprocess_text, vectorize
import torch

class TextClassificationService:
    def __init__(self, model):
        self.model = model

    def classify(self, text):
        tokens = preprocess_text(text)
        encoded = vectorize(tokens, self.model.max_length, self.model.word_to_idx)
        input_tensor = torch.tensor(encoded).unsqueeze(0)
        with torch.no_grad():
            output = self.model.model(input_tensor)
            pred_idx = int(torch.argmax(output, dim=1).item())
        return self.model.categories[pred_idx]