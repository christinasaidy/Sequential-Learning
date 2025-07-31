import torch

class RNN_Classifier(torch.nn.Module):
    def __init__(self,embedding_layer, hidden_dim, num_class = 4, embed_dim = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        self.rnn = torch.nn.RNN(embed_dim,hidden_dim,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x,h = self.rnn(x)
        return self.fc(x.mean(dim=1))