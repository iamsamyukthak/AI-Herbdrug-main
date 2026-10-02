import torch
import torch.nn as nn


class InteractionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(16, 32)

        self.fc2 = nn.Linear(32, 16)

        self.fc3 = nn.Linear(16, 3)

        self.relu = nn.ReLU()

    def forward(self, herb_embedding, drug_embedding):

        # Join herb and drug embeddings
        x = torch.cat(
            (herb_embedding, drug_embedding),
            dim=1
        )

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        return x