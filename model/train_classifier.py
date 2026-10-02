import torch
import torch.nn as nn
import torch.optim as optim

from prepare_graph import x, edge_index
from hgnn_model import HerbDrugHGNN

# Create Model
model = HerbDrugHGNN()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy Target
# (Temporary until we connect interaction labels)
target = torch.randn(48, 8)

loss_function = nn.MSELoss()

epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(x, edge_index)

    loss = loss_function(output, target)

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | Loss = {loss.item():.4f}"
        )

torch.save(model.state_dict(), "model/herbdrug_hgnn.pth")

print("\nTraining Complete!")

print("Model Saved Successfully!")