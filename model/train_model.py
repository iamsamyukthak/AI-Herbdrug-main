import torch

# Import graph tensors
from prepare_graph import x, edge_index

# Import our HGNN model
from hgnn_model import HerbDrugHGNN

# ============================================
# Create HGNN Model
# ============================================

model = HerbDrugHGNN()

print("\n========== HGNN MODEL ==========\n")
print(model)

# ============================================
# Forward Pass
# ============================================

embeddings = model(x, edge_index)

print("\n========== NODE EMBEDDINGS ==========\n")
print(embeddings)

print("\nEmbedding Shape:")
print(embeddings.shape)

# ============================================
# Save Node Embeddings
# ============================================

torch.save(embeddings, "model/node_embeddings.pt")

print("\nNode embeddings saved successfully!")