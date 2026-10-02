import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class HerbDrugInteractionModel(nn.Module):

    def __init__(
        self,
        num_herbs,
        num_drugs,
        embedding_dim=32
    ):

        super().__init__()

        # ==================================================
        # LEARNABLE HERB / DRUG EMBEDDINGS
        # ==================================================

        self.herb_embedding = nn.Embedding(
            num_herbs,
            embedding_dim
        )

        self.drug_embedding = nn.Embedding(
            num_drugs,
            embedding_dim
        )

        # ==================================================
        # GNN
        # ==================================================

        self.conv1 = GCNConv(
            2,
            32
        )

        self.conv2 = GCNConv(
            32,
            embedding_dim
        )

        # ==================================================
        # INTERACTION CLASSIFIER
        # ==================================================

        # herb = 32
        # drug = 32
        # product = 32
        # absolute difference = 32
        #
        # total = 128
        #
        self.fc1 = nn.Linear(
            embedding_dim * 4,
            64
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.fc3 = nn.Linear(
            32,
            3
        )

        self.dropout = nn.Dropout(
            0.25
        )

    # ======================================================
    # GRAPH EMBEDDINGS
    # ======================================================

    def get_node_embeddings(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        x,
        edge_index,
        herb_ids,
        drug_ids
    ):

        # --------------------------------------------------
        # GNN embeddings
        # --------------------------------------------------

        graph_embeddings = self.get_node_embeddings(
            x,
            edge_index
        )

        graph_herb = graph_embeddings[
            herb_ids
        ]

        graph_drug = graph_embeddings[
            drug_ids
        ]

        # --------------------------------------------------
        # Learnable ID embeddings
        # --------------------------------------------------

        learned_herb = self.herb_embedding(
            herb_ids
        )

        learned_drug = self.drug_embedding(
            drug_ids
        )

        # --------------------------------------------------
        # Combine GNN + learnable embeddings
        # --------------------------------------------------

        herb = (
            graph_herb +
            learned_herb
        )

        drug = (
            graph_drug +
            learned_drug
        )

        # --------------------------------------------------
        # Interaction features
        # --------------------------------------------------

        product = (
            herb * drug
        )

        difference = torch.abs(
            herb - drug
        )

        combined = torch.cat(
            [
                herb,
                drug,
                product,
                difference
            ],
            dim=1
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        combined = self.fc1(
            combined
        )

        combined = F.relu(
            combined
        )

        combined = self.dropout(
            combined
        )

        combined = self.fc2(
            combined
        )

        combined = F.relu(
            combined
        )

        output = self.fc3(
            combined
        )

        return output