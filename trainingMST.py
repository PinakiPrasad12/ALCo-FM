import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torch_geometric.nn as pyg_nn
from contiformer import ContiFormer  

class AccidentPredictionModel(nn.Module):
    def __init__(self, num_input_features, image_embed_dim, gnn_embed_dim, hidden_dim, num_classes):
        super(AccidentPredictionModel, self).__init__()
        
        # Transformer Encoder for Numerical Data (ContiFormer)
        self.num_encoder = ContiFormer(input_dim=num_input_features, embed_dim=hidden_dim)
        self.num_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Vision Transformer (T2T-ViT) for Image Features
        self.image_encoder = models.vit_b_16(pretrained=True)
        self.image_fc = nn.Linear(1000, image_embed_dim)
        
        # Graph Neural Network (MST-GAT) for Spatio-Temporal Graphs
        self.gnn1 = pyg_nn.GATConv(gnn_embed_dim, hidden_dim, heads=4, concat=True)
        self.gnn2 = pyg_nn.GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=False)
        
        # Gated Fusion Layer
        self.alpha = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, num_input, image_input, edge_index, node_features):
        # Numerical Data Encoding (ContiFormer)
        num_embed = self.num_encoder(num_input)
        num_embed = self.num_fc(num_embed)
        
        # Image Data Encoding
        image_embed = self.image_encoder(image_input)
        image_embed = self.image_fc(image_embed)
        
        # GNN-based Spatio-Temporal Learning
        gnn_embed = self.gnn1(node_features, edge_index)
        gnn_embed = self.gnn2(gnn_embed, edge_index)
        
        # Fusion with Gated Attention
        concat_features = torch.cat((num_embed, image_embed, gnn_embed), dim=1)
        fusion_weights = self.gate(concat_features)
        z = fusion_weights[:, 0] * num_embed + fusion_weights[:, 1] * image_embed + fusion_weights[:, 2] * gnn_embed
        
        # Prediction
        out = self.classifier(z)
        return out

# Hyperparameters
num_input_features = 21
image_embed_dim = 256
gnn_embed_dim = 128
hidden_dim = 512
num_classes = 5  

# Initialize Model
model = AccidentPredictionModel(num_input_features, image_embed_dim, gnn_embed_dim, hidden_dim, num_classes)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Training Loop
for epoch in range(40):
    for batch in train_loader:
        num_input, image_input, edge_index, node_features, labels = batch
        num_input, image_input, edge_index, node_features, labels = (
            num_input.to(device), image_input.to(device), edge_index.to(device), node_features.to(device), labels.to(device)
        )
        
        optimizer.zero_grad()
        outputs = model(num_input, image_input, edge_index, node_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
