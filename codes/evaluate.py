import json
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import models
import torch_geometric.nn as pyg_nn
from contiformer import ContiFormer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class AccidentDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="test"):
        self.mode = mode
        self.data = [self._create_dummy_sample() for _ in range(20)]
    def _create_dummy_sample(self):
        num_input = torch.randn(21)
        image_input = torch.randn(3, 224, 224)
        edge_index = torch.randint(0, 10, (2, 20))
        node_features = torch.randn(10, 128)
        label = torch.randint(0, 5, (1,)).item()
        return (num_input, image_input, edge_index, node_features, label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class AccidentPredictionModel(nn.Module):
    def __init__(self, num_input_features, image_embed_dim, gnn_embed_dim, hidden_dim, num_classes):
        super(AccidentPredictionModel, self).__init__()
        self.num_encoder = ContiFormer(input_dim=num_input_features, embed_dim=hidden_dim)
        self.num_fc = nn.Linear(hidden_dim, hidden_dim)
        self.image_encoder = models.vit_b_16(pretrained=True)
        self.image_fc = nn.Linear(1000, image_embed_dim)
        self.gnn1 = pyg_nn.GATConv(gnn_embed_dim, hidden_dim, heads=4, concat=True)
        self.gnn2 = pyg_nn.GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=False)
        self.alpha = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
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
        num_embed = self.num_encoder(num_input)
        num_embed = self.num_fc(num_embed)
        image_embed = self.image_encoder(image_input)
        image_embed = self.image_fc(image_embed)
        gnn_embed = self.gnn1(node_features, edge_index)
        gnn_embed = self.gnn2(gnn_embed, edge_index)
        concat_features = torch.cat((num_embed, image_embed, gnn_embed), dim=1)
        fusion_weights = self.gate(concat_features)
        z = fusion_weights[:, 0].unsqueeze(1) * num_embed + fusion_weights[:, 1].unsqueeze(1) * image_embed + fusion_weights[:, 2].unsqueeze(1) * gnn_embed
        out = self.classifier(z)
        return out

def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_input_features = config["hyperparameters"]["num_input_features"]
    image_embed_dim = config["hyperparameters"]["image_embed_dim"]
    gnn_embed_dim = config["hyperparameters"]["gnn_embed_dim"]
    hidden_dim = config["hyperparameters"]["hidden_dim"]
    num_classes = config["hyperparameters"]["num_classes"]

    test_dataset = AccidentDataset(config, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: list(zip(*x)))
    
    model = AccidentPredictionModel(num_input_features, image_embed_dim, gnn_embed_dim, hidden_dim, num_classes).to(device)
    
    model.load_state_dict(torch.load("bts_model.pth", map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            num_inputs, image_inputs, edge_indices, node_features_list, labels = batch
            num_inputs = torch.stack(num_inputs).to(device)
            image_inputs = torch.stack(image_inputs).to(device)
            edge_index = edge_indices[0].to(device)
            node_features = node_features_list[0].to(device)
            outputs = model(num_inputs, image_inputs, edge_index, node_features)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    evaluate(config)
