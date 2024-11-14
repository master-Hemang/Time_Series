import torch
import torch.nn as nn
import numpy
class SelfSupervisedModel(nn.Module):
    def __init__(self):
        super(SelfSupervisedModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # Aggregate features
        )
        self.fc = nn.Linear(128, 64)  # Project into lower dimension

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

# Contrastive Loss (using NT-Xent)
def contrastive_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim_ij = torch.diag(sim_matrix, len(z_i))
    sim_ji = torch.diag(sim_matrix, -len(z_i))
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    negatives = sim_matrix[~torch.eye(2 * len(z_i), dtype=bool)].view(2 * len(z_i), -1)
    labels = torch.zeros(len(positives), dtype=torch.long)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temperature
    return F.cross_entropy(logits, labels)

# Self-supervised training loop
def train_self_supervised(model, data, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, (x_i, x_j) in enumerate(data):
            optimizer.zero_grad()
            z_i, z_j = model(x_i), model(x_j)
            loss = contrastive_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
