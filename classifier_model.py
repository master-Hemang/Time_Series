import torch
import torch.nn as nn

class GestureClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(GestureClassifier, self).__init__()
        self.encoder = pretrained_model.encoder  # Reuse pre-trained encoder
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Assuming 10 classes for gestures
        )

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# Training function for classifier
def train_classifier(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, (x, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in test_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
