import torch.optim as optim
import torch.nn as nn

from data_utils import DataLoader
from ssl_model import SelfSupervisedModel, train_self_supervised
from classifier_model import GestureClassifier, train_classifier, evaluate

# Load and preprocess data
gesture_train = DataLoader('Gesture/train.pt').preprocess()
gesture_test = DataLoader('Gesture/test.pt').preprocess()
gesture_val = DataLoader('Gesture/val.pt').preprocess()
har_train = DataLoader('HAR/train.pt').preprocess()

# Self-supervised learning setup
ssl_model = SelfSupervisedModel()
optimizer = optim.Adam(ssl_model.parameters(), lr=0.001)
train_self_supervised(ssl_model, har_train, optimizer)

# Fine-tune for gesture classification
gesture_classifier = GestureClassifier(ssl_model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gesture_classifier.parameters(), lr=0.001)
train_classifier(gesture_classifier, gesture_train, criterion, optimizer)

# Evaluate on test set
evaluate(gesture_classifier, gesture_test)
