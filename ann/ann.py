import torch
import torch.nn as nn

class ANN(nn.Module):
  '''
  Modified neural network from baseline with CNN and LSTM in parallel training
  on image, with a third linear layer also trained in parallel on the
  corresponding music features.
  '''
  def __init__(self):
    super(ANN, self).__init__()
    self.fc1 = nn.Linear(57, 30)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(30, 10)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x_features = x["features"]

    out = self.relu(self.fc1(x_features))
    out = self.sigmoid(self.fc2(out))
    return out
