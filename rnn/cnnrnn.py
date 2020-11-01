import torch
import torch.nn as nn

convoluted_image_area = 12 * 49
hidden_dim = 100
layer_dim = 1

class CNNRNN(nn.Module):
  '''
  Baseline CNN neural network.
  '''
  def __init__(self):
    super(CNNRNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 4, 7)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(4, 4, 7)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.relu2 = nn.ReLU()
    self.lstm = nn.LSTM(2352, hidden_dim, layer_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 10)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.pool1(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))
    x = x.view(x.size(0), 1, -1)
    x, (hn, cn) = self.lstm(x)
    x = self.sigmoid(self.fc(x[:, -1, :]))
    return x