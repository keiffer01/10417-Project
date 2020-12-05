import torch
import torch.nn as nn

lstm_input = 8
hidden_dim = 16
layer_dim = 1
linear_dim = 1452

class Parallel(nn.Module):
  '''
  Modified neural network from baseline with CNN and LSTM trained in parallel.
  '''
  def __init__(self):
    super(Parallel, self).__init__()
    self.conv1 = nn.Conv2d(1, 4, 7)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(4, 4, 7)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.relu2 = nn.ReLU()
    self.lstmpool = nn.MaxPool2d(4, 4)
    self.lstm = nn.LSTM(lstm_input, hidden_dim, layer_dim, batch_first=True)
    self.fc = nn.Linear(linear_dim, 10)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x_1 = self.pool1(self.relu1(self.conv1(x)))
    x_1 = self.pool2(self.relu2(self.conv2(x_1)))

    x_2 = self.lstmpool(x)
    x_2 = x_2.view(x_2.size(0), -1, lstm_input)
    x_2, (hn, cn) = self.lstm(x_2)

    out = torch.cat((x_1.reshape(x_1.size(0), -1), x_2.reshape(x_2.size(0), -1)), 1)
    out = self.sigmoid(self.fc(out))
    return out
