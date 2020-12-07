import torch
import torch.nn as nn

lstm_input = 8
hidden_dim = 16
layer_dim = 1
linear_dim = 1472

class ParallelV2(nn.Module):
  '''
  Modified neural network from baseline with CNN and LSTM trained in parallel.
  '''
  def __init__(self):
    super(ParallelV2, self).__init__()
    self.conv1 = nn.Conv2d(1, 4, 7)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.reluconv1 = nn.ReLU()
    self.conv2 = nn.Conv2d(4, 4, 7)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.reluconv2 = nn.ReLU()
    self.lstmpool = nn.MaxPool2d(4, 4)
    self.lstm = nn.LSTM(lstm_input, hidden_dim, layer_dim, batch_first=True)
    self.fc1 = nn.Linear(57, 30)
    self.relulin1 = nn.ReLU()
    self.fc2 = nn.Linear(30, 20)
    self.relulin2 = nn.ReLU()
    self.fcfinal = nn.Linear(linear_dim, 10)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x_image = x["image"]
    x_features = x["features"]

    x_1 = self.pool1(self.reluconv1(self.conv1(x_image)))
    x_1 = self.pool2(self.reluconv2(self.conv2(x_1)))

    x_2 = self.lstmpool(x_image)
    x_2 = x_2.view(x_2.size(0), -1, lstm_input)
    x_2, (hn, cn) = self.lstm(x_2)

    x_3 = self.relulin1(self.fc1(x_features))
    x_3 = self.relulin2(self.fc2(x_3))

    out = torch.cat((x_1.reshape(x_1.size(0), -1),
                     x_2.reshape(x_2.size(0), -1),
                     x_3.reshape(x_3.size(0), -1)),
                     1)
    out = self.sigmoid(self.fcfinal(out))
    return out
