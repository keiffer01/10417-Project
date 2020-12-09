import torch
import torch.nn as nn

lstm_input = 8
hidden_dim = 16
layer_dim = 2
linear_dim = 864

class RNN(nn.Module):
  '''
  Modified neural network from baseline with CNN and LSTM in parallel training
  on image, with a third linear layer also trained in parallel on the
  corresponding music features.
  '''
  def __init__(self):
    super(RNN, self).__init__()
    self.lstmpool = nn.MaxPool2d(4, 4)
    self.lstm = nn.LSTM(lstm_input, hidden_dim, layer_dim, batch_first=True)
    self.fc1 = nn.Linear(linear_dim, linear_dim//2)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(linear_dim//2, 10)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x_image = x["image"]

    out = self.lstmpool(x_image)
    out = out.view(out.size(0), -1, lstm_input)
    out, (hn, cn) = self.lstm(out)
    out = out.reshape(out.size(0), -1)
    out = self.relu(self.fc1(out))
    out = self.sigmoid(self.fc2(out))
    return out
