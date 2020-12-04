import torch.nn as nn

convoluted_image_area = 3 * 49

class CNN(nn.Module):
  '''
  Baseline CNN neural network.
  '''
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 4, 7)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(4, 4, 7)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.relu2 = nn.ReLU()
    self.fc = nn.Linear(4 * convoluted_image_area, 10)
    self.sigmoid = nn.Sigmoid()


  def forward(self, x):
    x = self.pool1(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))
    x = x.view(-1, 4 * convoluted_image_area)
    x = self.sigmoid(self.fc(x))
    return x
