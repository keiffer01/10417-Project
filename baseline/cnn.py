import torch.nn as nn

convoluted_image_area = 65 * 1
image_height = 217

class CNN(nn.Module):
  '''
  Baseline CNN neural network.
  '''
  def __init__(self):
    super(CNN, self).__init__()
    self.conv = nn.Conv2d(3, 7, (image_height, 3))
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool3d((4, 1, 1))
    self.linear = nn.Linear(convoluted_image_area, 10)
    self.sigmoid = nn.Sigmoid()


  def forward(self, x):
    x = self.pool(self.relu(self.conv(x)))
    x = x.view(-1, convoluted_image_area)
    x = self.sigmoid(self.linear(x))
    return x
