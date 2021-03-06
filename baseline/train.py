import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
# Allows matplotlib plotting on the EC2 instance
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import cnn


### Define constants
data_dir = '../data/images_segmented'
net_dir = './baseline_net.pth'
saved_loss_and_accuracy_dir = './saved_loss_and_accuracy.npz'
num_epoch = 200
batch_size = 32
num_points = 10000
train_size = 8000
test_size = 2000
lr = 0.001
momentum = 0.9
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


### Load the data
dataset = datasets.ImageFolder(data_dir, transform=transform)
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=test_size,
                                         shuffle=False,
                                         num_workers=4)
classes = ('blues', 'classical', 'country', 'disco',
           'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')


### Train the network
net = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
loss_per_epoch_train = []
loss_per_epoch_test = []
accuracy_per_epoch_train = []
accuracy_per_epoch_test = []

for e in trange(num_epoch):
  running_loss = 0.0
  correct = 0
  iters = 0
  total = 0

  for data in trainloader:
    iters += 1
    inputs, labels = data
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  # Get average training loss and accuracy for this epoch
  loss_per_epoch_train.append(running_loss / iters)
  accuracy_per_epoch_train.append(100 * correct / total)

  # Get test loss and accuracy for this epoch
  running_loss = 0.0
  correct = 0
  iters = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      iters += 1
      images, labels = data
      outputs = net(images)
      loss = criterion(outputs, labels)

      running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  loss_per_epoch_test.append(running_loss / iters)
  accuracy_per_epoch_test.append(100 * correct / total)

# Save the trained net
torch.save(net.state_dict(), net_dir)

# Save the losses and accuracies
np.savez(saved_loss_and_accuracy_dir,
         loss_per_epoch_train,
         accuracy_per_epoch_train,
         loss_per_epoch_test,
         accuracy_per_epoch_test)

epoch_list = range(1, num_epoch+1)

# Plot train/test loss
plt.plot(epoch_list, loss_per_epoch_train, color="blue", label="Train loss")
plt.plot(epoch_list, loss_per_epoch_test, color="red", label="Test loss")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test Loss")
plt.savefig("./figures/loss.png", bbox_inches="tight")
plt.close()

# Plot train/test accuracy
plt.plot(epoch_list, accuracy_per_epoch_train, color="blue", label="Train accuracy")
plt.plot(epoch_list, accuracy_per_epoch_test, color="red", label="Test accuracy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train/Test Accuracy")
plt.savefig("./figures/accuracy.png", bbox_inches="tight")
plt.close()
