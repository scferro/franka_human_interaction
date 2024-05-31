import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the neural network architecture
class SortingNet(nn.Module):
    def __init__(self):
        super(SortingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # Learning rate
        self.lr_init = 0.00001
        self.lr_decay = 0.98

        # Create optimizer and criterion
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def train_network(self, images, label_tensors):
        for i in range(len(images)):
        # Clear the gradients
            self.optimizer.zero_grad()

            image = images[i]
            label_tensor = label_tensors[i]

            # Forward pass
            output = self.forward(image)

            # Calculate the loss
            loss = self.criterion(output, label_tensor)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

        # Update learning rate for next iteration
        self.scheduler.step()