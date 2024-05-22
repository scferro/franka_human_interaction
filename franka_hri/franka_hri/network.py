import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the neural network architecture
class SortingNet(nn.Module):
    def __init__(self):
        super(BlockSortingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        # Create optimizer and criterion
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def train_network(self, image, label):

        # Convert the label to a tensor
        label = torch.tensor([label], dtype=torch.float32)

        # Clear the gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.forward(image.unsqueeze(0))

        # Calculate the loss
        loss = self.criterion(output, label)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
