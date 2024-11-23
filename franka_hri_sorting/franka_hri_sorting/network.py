import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch


# Define the neural network architecture
class SortingNet(nn.Module):
    def __init__(self):
        super(SortingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 64 * 64, 16)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

        # Learning rate
        self.lr_init = 0.00002
        self.lr_decay = 0.98

        # Create optimizer and criterion
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
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

class GestureNet(nn.Module):
    def __init__(self, sequence_length=20):
        super(GestureNet, self).__init__()
        
        self.input_channels = 4
        self.sequence_length = sequence_length
        
        # Simple but effective architecture
        self.conv1 = nn.Conv1d(self.input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions and pooling
        conv_output_size = (sequence_length // 4) * 32
        
        # Simple feedforward layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # Higher learning rate for faster adaptation
        self.lr_init = 0.0005
        self.lr_decay = 0.98
        
        # Standard BCE loss - simple and effective
        self.criterion = nn.BCELoss()
        
        # Adam optimizer with higher learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        
        # Simple learning rate schedule
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

    def train_network(self, gestures, labels):
        # Train on each sequence
        for i in range(len(gestures)):
            self.optimizer.zero_grad()

            gesture = gestures[i]
            label = labels[i]

            # Forward pass
            output = self.forward(gesture)
            loss = self.criterion(output, label)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()