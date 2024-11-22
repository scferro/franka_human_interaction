import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


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
        
        # Input channels: 4 (3 from mod48 + 1 from mod51, after dropping timestamps)
        self.input_channels = 4  # Changed from 5 to 4
        self.sequence_length = sequence_length
        
        # 1D convolution layers for temporal feature extraction
        self.conv1 = nn.Conv1d(self.input_channels, 32, kernel_size=3, stride=1, padding=1)  # Input channels now 4
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions and pooling
        conv_output_size = (sequence_length // 4) * 64  # Divided by 4 due to two pooling layers
        
        # Rest of the network remains the same
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(p=0.3)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize training parameters
        self.lr_init = 0.001
        self.lr_decay = 0.98
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
                            where channels = 4 (combined modalities features)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

    def train_network(self, gestures, labels):
        for i in range(len(gestures)):
            # Clear the gradients
            self.optimizer.zero_grad()

            gesture = gestures[i]
            label = labels[i]

            # Forward pass
            output = self.forward(gesture)

            # Calculate the loss
            loss = self.criterion(output, label)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

        # Update learning rate for next iteration
        self.scheduler.step()

    def train_single_gesture(self, gesture, label):
        """Method for online learning of single gestures"""
        self.optimizer.zero_grad()
        
        output = self.forward(gesture)
        loss = self.criterion(output, label)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()