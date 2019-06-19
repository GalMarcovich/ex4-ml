import torch.nn as nn
from gcommand_loader import GCommandLoader
import torch

# Hyperparameters
num_epochs = 10
num_classes = 30
image_size = 101 * 161
learning_rate = 0.01


class ConvNet(nn.Module):

    def __init__(self):
        # calling the init of the nn.Module
        super(ConvNet, self).__init__()
        # defining the layers and creating the filters - creates a set of convolutional filters
        # the first param is the num of input channel
        # the second param is the num of output channel
        # the third param is the kernel_size - the filter size is 5X5
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # use drop-out to avoid over-fitting
        self.drop_out = nn.Dropout()
        # to create a fully connected layer:
        # first layer will be of size 64,000 nodes and will connect to the second layer of 1000 nodes
        self.fullyConn1 = nn.Linear(5880, 1000)
        # second layer will be of size 1000 nodes and will connect to the output layer of 100 nodes
        self.fullyConn2 = nn.Linear(1000, 100)
        # third layer will be of size 100 nodes and will connect to the output layer of 30 nodes - the num of classes
        self.fullyConn3 = nn.Linear(100, num_classes)

    # the forward-propagation
    def forward(self, example):
        output = self.layer1(example)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.size(0), -1)
        output = self.drop_out(output)
        output = self.fullyConn1(output)
        output = self.fullyConn2(output)
        output = self.fullyConn3(output)
        return output


# Train the model
def train_data(train_loader, optimizer, criterion, model):
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward-propagation pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Back-propagation and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Example [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


# Test the model
def test_data(test_loader, model):
    print("in test_data")
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))


def write_to_file(self, test_x, model):
    file = open("test_y", "w")
    for images, labels in test_x:
        outputs = model(images)
        _, y_hat = torch.max(outputs.data, 1)
        # file.write( example + ','+ str(y_hat) + '\n')
    file.close()


if __name__ == "__main__":
    dataset = GCommandLoader('./ML4_dataset/data/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    validation_set = GCommandLoader('./ML4_dataset/data/valid')

    valid_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    model = ConvNet()
    # model.write_to_file(valid_loader, model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # calculate the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # for the back-propagation

    # train the model
    train_data(train_loader, optimizer, criterion, model)

    # test the model
    test_data(valid_loader, model)
