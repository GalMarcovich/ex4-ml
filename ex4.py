import torch.nn as nn
from gcommand_loader import GCommandLoader
import torch


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


def do_back_propagation(loss):
    # Back-propagation and perform Adam optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Train the model
def train_data(train_loader, criterion, optimizer, model):
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

            do_back_propagation(loss)

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Example [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


def print_func(correct, total):
    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))


# Test the model
def test_data(test_loader, model):
    print("in test_data")
    # Test the model
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print_func(correct, total)


# Write to file the results of the test
def write_to_file(test_loader, test_x, model):
    list_commands = []
    list_y_hat = []
    enter = '\n'
    comma = ', '
    zero = 0
    one = 1

    file = open("test_y", "w")

    # the path of each file
    for x in test_x:
        x = x[zero]
        x = x.split("/")
        y = x[len(x)-one]
        list_commands.append(y)

    # the prediction of each file
    for voices, labels in test_loader:
        outputs = model(voices)
        _, y_hat = torch.max(outputs.data, 1)
        list_y_hat.extend(y_hat.tolist())

    # write each path of file and it's prediction into the file
    for x, y in zip(list_commands, list_y_hat):
        file.write(x + comma + str(y) + enter)
    file.close()


if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 30
    len_image = 101
    len_image_2 = 161
    image_size = len_image * len_image_2
    size_of_batch = 100

    # get the data-set
    dataset = GCommandLoader('./gcommands/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=size_of_batch, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    # get the validation-set
    validation_set = GCommandLoader('./gcommands/valid')

    valid_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=size_of_batch, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    # get the test-set
    test_set = GCommandLoader('./gcommands/test')

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=size_of_batch, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    model = ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # calculate the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # for the back-propagation

    # train the model
    train_data(train_loader, criterion, optimizer, model)

    # test the model
    test_data(valid_loader, model)

    # write to file the results of the test
    write_to_file(test_loader, test_set.spects, model)
