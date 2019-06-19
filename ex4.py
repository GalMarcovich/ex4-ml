import torch
import torch.nn as nn
from gcommand_loader import GCommandLoader
#from torch import optim


class ConvNet(nn.Module):
    def __init__(self):
        # calling the init of the "nn.Module" - the father
        super(ConvNet, self).__init__()
        # defining the layers - Conv2d is the filter
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # to avoid over-fitting
        self.drop_out = nn.Dropout()
        self.fullyConn1 = nn.Linear(64000, 1000)
        self.fullyConn2 = nn.Linear(1000, 5)

    # the forward-propagation
    def forward(self, example):
        output = self.layer1(example)
        output = self.layer2(output)
        output = output.reshape(output.size(0), -1)
        output = self.drop_out(output)
        output = self.fullyConn1(output)
        output = self.fullyConn2(output)
        return output


# class FirstNet(nn.Module):
#     def __init__(self, image_size):
#         super(FirstNet, self).__init__()
#         self.image_size = image_size
#         self.fc0 = nn.Linear(image_size, 1000)
#         self.fc1 = nn.Linear(1000, 50)
#         self.fc2 = nn.Linear(50, 10)


    # def forward(self, x):
    #     x = x.view(-1, self.image_size)
    #     x = F.relu(self.fc0(x))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return F.log_softmax(x)


#model = FirstNet(image_size=28 * 28)


if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 8
    num_classes = 10
    batch_size = 100
    learning_rate = 0.01

    dataset = GCommandLoader('./gcommands/train/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    dataset = GCommandLoader('./validation/valid')

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    model = ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # calculate the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # for the back-propagation

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward-propagation pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # # Save the model and plot
    # torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
