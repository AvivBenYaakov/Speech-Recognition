import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from gcommand_dataset import GCommandLoader
import sys
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# A network with four convolution layers, and after that three layers of FC.
# In the first convolution layer we use 16 kernels in size 5x5.
# In the second convolution layer we use 32 kernels in size 5x5.
# In the third convolution layer we use 64 kernels in size 6x6.
# In the fourth convolution layer we use 128 kernels in size 3x3.
# We also use max pooling 2x2 with stride = 2.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 6)
        self.conv4 = nn.Conv2d(64, 128, 3)

        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=(1, 0))

        self.fc1 = nn.Linear(4 * 8 * 128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 30)

        self.batch_normalization1 = nn.BatchNorm1d(num_features=120)
        self.batch_normalization2 = nn.BatchNorm1d(num_features=84)
        self.batch_normalization3 = nn.BatchNorm1d(num_features=30)
        self.name = "Net"

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.leaky_relu(self.batch_normalization1(self.fc1(x)), 0.01)
        x = F.leaky_relu(self.batch_normalization2(self.fc2(x)), 0.01)
        x = self.batch_normalization3(self.fc3(x))
        return F.log_softmax(x, dim=1)


# Given a model, his optimizer, and his train loader data,
# we train our network using the nll loss.
def train(my_model, my_optimizer, my_train_loader):
    my_model.train()
    for batch_idx, (data, labels) in enumerate(my_train_loader):
        data = data.to(device)
        labels = labels.to(device)

        my_optimizer.zero_grad()
        output = my_model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        my_optimizer.step()


# Given a model and his data loader we validate the model on the given data,
# and returns the amount of mistake on this data.
def validate(my_model, my_test_loader):
    my_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in my_test_loader:
            data = data.to(device)
            target = target.to(device)

            output = my_model(data)
            predication = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += predication.eq(target.view_as(predication)).to(device).sum()

    return len(my_test_loader.dataset) - correct


# Given a model, and his test loader data, we test the model on the given test,
# and return a list of the predictions.
def test(my_model, my_test_loader):
    my_model.eval()
    predictions = []
    i = 0
    with torch.no_grad():
        for data, labels in my_test_loader:
            data = data.to(device)
            output = my_model(data)
            batch_predications = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            for prediction in batch_predications:
                path, file_name = os.path.split(my_test_loader.dataset.spects[i][0])
                label = train_loader.dataset.classes[prediction]
                predictions.append((file_name, label))
                i += 1

    return predictions


# Write the results to a file: 'test_y'.
def write_results(predictions):
    predictions.sort(key=lambda x: int(x[0].split('.')[0]))

    test_y = open('test_y', "w")
    for prediction in predictions:
        file_name = prediction[0]
        label = prediction[1]
        test_y.write(file_name + ',' + label + '\n')
    test_y.close()


# Given audio files, we need to identify what was said in the audio files
# and classify it from thirty classes.
if __name__ == '__main__':

    train_set = GCommandLoader('./gcommands/train')
    validation_set = GCommandLoader('./gcommands/valid')
    test_set = GCommandLoader('./gcommands/test')

    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False,
                                                    pin_memory=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)

    min_mistake = sys.maxsize
    best_model = None
    best_optimizer = None
    checkpoint = None
    best_model_path = "best_model_path.pth"

    # validation_mistake = amount of mistakes on validation for the model on a specific epoch.
    validation_mistake = 0

    for epoch in range(10):
        # train and validate:
        train(model, optimizer, train_loader)
        validation_mistake = validate(model, validation_loader)

        # find the best model:
        if min_mistake > validation_mistake:
            min_mistake = validation_mistake
            best_model = model
            torch.save({
                'best_model_state_dict': best_model.state_dict()
            }, best_model_path)
            checkpoint = torch.load(best_model_path)

    # load best model:
    best_model.load_state_dict(checkpoint['best_model_state_dict'])

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    class_predictions = test(best_model, test_loader)
    write_results(class_predictions)
