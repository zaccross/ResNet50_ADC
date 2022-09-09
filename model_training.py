"""
Zac Cross
Dr. Fuding Lin
KCGIP Summer 2022

Final Internship Project

------------------------------------------------------------------------------------------------------------------------
My ResNet Model for Metal Surface Defect Classification.

GOAL: Use state of the art ML architecture to work as automatic defect classifier (ADC).

My overall aim is to use this in the context of semiconductor manufacturing, but due to a lack of open data, this is a
proof of concept that these techniques apply to surface defects, similar to larger semiconductor defects or bare wafer
deformities and defects.

------------------------------------------------------------------------------------------------------------------------
(1) Training structure taken from: https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
    with tweaks to better suit the task.

(2) ResNet50 Model from Aladdin Persson <aladdin.persson at hotmail dot com>
    *    2020-04-12 Initial coding.

(3) Data Set taken and modified from North Eastern University's NEU-DET dataset. I have set up the architecture to work
    with all 28800 augmented images,  but due to time / resource constraints we limit to the
    original 1800 image data set.
------------------------------------------------------------------------------------------------------------------------
"""
# Useful imports
import torch # Our heavy lifter for all ML related things
import Resnet as r # The resnet50 model
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam # Defining Loss function
import matplotlib.pyplot as plt
import numpy as np

class Run_Exp:
    """Container class for the whole experiment"""

    def __init__(self, root, model, epochs, learning_rate, batch, num_exp):

        self.root = root
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch = batch
        self.num = num_exp

        self.lossfn = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0000)

        self.transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train, self.test = self._load_data()

        # save a list to plot this later
        self.epoch_loss = []



    def _load_data(self):
        """Load in training and testing data sets"""
        data = torchvision.datasets.ImageFolder(root=self.root, transform=transforms.ToTensor())
        train, test = torch.utils.data.random_split(data, [int(0.9*self.num), int(self.num * 0.1)])
        return train, test

    def training(self):
        best_accuracy = 0.0

        # not sure if i need
        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", device, "device")
        # Convert model parameters and buffers to CPU or Cuda
        model.to(device)
        ######
        train_len = int(.9* self.num * 0.8)
        valid_len = int(.9*self.num * .2)

        # divide into training and validation each epoch
        train, valid = torch.utils.data.random_split(self.train, [train_len, valid_len])
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch, shuffle=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=self.batch, shuffle=True, num_workers=0)

        # counter incase training plataeus
        epoch_counter = 0
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_acc = 0.0

            for i, (images, labels) in enumerate(train_loader, 0):
                # get the inputs
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = self.model(images)
                # compute the loss based on model output and real labels
                loss = self.lossfn(outputs, labels)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                self.optimizer.step()

                # Let's print statistics for every 1,000 images
                running_loss += loss.item()  # extract the loss value
                if i % 100 == 99:
                    # print every 1000 (twice per epoch)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    # zero the loss
                    running_loss = 0.0

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = self._valid_accuracy(valid_loader)
            test_accuracy = self._valid_accuracy(train_loader)
            self.epoch_loss.append(100-accuracy)
            print('For epoch', epoch + 1, 'the validation accuracy over the whole validation set is\n %d %%' % (accuracy))
            print('For epoch', epoch + 1,
                  'the training accuracy over the whole training set is\n %d %%' % (test_accuracy))
            # move on if training plateus


            if abs(accuracy - best_accuracy) <= 0.005 or accuracy < best_accuracy:
                epoch_counter += 1
            else:
                epoch_counter = 0
            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                saveModel()
                best_accuracy = accuracy
            # end training if we go the wrong direction or plateua
            if epoch_counter > 4:
                break
            # save a copy in case curious
            self.check_point(epoch)

        # load best model to finish
        path = "Metal_Model_II.pth"
        self.model = model.load_state_dict(torch.load(path))

    def _valid_accuracy(self, valid_loader):
        self.model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in valid_loader:
                images, labels = data
                # run the model on the test set to predict labels
                outputs = model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        accuracy = (100 * accuracy / total)
        return accuracy

    def testAccuracy(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0

        test_loader = torch.utils.data.DataLoader(self.test, batch_size=self.batch, shuffle=True, num_workers=0)
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # run the model on the test set to predict labels
                outputs = model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return (accuracy)

    def model_performance(self):
        test_acc = self.testAccuracy()
        print("###############################")
        print(f"# Final Test Error:   {round(100 - test_acc, 2)}%  ##")
        print("###############################")


        x = list(range(0,self.epochs))

        plt.plot(x, self.epoch_loss)
        plt.title("Validation Loss at each epoch")
        plt.ylabel("Validation Error %")
        plt.xlabel("Epoch")
        plt.show()




    def check_point(self, i):
        """Check Point model by epoch"""
        path = f"./checkpoint/I_epoch{i}.pth"
        torch.save(model.state_dict(), path)


def saveModel():
    """Saves the model to determined path.
    """
    path = "./Metal_Model_II.pth"
    torch.save(model.state_dict(), path)


if __name__ == "__main__":

    # Set path to pull model from
    path = './Metal_Model_II.pth'
    # Pull Model from ResNet File
    model = r.ResNet50(3, 6)
    # Save previous training, comment out if no model exists yet
    model.load_state_dict(torch.load(path))
    # Build training class
    EXP = Run_Exp(root='./NEU-DET/images', model=model, epochs=1000, learning_rate=0.001, batch=4, num_exp=28800)
    print("Experiment Class Built and Data Loaded")
    # Train!!!!
    EXP.training()
    print('Finished Training')
    # Check Results
    # Test which classes performed well
    EXP.model_performance()
