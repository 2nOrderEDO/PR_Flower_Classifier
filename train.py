# Programed by Enrique Corpa Rios
# Udacity Introduction to AI programming with python
# Final project 17 Aug 2018

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as False

import argparse


from torchvision import transforms, models, datasets
from PIL import Image


class  NetworkNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        if isinstance((self.hidden_layers), list):

            for linear in self.hidden_layers:
                x = F.relu(linear(x))
                x = self.dropout(x)
            
            x = self.output(x)

        else:
            layers = np.power(range(self.hidden_layers), -2)
            for linear in layers: # hacer que se reduzcan en potencias de dos
                x = F.relu(linear(x))
                x = self.dropout(x)
            
            x = self.output(x)
        
        return F.log_softmax(x, dim=1)


def main():

    data_dir = in_arg.database
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Transform for training augmentation
    train_transforms =  transforms.Compose([transforms.RandomRotation(15),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform =data_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

    print('Starting execution')

    if in_arg.vgg != None:
        model = models.vgg16(pretrained=True)
    elif in_arg.resnet != None:
        model = models.resnet(pretrained=True)
    else:
        print('Model not specified, using VGG16 by default')
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # TODO: implementar la estructura del clasificador
    classifier = Network(784, 100, 3, drop_p=0.5)

    model.classifier = classifier
    model.to(device)

    model = train(model, trainloader, testloader, validationloader)

    print('Training completed')


def train(model, trainloader, testloader, validationloader):
    epochs = in_arg.epochs
    running_loss = 0
    steps = 0
    print_every = 30

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr = in_arg.lr) # 0.01 SGD

    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    accuracy, test_loss = validation(model, validationloader, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                model.train()
    model.eval()
    return model

def validation(model, loader, criterion):
    correct = 0
    total = 0
    test_loss = 0
    for inputs, labels, in loader:

        inputs, labels = inputs.to(device), labels.to(device)
        total += len(labels)
        output = model(inputs)

        _, predicted = torch.max(output.data, 1)
        test_loss += criterion(output, labels).item()

        correct += (labels == predicted).sum().item()
    
    accuracy = correct / total

    return accuracy, test_loss

def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='./flowers', 
    help='path to root database images')

    parser.add_argument('--nn_param', type=np.array, default=None, 
    help='path to root database images')

    parser.add_argument('--vgg', type=str, 
    help='Starting model is VGG16')

    parser.add_argument('--resnet', type=str, 
    help='Starting model is VGG16')

    parser.add_argument('--lr', type=float, default=0.01, 
    help='Sets the learning rate for training')

    parser.add_argument('--epochs', type=int, default=2, 
    help='Number of epochs')

    parser.add_argument('--cuda', type=bool, default=True, 
    help='Execute code on GPU')

    return parser.parse_args()

if __name__ == "__main__":

    in_arg = get_input_args()

    if(in_arg.cuda and bool(torch.cuda.is_available())):
        device = 'cuda'
        print('Executing model in GPU')
    elif (not torch.cuda.is_available):
        device = 'cpu'
        print('Your system is not compatible with CUDA')
    else:
        device = 'cpu'
        print('Executing model in CPU')

    main()