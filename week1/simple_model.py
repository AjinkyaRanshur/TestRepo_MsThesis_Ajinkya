#Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

#Normalizing the images
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

batch_size=4

trainset=torchvision.datasets.CIFAR10(root='D:\datasets',train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='D:\datasets',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)

classes= ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1) #Flatten all dimensions except batch
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

def evaluation_metric(net):
    # Testing
    net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * (total_correct / total_samples)


    return accuracy

def plot_metrics(x,y):

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2, markersize=6)
    plt.title('Training Evaluation')
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig('avgloss_vs_epoch.png')

    return True


def fwd_training(net,criterion):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_arr = []
    for epoch in range(4):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer.zero_grad()
            ypred = net(images)
            loss = criterion(ypred, labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_metric(net)
    iters = range(1, 5)
    plot_bool=plot_metrics(iters,loss_arr)
    if plot_bool==True:
        print("Plots Successfully Stored")
    print(f'Accuracy = {accuracy:.2f}%')

def main():
    # Your training and testing code goes here
    net = Net()
    criterion = nn.CrossEntropyLoss()
    fwd_training(net,criterion)
    writer = SummaryWriter('runs/cifar10_experiment')
    sample_input = torch.randn(1, 3, 32, 32)
    writer.add_graph(net, sample_input)
    writer.close()  






# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()
