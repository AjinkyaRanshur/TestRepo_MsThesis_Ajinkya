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

        self.fc3_fb=nn.Linear(10,84)
        self.fc2_fb=nn.Linear(84,120)
        self.fc1_fb=nn.Linear(120,16*5*5)
        self.deconv2_fb=nn.ConvTranspose2d(16, 6, 5,stride=2, padding=2, output_padding=1)
        self.unpool1=nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1_fb=nn.ConvTranspose2d(6, 3, 5,stride=2, padding=2, output_padding=1)
        self.final_upsample=nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)


    def forward(self,x,direction):
        if direction=="forward":
            x=self.pool(F.relu(self.conv1(x)))
            x=self.pool(F.relu(self.conv2(x)))
            x=torch.flatten(x,1) #Flatten all dimensions except batch
            x=F.relu(self.fc1(x))
            x=F.relu(self.fc2(x))
            x=self.fc3(x)
            return x
        if direction=="backward":
            #You get the features from the forward direction
            x=self.pool(F.relu(self.conv1(x)))
            x=self.pool(F.relu(self.conv2(x)))
            x=torch.flatten(x,1) #Flatten all dimensions except batch
            x=F.relu(self.fc1(x))
            x=F.relu(self.fc2(x))
            x=self.fc3(x)
            #Now once you have the features you just go back in the opposite direction and then reconstruct the input and adjust the weights based on the error 
            x=F.relu(self.fc3_fb(x))
            x=F.relu(self.fc2_fb(x))
            x=F.relu(self.fc1_fb(x))
            x = x.view(-1, 16, 5, 5)
            x=F.relu(self.deconv2_fb(x))
            x=self.unpool1(x)
            x=F.relu(self.deconv1_fb(x))
            #This should be the input
            x=self.final_upsample(x) 
            return x

def evaluation_metric(net,direction):
    # Testing
    net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            outputs = net(images,direction)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * (total_correct / total_samples)


    return accuracy

#It's showing the error not the accuracy fix that

def evaluation_reconstruction(net,criterion_recon):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in testloader:
            xpred = net(images, "backward")
            total_loss += criterion_recon(xpred, images).item()
    return total_loss / len(testloader)


def plot_metrics(x,y,direction):

    if direction=="forward":
        title="Forward Training Evaluation"

    if direction=="backward":
        title="Backward Training Evaluation"
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2, markersize=6)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(f'avgloss_vs_epoch_{direction}.png')

    return True


def feedfwd_training(net):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_fwd = optim.SGD(list(net.conv1.parameters())+list(net.conv2.parameters())+list(net.fc1.parameters())+list(net.fc2.parameters())+list(net.fc3.parameters()), lr=0.001, momentum=0.9)
    loss_arr = []
    for epoch in range(4):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_fwd.zero_grad()
            ypred = net(images,"forward")
            loss = criterion(ypred, labels)
            loss.backward()
            optimizer_fwd.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_metric(net,"forward")
    iters = range(1, 5)
    plot_bool=plot_metrics(iters,loss_arr,"forward")
    if plot_bool==True:
        print("Plots Successfully Stored")
    print(f'Accuracy = {accuracy:.2f}%')

    print("Forward Training Succesful")


def feedback_training(net):
    net.train()
    criterion_recon = nn.MSELoss()
    optimizer_bck = optim.SGD(list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+list(net.fc1_fb.parameters())+list(net.fc2_fb.parameters())+list(net.fc3_fb.parameters()), lr=0.001, momentum=0.9)
    loss_arr = []
    for epoch in range(4):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_bck.zero_grad()
            xpred = net(images,"backward")
            loss = criterion_recon(xpred, images)
            loss.backward()
            optimizer_bck.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_reconstruction(net,criterion_recon)
    iters = range(1, 5)
    plot_bool=plot_metrics(iters,loss_arr,"backward")
    if plot_bool==True:
        print("Plots Successfully Stored")
    print(f'Backward Connections Accuracy = {accuracy:.2f}%')

    print("Backward Training Succesful")

def predictive_coding_training(net):

    gammaFw = 1.0 
    alphaRec = 1.0
    betaFB =  1.0
    memory = 1.0

    
    
    return None

def visualize_model(net):
    writer = SummaryWriter('runs/cifar10_experiment')
    sample_input = torch.randn(1, 3, 32, 32)
    writer.add_graph(net, sample_input)
    writer.close()
    #To launch tensorboard use this command: tensorboard --logdir=runs and then click on the link that it generates  


def main():
    # Your training and testing code goes here
    net = Net()
    feedfwd_training(net)
    #visualize_model(net)
    feedback_training(net)



# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()
