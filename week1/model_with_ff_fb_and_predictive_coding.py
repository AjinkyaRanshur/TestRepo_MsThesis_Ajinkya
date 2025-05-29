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

#trainset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=True,download=True,transform=transform)
trainset=torchvision.datasets.CIFAR10(root='"D:\datasets"',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
#testset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=False,download=True,transform=transform)
testset=torchvision.datasets.CIFAR10(root='"D:\datasets"',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)

classes= ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

epochs=4

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
        self.deconv2_fb=nn.ConvTranspose2d(16, 6, 5,stride=1, padding=0)
        self.unpool1=nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)
        self.deconv1_fb=nn.ConvTranspose2d(6, 3, 5,stride=2, padding=2, output_padding=1)
        self.final_upsample=nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)


    def forward(self,x,direction):
        if direction=="forward":
            ft_AB=self.pool(F.relu(self.conv1(x)))
            ft_BC=self.pool(F.relu(self.conv2(ft_AB)))
            ft_BC=torch.flatten(ft_BC,1) #Flatten all dimensions except batch
            ft_CD=F.relu(self.fc1(ft_BC))
            ft_DE=F.relu(self.fc2(ft_CD))
            output=self.fc3(ft_DE)

            return ft_AB,ft_BC,ft_CD,ft_DE,output

        if direction=="backward":
            #You get the features from the forward direction
            ft_AB=self.pool(F.relu(self.conv1(x)))
            ft_BC_will_flatten=self.pool(F.relu(self.conv2(ft_AB)))
            ft_BC=torch.flatten(ft_BC_will_flatten,1) #Flatten all dimensions except batch
            ft_CD=F.relu(self.fc1(ft_BC))
            ft_DE=F.relu(self.fc2(ft_CD))
            output=self.fc3(ft_DE)
            #Now once you have the features you just go back in the opposite direction and then reconstruct the input and adjust the weights based on the error 
            ft_ED=F.relu(self.fc3_fb(output))
            ft_DC=F.relu(self.fc2_fb(ft_ED))
            ft_CB=F.relu(self.fc1_fb(ft_DC))
            ft_CB = ft_CB.view(-1, 16, 5, 5)
            ft_BA=F.relu(self.deconv2_fb(ft_CB))
            ft_BA=self.unpool1(ft_BA)
            x=F.relu(self.deconv1_fb(ft_BA))
            #This should be the input
            x=self.final_upsample(x) 
            return ft_AB,ft_BA,ft_BC_will_flatten,ft_CB,ft_CD,ft_DC,ft_DE,ft_ED,x

def evaluation_metric(net,direction):
    # Testing
    net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            ft_AB,ft_BC,ft_CD,ft_DE,output = net(images,direction)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * (total_correct / total_samples)


    return accuracy

#It's showing the error not the accuracy fix that

def evaluation_reconstruction(net,direction):
    net.eval()
    total_pixels = 0
    correct_pixels = 0
    threshold=0.1
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            _,_,_,_,_,_,_,_,xpred = net(images,direction)
            diff=torch.abs(xpred-images)
            correct=(diff<threshold).float().sum().item()
            total=images.numel()
            correct_pixels=+correct
            total_pixels=+total

    accuracy = 100 * (correct_pixels / total_pixels)

    return accuracy



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
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_fwd.zero_grad()
            ft_AB,ft_BC,ft_CD,ft_DE,ypred = net(images,"forward")
            loss = criterion(ypred, labels)
            loss.backward()
            optimizer_fwd.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_metric(net,"forward")
    iters = range(1, epochs+1)
    plot_bool=plot_metrics(iters,loss_arr,"forward")
    if plot_bool==True:
        print("Plots Successfully Stored")
    print(f'Accuracy = {accuracy:.2f}%')

    print("Forward Training Succesful")

    return ft_AB,ft_BC,ft_CD,ft_DE,ypred



def feedback_training(net,ft_AB,ft_BC,ft_CD,ft_DE,output):
    net.train()
    criterion_recon = nn.MSELoss()
    optimizer_bck = optim.SGD(list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+list(net.fc1_fb.parameters())+list(net.fc2_fb.parameters())+list(net.fc3_fb.parameters()), lr=0.001, momentum=0.9)
    loss_arr = []
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_bck.zero_grad()
            ft_AB,ft_BA,ft_BC,ft_CB,ft_CD,ft_DC,ft_DE,ft_ED,xpred = net(images,"backward")
            lossAtoB = criterion_recon(ft_AB, ft_BA)
            lossBtoC = criterion_recon(ft_BC, ft_CB)
            lossCtoD = criterion_recon(ft_CD, ft_DC)
            lossDtoE = criterion_recon(ft_DE, ft_ED)
            loss_input_and_recon = criterion_recon(xpred, images)
            final_loss=lossAtoB+lossBtoC+lossCtoD+lossDtoE+loss_input_and_recon
            final_loss.backward()
            optimizer_bck.step()
            running_loss.append(final_loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_reconstruction(net,criterion_recon)
    iters = range(1, epochs+1)
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
    ft_AB,ft_BC,ft_CD,ft_DE,output=feedfwd_training(net)
    #visualize_model(net)
    feedback_training(net,ft_AB,ft_BC,ft_CD,ft_DE,output)




# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()
