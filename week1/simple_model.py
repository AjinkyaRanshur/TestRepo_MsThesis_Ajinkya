#Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Normalizing the images
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

batch_size=4

trainset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets/cifardataset',train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)

testset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets/cifar10dataset',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=4)

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


net=Net()

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#Training Module 
#We start with some epochs and then iterate through each batch in the trainloader we calculate the loss for each image and then average it over each epoch 

loss_arr=[]
for epoch in range(4):
    
    running_loss=[]
    #running_loss=np.array(running_loss)
    for batch_idx,batch in enumerate(trainloader):
        images,labels=batch
        optimizer.zero_grad()
        ypred=net(images)
        loss=criterion(ypred,labels)
        #Used to compute the gradients of the network
        loss.backward()
        #This is used to update the weights of the network
        optimizer.step()
        #print(type(loss.item()))
        running_loss.append(loss.item())
        
    avg_loss=np.mean(running_loss)
    print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
    loss_arr.append(avg_loss)

#Testing Model

#Set the model into evaluation mode
net.eval() 

#for epoch in range(4):
    
total_correct=0
total_samples=0

with torch.no_grad():
    for batch_idx,batch in enumerate(testloader):
        images,labels=batch
        outputs=net(images)
        #torch.max outputs the probability along specified dimension in this case 1
        _,predicted=torch.max(outputs,1)
        total_correct+=(predicted == labels).sum().item()
        total_samples+=labels.size(0)

accuracy=100 * (total_correct/total_samples)
print(f'Accuracy = {accuracy:.2f}%')

#print(loss_arr)
iters=range(1,5)

plt.figure(figsize=(8,6))
plt.plot(iters,loss_arr,linewidth=2,markersize=6)
plt.title('Training Evaluation')
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.xticks(iters)
plt.tight_layout()
plt.savefig('tr_loss.png')


