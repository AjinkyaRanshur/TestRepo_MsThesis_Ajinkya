import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size=1

transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def noisy_img(image,noisetype,noiseparameter):

    if noisetype == "gauss":
        gaussian_noise=torch.randn(image.shape)*noiseparameter
        noisy=image+gaussian_noise
        return noisy

    if noisetype == "s&p":
        #They are used as thresholds to decide which pixel get salt and which get pepper
        decide_salt=noiseparameter/2 * torch.ones(image.shape[2],image.shape[3])
        decide_pepper=(1-noiseparameter/2) * torch.ones(image.shape[2],image.shape[3])
        noisy=image
        #Random values in the image to decide what happens to each image
        saltNpepper=torch.rand(image.shape[2],image.shape[3])


        #The image tensor but the value of each pixel is maximum
        add_salt=torch.max(image) * torch.ones(image.shape)
        add_pepper=torch.max(image) * torch.ones(image.shape)

        noisy=torch.where(saltNpepper >= decide_salt,noisy,add_salt)
        noisy=torch.where(saltNpepper<= decide_pepper,noisy,add_pepper)

        return noisy
        

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)


#images_modif=noisy_img(images,"gauss",0.8)
images_modif=noisy_img(images,"s&p",0.1)
# show images
imshow(torchvision.utils.make_grid(images))
imshow(torchvision.utils.make_grid(images_modif))


