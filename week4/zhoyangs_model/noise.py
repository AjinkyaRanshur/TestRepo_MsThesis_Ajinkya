import torch

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

