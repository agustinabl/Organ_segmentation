import os
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from tcia_utils import nbia  #cancer imaging archive
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImaged, Orientation, Orientationd, EnsureChannelFirst, EnsureChannelFirstd, Compose
from rt_utils import RTStructBuilder
import json 

datadir= "C:\.Projects\Medical_storage\Data"

#Part 1 
image_loader=LoadImage(image_only=True)
CT= image_loader(CTfolder)
CT.meta 
CT_coronal=CT[:,220].cpu().numpy()
#view image
plt.figure(figsize=(3,8))
plt.pcolormesh(CT_coronal.T, cmap="Greys_r")
plt.colorbar(label="HU")
plt.axis("off")
plt.show()
CT.shape

#transform
channeltransf=EnsureChannelFirst()
CT=channeltransf(CT)
CT.shape
#reorient image
orientransf=Orientation(axcodes=("LPS"))
CT= orientransf(CT)
CT_coronal= CT[0,:,256].cpu().numpy()

#view image
plt.figure(figsize=(3,8))
plt.pcolormesh(CT_coronal.T, cmap="Greys_r")
plt.colorbar(label="HU")
plt.axis("off")
plt.show()

#combine all the transforms in 1 image
preproc_pipeline= Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="LPS")
])

#open pipeline
CT=preproc_pipeline(CTfolder)
CT_coronal= CT[0,:,256].cpu().numpy()

plt.figure(figsize=(3,8))
plt.pcolormesh(CT_coronal.T, cmap="Greys_r")
plt.colorbar(label="HU")
plt.axis("off")
plt.show()


####Part 2
modelpath=os.path.join(datadir, "wb_ctsegm", "models", "model_lowres.pt" )
configpath=os.path.join(datadir, "wb_ctsegm", "configs", "inference.json")
config=ConfigParser()
config.read_config(configpath)
#preprocessing pipeline
preprocessing=config.get_parsed_content("preprocessing")
data = preprocessing({'image': CTfolder})