import network
import os
from PIL import Image
import cv2
import numpy as np


if not(os.path.isfile("no_scaling_fix_overnight")):
    import urllib.request
    print("Downloading weights, could take up to 5 minutes (400 MB)") 
    response = urllib.request.urlopen("https://data.kitware.com/api/v1/item/5b55f4948d777f06857bfd21/download")
    content = response.read()
    f = open("no_scaling_fix_overnight", "wb")
    f.write(content)
    f.close()
    print("finished downloading")

x = Image.open("target.png")

#Cut off alpha channel
z = np.array([np.array(x)[:,:,:3]])

network.set_patch_size_to_fit(z)

model = network.nvidia_unet()

model.load_weights("no_scaling_fix_overnight")

while True:
    x = Image.open("target.png")

    #Cut off alpha channel
    z = np.array([np.array(x)[:,:,:3]])
    #Assume Green Pixels are the mask
    mask = 1 - np.repeat(np.expand_dims(np.all(z == np.array([[[[0, 255, 0]]]]), axis=-1), axis=-1), axis=-1, repeats=3)
    
    network_input, network_mask = network.pad_to_patch_size(z, mask)
    o = model.predict([network_input / 256., network_mask])
    #cut off padding
    o = o[:, :z.shape[1], :z.shape[2]]




    cv2.imshow("Processed Image", o[0][:,:,[2, 1, 0]])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
