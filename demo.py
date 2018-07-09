import network
import os
from PIL import Image
import cv2
import numpy as np


if not(os.path.isfile("nvidia_monday")):
    import urllib.request
    print("Downloading weights, could take up to 5 minutes (400 MB)") 
    response = urllib.request.urlopen("https://data.kitware.com/api/v1/item/5b43b8528d777f2e622599eb/download")
    content = response.read()
    f = open("nvidia_monday", "wb")
    f.write(content)
    f.close()
    print("finished downloading")


network.model.load_weights("nvidia_monday")


while True:
    x = Image.open("target.png")

    #Cut off alpha channel
    z = np.array([np.array(x)[:,:,:3]])
    
    #Assume Green Pixels are the mask
    mask = 1 - np.repeat(np.expand_dims(np.all(z == np.array([[[[0, 255, 0]]]]), axis=-1), axis=-1), axis=-1, repeats=3)


    o = network.model.predict([z / 256., mask])

    cv2.imshow("Processed Image", cv2.resize(o[0][:,:,[2, 1, 0]], (512, 512)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
