
Requirements:

numpy
keras
tensorflow
PIL
opencv

demo.py will download the pretrained weights, then continuously process "target.png" in the project directory, by masking out and then inpainting all pixels of the color (0, 255, 0), and displaying to a window.

Currently only 256x256 images are supported. Other sizes are the next planned feature

Workflow:

Open an image in your preferred editor.

Resize to 256x256

Save as target.png in the project directory

Execute demo.py with python3

Draw on image in bright green (0, 255, 0). 

Save image to run inpainting algorithm
