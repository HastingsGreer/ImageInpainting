
# Requirements:

numpy
keras
tensorflow
PIL
opencv

demo.py will download the pretrained weights, then continuously process "target.png" in the project directory, by masking out and then inpainting all pixels of the color (0, 255, 0), and displaying to a window.

# Workflow:

Open an image in your preferred editor.

Save as target.png in the project directory

Execute demo.py with python3

Draw on image in bright green (0, 255, 0). 

Save image to run inpainting algorithm

# Docker Workflow (CPU only):

Run `demo.py` once to download network weights.

Build dockerfile with command `sudo docker build -t inpainting .`

Process an image with command 

`sudo docker run -v ~/full_path_to_image_dir/:/img/ inpainting /img/target.png /img/out.png`

