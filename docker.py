import click
import network
from PIL import Image
import numpy as np

@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
#@click.argument('mask', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path())
def main(image, output):
    model = network.nvidia_unet()
    model.load_weights("/nvidia_monday")

    img = Image.open(image)

    # Cut off alpha channel
    img = np.array([np.array(img)[:, :, :3]])

    # Assume Green Pixels are the mask
    # TODO pass mask separately to avoid in-band signal
    mask = 1 - np.repeat(np.expand_dims(np.all(img == np.array([[[[0, 255, 0]]]]), axis=-1), axis=-1),
                         axis=-1, repeats=3)

    o = model.predict([img / 256., mask])
    print(o)

#cv2.imshow("Processed Image", cv2.resize(o[0][:, :, [2, 1, 0]], (512, 512)))

#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break

if __name__ == '__main__':
    main()
