import click
import network
from PIL import Image
import numpy as np


@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
@click.argument('mask', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path())
def main(image, mask, output):
    img = np.array(Image.open(image))

    mask = np.array(Image.open(mask))
    if mask.shape[:2] != img.shape[:2]:
        raise Exception('Image and mask size do not match (%dx%d) vs. (%dx%d).' % (
            img.shape[0], img.shape[1], mask.shape[0], mask.shape[1]))

    if img.shape[2] > 3:  # Cut off alpha channel if it exists
        img = img[:, :, :3]
    patch_size = network.compute_patch_size_to_fit(img)

    # Convert mask to binary image by finding the white pixels (1=inpaint, 0=background)
    mask = np.all(mask == [255, 255, 255, 255], axis=-1).astype(np.uint8)
    padded_img, padded_mask = network.pad_to_patch_size(img, mask, patch_size)

    model = network.nvidia_unet(patch_size)
    model.load_weights('/no_scaling_fix_overnight')
    o = model.predict([padded_img / 256., padded_mask])

    # Cut off the padding.
    o = o[:, :img.shape[1], :img.shape[2]]
    Image.fromarray((np.clip(o[0], 0, 1) * 256).astype(np.uint8)).save(output)


if __name__ == '__main__':
    main()
