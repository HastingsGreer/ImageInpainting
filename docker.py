import click
import json


def progress(pipe, **kwargs):
    if pipe:
        pipe.write(json.dumps(kwargs) + '\n')
        pipe.flush()


@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
@click.argument('mask', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path())
@click.option('--progress-pipe', type=click.File('w'),
              help='A named pipe where progress events should be written')
def main(image, mask, output, progress_pipe):
    progress(progress_pipe, message='Loading libraries', total=100, current=0)

    import network  # TensorFlow takes a couple seconds to load
    import numpy as np
    from PIL import Image

    progress(progress_pipe, message='Loading images', current=20)

    img = np.array(Image.open(image))
    mask = np.array(Image.open(mask))
    if mask.shape[:2] != img.shape[:2]:
        raise Exception('Image and mask size do not match (%dx%d) vs. (%dx%d).' % (
            img.shape[0], img.shape[1], mask.shape[0], mask.shape[1]))

    if img.shape[2] > 3:  # Cut off alpha channel if it exists
        img = img[:, :, :3]

    img = np.array([img])  # Network expects list of images, but we are just passing one
    patch_size = network.compute_patch_size_to_fit(img)

    # Convert mask to binary image by finding the white pixels (0=inpaint, 1=background)
    mask = np.all(mask != [255, 255, 255, 255], axis=-1).astype(np.uint8)
    # Network expects 3 channels in mask: [0, 0, 0] or [1, 1, 1] at each pixel
    mask = np.repeat(np.expand_dims(mask, axis=-1), axis=-1, repeats=3)
    mask = np.array([mask])

    # Pad the image and mask as needed
    padded_img, padded_mask = network.pad_to_patch_size(img, mask, patch_size)

    progress(progress_pipe, message='Configuring neural network', current=40)

    model = network.nvidia_unet(patch_size)

    progress(progress_pipe, message='Loading network weights', current=60)

    model.load_weights('/no_scaling_fix_overnight')

    progress(progress_pipe, message='Running prediction', current=80)

    o = model.predict([padded_img / 256., padded_mask])

    # Cut off the padding.
    o = o[:, :img.shape[1], :img.shape[2]]
    Image.fromarray((np.clip(o[0], 0, 1) * 255).astype(np.uint8)).save(output)


if __name__ == '__main__':
    main()
