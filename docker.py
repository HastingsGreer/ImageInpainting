import click
import network
from PIL import Image
import numpy as np

@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
#@click.argument('mask', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path())
def main(image, output):
    

    img = Image.open(image)

    # Cut off alpha channel
    img = np.array([np.array(img)[:, :, :3]])

    #patch size is a global variable in network.
    network.set_patch_size_to_fit(img)


    model = network.nvidia_unet()
    model.load_weights("/no_scaling_fix_overnight")


    # Assume Green Pixels are the mask
    # TODO pass mask separately to avoid in-band signal
    mask = 1 - np.repeat(np.expand_dims(np.all(img == np.array([[[[0, 255, 0]]]]), axis=-1), axis=-1),
                         axis=-1, repeats=3)
    
    #pad input and mask to the size the network expects.
    padded_img, padded_mask = network.pad_to_patch_size(img, mask)

    o = model.predict([padded_img / 256., padded_mask])

    #cut off the padding.
    o = o[:, :img.shape[1], :img.shape[2]]

    output_image = Image.fromarray((np.clip(o[0], 0, 1) * 256).astype(np.uint8))

    output_image.save(output)

    

#cv2.imshow("Processed Image", cv2.resize(o[0][:, :, [2, 1, 0]], (512, 512)))

#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break

if __name__ == '__main__':
    main()
