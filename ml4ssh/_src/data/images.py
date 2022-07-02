import  imageio

def load_fox():
    # FOX
    image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
    return img

def load_earth():
    # EARTH
    image_url = "https://i0.wp.com/thepythoncodingbook.com/wp-content/uploads/2021/08/Earth.png?w=301&ssl=1"
    img = imageio.imread(image_url)[..., :3] / 255.
    # TODO: crop the image slightly
    return img

import skimage
def load_cameraman():
    img = skimage.data.camera() / 255.
    return img[..., None]
