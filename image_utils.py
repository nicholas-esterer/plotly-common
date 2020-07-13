import plotly.express as px
import numpy as np
import itertools

def fromhex(n):
    return int(n, base=16)


def label_to_colors(
    img, colormap=px.colors.qualitative.Light24, alpha=128, color_class_offset=0
):
    """
    Take a tensor containing integers representing labels and return an dim0x...dim(D-1)x4
    matrix where each label has been replaced by a color looked up in colormap
    and D is the number of dimensions in the original tensor.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel. If a list, it is cycled and zipped
    with the colormap to give that alpha channel for each corresponding color.
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """
    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]
    if type(alpha) is not type(list()):
        alpha=[alpha]
    cm_alpha=list(zip(colormap,itertools.cycle(alpha)))
    cimg = np.zeros(img.shape + (3,), dtype="uint8")
    alpha = np.zeros(img.shape + (1,), dtype="uint8")
    for c in set(img.flatten()):
        cimg[img == c],alpha[img == c] = cm_alpha[(c + color_class_offset) % len(colormap)]
    return np.concatenate(
        (cimg, alpha), axis=len(cimg.shape)-1
    )

def combine_last_dim(
    img,
    output_n_dims=3,
    combiner=lambda x: (np.sum(np.abs(x),axis=-1)!=0).astype('float')
):
    if len(img.shape)==output_n_dims:
        return img
    imgout=combiner(img)
    return imgout
