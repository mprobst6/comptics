
import os
# import cv2 as cv
import numpy as np
import importlib.util as imp
from matplotlib import pyplot as plt


def make_random_structures(
    save_name: str = 'test',
    resolution: int = 30,
    shapesize: list = [3,3],
    feature_sizes: list = [1,2,3], # recommended 1-10
    num_copies: int = 5
):

    x = int(resolution*shapesize[0])
    y = int(resolution*shapesize[1])
    
    middle = int(x/2)
    center = int(y/2)
    
    x_g = np.linspace(0,x,x)
    y_g = np.linspace(0,y,y)
    X_g,Y_g = np.meshgrid(x_g,y_g,sparse=True, indexing='ij')

    position_y = 0
    final_dict = {}
    for feature_size in feature_sizes:
        
        for copy in range(num_copies):
            # make a mask which determines the feature size
            mask = (
                (X_g-middle)**2+(Y_g-center)**2 >= feature_size**2
                )
            # show(mask)
            
            # generate random numbers in the shape of the mask
            filter_coeffs = (np.random.rand(x,y)-0.5)*x*y+1j*(np.random.rand(x,y)-0.5)*x*y
            filter_coeffs[mask] = 0
            # show(np.real(filter_coeffs))

            # ifft generates random shapes
            shape = np.abs(np.fft.ifft2(filter_coeffs))
            # show(shape)
            
            # scale, normalize, and binarize the shapes
            scale = 0.5/np.median(shape) # scale so the median is at 0.5
            normalized_shape = np.clip(shape*scale,0,1)
            binarized_shape = normalized_shape > 0.5
            # show(binarized_shape)
            
            
            bordered_shape = make_border_zero(binarized_shape, feature_size, resolution)
            # show(bordered_shape)
            
            final_dict[f'shapesize-{shapesize}_size-{feature_size}_copy-{copy}'] = bordered_shape
        
        # save examples
        save(bordered_shape, name=f'examples/shapesize-{shapesize}_size-{feature_size}')


    # save a different npz for each geometry
    to_npz(
        filename = save_name,
        array_dict = final_dict,
    )

def make_border_zero(array: np.array, min_features, resolution) -> np.array:
    height, width = np.shape(array)
    border_thickness = int(0.25 * resolution) # NOTE: can be changed
    
    array[:border_thickness] = 0
    array[height-border_thickness:] = 0
    array[:,:border_thickness] = 0
    array[:,width-border_thickness:] = 0
    return array


def to_npz(filename: str, array_dict: dict):
    os.makedirs(f'data/random_shapes',exist_ok=True)
    np.savez(f'data/random_shapes/{filename}.npz',**array_dict)
    
def show(image: np.array) -> None:
    fig = plt.figure()
    plt.imshow(image); plt.colorbar()
    plt.show()
    plt.close(fig)
    
def save(image: np.array, name: str) -> None:
    plt.figure()
    plt.imshow(image); plt.colorbar()
    plt.savefig(f'{name}.png')
    
if __name__ == '__main__':
    make_random_structures(
        resolution=100,
        shapesize = [2,2],
        feature_sizes = [1,2,3,4],
        num_copies = 2
    )