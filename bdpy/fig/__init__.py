'''Figure package

This package is a part of BdPy.
'''

from .fig import *
from .tile_images import tile_images
from .draw_group_image_set import draw_group_image_set
from .makeplots import makeplots
try:
    #need to import cv2
    from .draw_group_movie_set import draw_group_movie_set, load_video, save_video
except:
    pass
