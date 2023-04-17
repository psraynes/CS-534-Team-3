import cv2
import numpy as np


###
# function to take in a path to a mask image, and return an array for the pixels.
# Key for returned array:
# 0 - Not corrosion, black in validation
# 1 - Fair, Red in validation (Maps to 38)
# 2 - Poor, Green in validation (Maps to 75)
# 3 - Severe, Yellow in validation (Maps to 113)

###
def load_mask(path, levels=False):
    picture = cv2.imread(path, 0)
    # If levels of corrosion are needed, map to 1,2,3, otherwise, map nonzero values to 1
    if levels:
        picture[picture == 38] = 1
        picture[picture == 75] = 2
        picture[picture == 113] = 3
    else:
        picture[picture != 0] = 1
    return picture
