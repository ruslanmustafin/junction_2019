import cv2
import imutils
import numpy as np



from PIL import Image

CLASSES = ['Background', 'T-Shirt', 'Dress', 'Pants']
    
# In[]:
def get_image(path, input_shape):
    img = Image.open(path)
    img = img.resize(input_shape[:2][::-1])
    img = np.array(img)[...,:3]
    return img

def fill_holes(mask):
    mask_floodfill = mask.astype('uint8').copy()
    h, w = mask.shape[:2]
    cv2.floodFill(mask_floodfill, np.zeros((h+2, w+2), np.uint8), (0,0), 255)

    out = mask | cv2.bitwise_not(mask_floodfill)
    
    return out.astype(np.bool)

def extract_largest_blobs(mask, area_threshold, num_blobs=1):

    (nb_components, 
     output, 
     stats, 
     centroids) = cv2.connectedComponentsWithStats(mask.astype('uint8'), 
                                                   connectivity=4)
                
    st = stats[:,-1][stats[:,-1] > area_threshold] #Ban areas smaller than threshold
                    
    if nb_components == 1 or len(st) < 2:
        return None, None, None

    if (num_blobs <= len(st)-1):
        n = num_blobs+1
    else:
        n = len(st)
    
    blob_index = np.argsort(stats[:,-1])[-n:-1]
                
    return output, blob_index[::-1], centroids[blob_index[::-1]]