# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import segmentation_models as sm
from segmentation_models import Linknet
import cv2
import imutils

# In[]: Parameters
num_classes = 4
input_shape = (704, 512, 3)
backbone = 'resnet18'
CLASSES = ['Background', 'T-Shirt', 'Dress', 'Pants']
    
# In[]:
def get_image(path):
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

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=4)
                
    st = stats[:,-1][stats[:,-1] > area_threshold] #Ban areas smaller than threshold
                    
    if nb_components == 1 or len(st) < 2:
        return None, None, None

    if (num_blobs <= len(st)-1):
        n = num_blobs+1
    else:
        n = len(st)
    
    blob_index = np.argsort(stats[:,-1])[-n:-1]
                
    return output, blob_index[::-1], centroids[blob_index[::-1]]
            
# In[]:
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
model.load_weights('weights/clothes.hdf5')

# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

img_path = "2.jpg"
x = get_image(img_path)
y_pred = np.squeeze(model.predict(np.expand_dims(preprocessing_fn(x), axis=0)))

#y_tshirt = y_pred[0,...,1] > conf
#y_dress = y_pred[0,...,2] > conf
#y_pants = y_pred[0,...,3] > conf
#plt.imshow(y_tshirt)

# In[]: Find contours
conf = 0.9

items = []

for i in range(1,y_pred.shape[-1]):
#    print(CLASSES[i])
    mask = y_pred[...,i] > conf
    output, blob_indexes, centroids = extract_largest_blobs(mask, 5000, num_blobs=2)
    if blob_indexes is not None:
        item = {}
        item['Class'] = CLASSES[i]
        for bi in blob_indexes:
             m = output == bi
             cnts = cv2.findContours(m.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             cnts = imutils.grab_contours(cnts)
             c = np.squeeze(max(cnts, key=cv2.contourArea))
             item['Points'] = c
#             print(c.shape)
             items.append(item)

# In[]: 





# In[]: 





# In[]: 





# In[]: 





