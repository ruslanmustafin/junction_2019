# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
from keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import segmentation_models as sm
from segmentation_models import Linknet
from losses import dice_coef_multiclass_loss
from keras import optimizers
import cv2

# In[]: Parameters
num_classes = 4
input_shape = (704, 512, 3)
backbone = 'resnet18'
    
# In[]:
def get_image(path):
    img = Image.open(path)
    img = img.resize(input_shape[:2][::-1])
    img = np.array(img)[...,:3]
    return img
            
# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

# In[]: Bottleneck
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
model.load_weights('weights/clothes.hdf5')

# In[]:
img_path = "1.jpg"
x = get_image(img_path)
y_pred = model.predict(np.expand_dims(preprocessing_fn(x), axis=0))
#y_pred = np.squeeze(np.argmax(y_pred, axis=-1)).astype('int64')

conf = 0.9
y_tshirt = y_pred[0,...,1] > conf
y_dress = y_pred[0,...,2] > conf
y_pants = y_pred[0,...,3] > conf
plt.imshow(y_tshirt)
