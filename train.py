# coding: utf-8

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import json
import matplotlib.pylab as plt
from glob import glob
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
import segmentation_models as sm
from segmentation_models import Linknet
from keras import optimizers, callbacks
from losses import dice_coef_multiclass_loss

# In[]: Parameters
verbose = 1
visualize = True
num_classes = 4
input_shape = (704, 512, 3)
backbone = 'resnet18'
batch_size = 1

# In[]: Dataset
dataset_dir = "./data/"
subdirs = ["ds"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("DATASETS USED: {}".format(subdirs))
print("TOTAL FILES COUNT: {}\n".format(len(ann_files)))

# In[]:
def get_image(path, label = False):
    img = Image.open(path)
    img = img.resize(input_shape[:2][::-1])
    img = np.array(img)[...,:3]
    if label:
        return img[...,0]
    return img
    
img_path = ann_files[0].replace('/ann/', '/img/').split('.json')[0]
label_path = ann_files[0].replace('/ann/', '/masks_machine/').replace('jpg','png').split('.json')[0]

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Labels dtype: {}\n".format(get_image(label_path, label = True).dtype))
print("Images shape: {}".format(get_image(img_path).shape))
print("Labels shape: {}\n".format(get_image(label_path, label = True).shape))

# In[]: Visualise
if visualize:
    i = 6
    img_path = ann_files[i].replace('/ann/', '/img/').split('.json')[0]
    label_path = ann_files[i].replace('/ann/', '/masks_machine/').replace('jpg','png').split('.json')[0]
    x = get_image(img_path)
    y = get_image(label_path, label = True)
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y)
    fig.tight_layout()
        
# In[]: Class weight counting
def cw_count(ann_files):
    print("Class weight calculation started")
    cw_seg = np.zeros(num_classes, dtype=np.int64)

    for af in tqdm(ann_files):
        label_path = af.replace('/ann/', '/masks_machine/').replace('jpg','png').split('.json')[0]
        l = get_image(label_path, label = True)
        
        for i in range(num_classes):
            cw_seg[i] += np.count_nonzero(l==i)
        
    if sum(cw_seg) == len(ann_files)*input_shape[0]*input_shape[1]:
        print("Class weights calculated successfully:")
        class_weights = np.median(cw_seg/sum(cw_seg))/(cw_seg/sum(cw_seg))
        for cntr,i in enumerate(class_weights):
            print("Class {} = {}".format(cntr, i))
    else:
        print("Class weights calculation failed")
        
    return class_weights
        
class_weights = cw_count(ann_files)
    
# In[]: 
def train_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0])
            y = get_image(ann_files[i].replace('/ann/', '/masks_machine/').replace('jpg','png').split('.json')[0], label=True)
            
            x_batch[b] = x
            y_batch[b] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
            
        yield (x_batch, y_batch)
    
# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

train_gen = train_generator(files = ann_files, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

# In[]: Bottleneck
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
    
print("\nModel summary:")
model.summary()

# In[]: 
loss = [dice_coef_multiclass_loss]
metrics = ['categorical_accuracy']

learning_rate = 1e-04
optimizer = optimizers.Adam(lr = learning_rate)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("\nOptimizer: {}\nLearning rate: {}\nLoss: {}\nMetrics: {}\n".format(optimizer, learning_rate, loss, metrics))

# In[]:
reduce_lr = callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)
model_checkpoint = callbacks.ModelCheckpoint('weights/clothes.hdf5', monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)

clbacks = [reduce_lr, early_stopper, model_checkpoint]

print("Callbacks used:")
for c in clbacks:
    print("{}".format(c))

# In[]: 
steps_per_epoch = len(ann_files)//batch_size
epochs = 1000

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        verbose = verbose,
        callbacks = clbacks,
        class_weight = class_weights
)
print("Finished training\n")