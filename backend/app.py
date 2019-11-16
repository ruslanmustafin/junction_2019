import glob
import os
import json

import cv2
import cloudinary
import imutils

import segmentation_models as sm
import numpy as np

from cloudinary import uploader

from flask import Flask
from flask import jsonify
from flask import request

from segmentation_models import Linknet

import aliseeksapi

from utils import get_image, fill_holes, extract_largest_blobs
from utils import CLASSES


cloudinary.config(
  cloud_name = 'rsmustafin',  
  api_key = '687986331479948',  
  api_secret = 'Lo3tm7cJqga-C3JG3_hJr-yMFg4'  
)

configuration = aliseeksapi.Configuration()
configuration.api_key['X-API-CLIENT-ID'] = "AGGXIQITSUSZVGLY"

version_prefix = '1'
url_prefix = f'/api/v{version_prefix}'

input_shape = (704, 512, 3)

def load_database():

    database = {}

    files = glob.glob('./mock_db/*.json')
    for f in files:
        f_name = os.path.basename(f)[:-5]

        with open(f) as fp:
            json_dict = json.load(fp)

        database[f_name] = [{'type':obj['classTitle'], 
                             'points':obj['points']['exterior']}
                            
                            for obj in json_dict['objects']]


    return database

def load_model(path, 
               input_shape, 
               num_classes=4,
               backbone='resnet18'):
    
    model = Linknet(backbone_name=backbone, 
                    input_shape=input_shape, 
                    classes=num_classes, 
                    activation='softmax')
        
    model.load_weights(path)
    
    return model, sm.get_preprocessing(backbone)


def crop(img, points):
    """ 
    
    Arguments:
        img (PIL.Image): image
        points (numpy.array): array of points

    """
    
    # get rect cooreds
    xmin, xmax = points[:,0].min(), points[:,0].max()
    ymin, ymax = points[:,1].min(), points[:,1].max()
    
    crop = img[ymin:ymax, xmin:xmax]
    
    return crop

cache = {}

api_instance = aliseeksapi.SearchApi(aliseeksapi.ApiClient(configuration))

database = load_database()    
(model, 
 preprocessing_fn) = load_model('weights/clothes.hdf5', input_shape)

app = Flask(__name__)

@app.route(f'{url_prefix}/scan')
def scan_img(conf=0.9):
    img_id = request.args.get('img')
    print(img_id)
    
    cid = 0
    
    if img_id in database:
        
        x = get_image(os.path.join('imgs', img_id), input_shape)
        
        y_pred = model.predict(preprocessing_fn(x)[None]).squeeze()
        
        items = []
        
        for i in range(1, y_pred.shape[-1]):
            
            mask = y_pred[...,i] > conf
            (output, 
             blob_indexes, 
             centroids) = extract_largest_blobs(mask, 
                                                5000, 
                                                num_blobs=2)
            
            if blob_indexes is not None:
                
                item = {}
                item['type'] = CLASSES[i]
                
                for bi in blob_indexes:
                    m = output == bi
                     
                    cnts = cv2.findContours(m.astype(np.uint8).copy(), 
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                    
                    cnts = imutils.grab_contours(cnts)
                    c = np.squeeze(max(cnts, key=cv2.contourArea))
                    
                    item['points'] = [i.tolist() for i in c]
                    
                    # TODO:
                    # send a request via aliseeks here
                    
                    cropped = crop(x, c)
                    
                    cv2.imwrite(f'./crops/tmp.jpg', cropped[:,:,::-1])
                    result = uploader.upload('./crops/tmp.jpg')       
                    # print(uuid)
                    
                    url = result['url']
                    
                    req = aliseeksapi.UploadImageByUrlRequest(url)
                    resp = api_instance.upload_image_by_url(req)

                    search_req = \
                        aliseeksapi.ImageSearchRequest(resp.upload_key)    
                    resp = api_instance.search_by_image(search_req)
                    
                    
                    
                    
                    item['suggestions'] = [i.to_dict() for i in resp.items]
                    
                    items.append(item)
        
        return jsonify({'items':items})
    else:
        return jsonify({})
    
    
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')


