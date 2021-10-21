import coco_text
import os
ct = coco_text.COCO_Text('../cocotext.v2.json')
ct.info()
imgs = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible'),('class','machine printed')])
anns = ct.getAnnIds(imgIds=ct.val, 
                        catIds=[('legibility','legible'),('class','machine printed')], 
                        areaRng=[0,200])
dataDir='../..'
dataType='train2014'

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# get all images containing at least one instance of legible text
imgIds = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible')])
# pick one at random
for imgId in imgIds:
    img = ct.loadImgs(imgId)[0]
    I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
    print('/images/%s/%s'%(dataType,img['file_name']))
    print(I.shape)
    plt.figure()
    # # plt.imshow(I)
    plt.imshow(I)
    annIds = ct.getAnnIds(imgIds=img['id'])
    anns = ct.loadAnns(annIds)
    ct.showAnns(anns,show_mask=True)
    # plt.show()
    plt.savefig(os.path.join("outputs", img['file_name']))