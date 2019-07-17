import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

def np_cvt_coord_to_mid_point(coordinates):
    """
    Converts bounding boxes from corner coordinates to
    mid point coordinates
    
    Args:
        coordinates: A tensor of shape (N,B,4)
    Returns:
        Tensor of shape (N,B,4)
    """
    #Get mid points xc and yc
    # xc = (x1 + x2) / 2
    xc = (coordinates[:,:,0] + coordinates[:,:,2]) / 2.0
    # yc = (y1 + y2) / 2
    yc = (coordinates[:,:,1] + coordinates[:,:,3]) / 2.0
    #Get dimensions of the box
    # w = |x1 - x2|
    # h = |y1 - y2|
    w = np.abs(coordinates[:,:,0] - coordinates[:,:,2])
    h = np.abs(coordinates[:,:,1] - coordinates[:,:,3])
    
    #Add 2nd dimension to xc, yc, w, h tensor
    xc = np.expand_dims(xc,axis=2)
    yc = np.expand_dims(yc,axis=2)
    w = np.expand_dims(w,axis=2)
    h = np.expand_dims(h,axis=2)
    #Return concatenated tensor (xc,yc,w,h)
    return np.concatenate((xc,yc,w,h),axis=2)


def np_cvt_coord_to_diagonal(coordinates):
    """
    Converts bounding boxes from mid point coordinates to
    corner coordinates
    
    Args:
        coordinates: A tensor of shape (N,B,4)
    Returns:
        Tensor of shape (N,B,4)
    """
    # xmin = xc - w/2
    #xmax = xc + w/2
    #ymin = yc - w/2
    #ymax = yc + w/2
    #add 2nd dimensions to all above tensors
    #return concatenated tensor of (xmin,ymin,xmax,ymax)

    xmin = coordinates[:,:,0] - coordinates[:,:,2]/2.0
    xmax = coordinates[:,:,0] + coordinates[:,:,2]/2.0
    ymin = coordinates[:,:,1] - coordinates[:,:,3]/2.0
    ymax = coordinates[:,:,1] + coordinates[:,:,3]/2.0
    xmin = np.expand_dims(xmin,axis=2)
    xmax = np.expand_dims(xmax,axis=2)
    ymin = np.expand_dims(ymin,axis=2)
    ymax = np.expand_dims(ymax,axis=2)
    return np.concatenate((xmin,ymin,xmax,ymax),axis=2)


def np_intersection_over_union(boxA, boxB):
    """
    Finds the intersection over union between boxes boxA and boxB.
    Coordinates must be in corner coordinate form. 
    
    Args:
        boxA: A tensor of shape (N,B,4)
        boxB: A tensor of shape (N,B,4)
    Returns:
        Tensor of shape (N, B) containing IOU values. 
    """
    #Find coordinates of region of intersection (xA,yA, xB,yB)
    #xA = max(boxA[xmin], boxB[xmin])

    #yA = max(boxA[ymin], boxB[ymin])

    #xB = max(boxA[xmax], boxB[xmax])

    #yB = max(boxA[ymax], boxB[ymax])

    #find area of intersection. If there is not intersection then xB - xA or
    #yB - yA is -ve. Hence interArea will be zero

    #Find area of box A = (x2 - x1)*(y2 - y1)

    #Find area of box B = (x2 - x1)*(y2 - y1)

    #Find iou = areaOfIntersection / (areOfBoxA + areaOfBox - areaOfIntersection)

    xA = np.maximum(boxA[:,:,0], boxB[:,:,0])
    yA = np.maximum(boxA[:,:,1], boxB[:,:,1])
    xB = np.minimum(boxA[:,:,2], boxB[:,:,2])
    yB = np.minimum(boxA[:,:,3], boxB[:,:,3])
    interArea = np.maximum(0, xB - xA ) * np.maximum(0, yB - yA )
    boxAArea = (boxA[:,:,2] - boxA[:,:,0] ) * (boxA[:,:,3] - boxA[:,:,1] )
    boxBArea = (boxB[:,:,2] - boxB[:,:,0]) * (boxB[:,:,3] - boxB[:,:,1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

#%%
def read_annotation_file(file_path):
    """
    Read pascal VOC xml file
    Args:
        file_path: Pascal VOC xml file path
    Returns:
        panda frame containing class name and bounding boxes and original 
        height and width of image
        ['label','xmin','ymin','xmax','ymax', 'img_width','img_height']
    """
    xml = ET.parse(file_path)
    root = xml.getroot()
    data = []
    height = int(root.find('size').find('height').text,10)
    width = int(root.find('size').find('width').text,10)
    for object in root.findall('object'):
        name = object.find('name').text
        xmin = int(object.find('bndbox').find('xmin').text,10)
        ymin = int(object.find('bndbox').find('ymin').text,10)
        xmax = int(object.find('bndbox').find('xmax').text,10)
        ymax = int(object.find('bndbox').find('ymax').text,10)
        data.append([name,xmin,ymin,xmax,ymax,width,height])
    df = pd.DataFrame(data=data,columns=['label','xmin','ymin','xmax','ymax',
                                         'img_width','img_height'])
    return df