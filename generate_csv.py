# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:36:18 2019
This is the script to generate csv file of bounding boxes for train and test data
@author: siddh
"""

#%%
import os
import csv
import numpy as np
import sys
import argparse
from util import read_annotation_file, np_cvt_coord_to_mid_point
#%%

parser = argparse.ArgumentParser(
    description='Utility to generate CSV files containing bounding boxes from PASCAL VOC annotation files')

parser.add_argument('data_path',nargs=1,type=str,metavar='data_path',
                   help= 'Path to directory containing PASCAL VOC xml files')

parser.add_argument('save_path',nargs=1,type=str, metavar='save_path',
                   help='Output CSV file path')
args = parser.parse_args()

DATA_PATH = args.data_path[0]
CSV_NAME = args.save_path[0]

print(DATA_PATH)
print(CSV_NAME)


#%%
#Make train csv file
print("Writing csv file..........")

file_list = os.listdir(path=DATA_PATH)
xml_file_list = []
for i in file_list:
    if i.endswith('.xml'):
        xml_file_list.append(i)

if len(xml_file_list) == 0:
    raise Exception('No XML annotations found')

#%%
with open(CSV_NAME,'w') as file:
    column_names = ['file_path','label','xmin','ymin','xmax','ymax','xc','yc','w','h']
    
    writer = csv.DictWriter(file,fieldnames=column_names)
    writer.writeheader()
    for i in range(len(xml_file_list)):
        xml_path = os.path.join(DATA_PATH,xml_file_list[i])
        annotation = read_annotation_file(xml_path)
        image_w = annotation.iloc[0,5]
        image_h = annotation.iloc[0,6]
        
        annotation[['xmin','xmax']] /= image_w 
        annotation[['ymin','ymax']] /= image_h
        
        diag_boxes = np.array(annotation[['xmin','ymin','xmax','ymax']]).reshape(1,-1,4)
        mid_boxes = np_cvt_coord_to_mid_point(diag_boxes).reshape(-1,4)
        
        row = {}
        for j in range(annotation.shape[0]):
            row['file_path'] = xml_path
            row['label'] = annotation.iloc[j,0]
            row['xmin'] = annotation.iloc[j,1]
            row['ymin'] = annotation.iloc[j,2]
            row['xmax'] = annotation.iloc[j,3]
            row['ymax'] = annotation.iloc[j,4]
            row['xc'] = mid_boxes[j,0]
            row['yc'] =  mid_boxes[j,1]
            row['w'] = mid_boxes[j,2]
            row['h'] =  mid_boxes[j,3]
            writer.writerow(row)
        print('Processed ',i+1,'out of ',len(xml_file_list),' files')
print('Finished writing csv')  

        
        
    


