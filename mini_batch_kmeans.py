# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import np_cvt_coord_to_diagonal, np_intersection_over_union, np_cvt_coord_to_mid_point
#%%



class MiniBatchKMeans:
    
    def __init__(self,k,max_iteration,mini_batch_size):
        self.k = k
        self.max_iteration = max_iteration
        self.mini_batch_size = mini_batch_size
        
        #Initialize clusters
        self.cluster_vectors = np.random.uniform(size=(self.k,2))
        
    def train(self,data,iteration_hist=True):
        min = data.min()
        max = data.max()
        self.cluster_vectors = np.random.uniform(low=min,high=max,size=(self.k,2))
        if data.shape[0] < self.mini_batch_size:
            batch_size = data.shape[0]
        else:
            batch_size = self.mini_batch_size
        
        iteration = 0
        
        #Add origin as the center to all the cluster centroids
        b = np.zeros(shape=(self.k,2))
        clusters = np.concatenate((b,self.cluster_vectors),axis=1)
        
        while iteration < self.max_iteration:
            #randomly form minibatch of data
            idx = np.arange(0,data.shape[0])
            sample_idx = np.random.choice(idx,batch_size,replace=False)
            data_batch = data[sample_idx]
            
            #Add origin as the center to all the bounding boxes
            b = np.zeros(shape=(batch_size,2))
            data_batch = np.concatenate((b,data_batch),axis=1)
            
            #expand clusters dimensions to (k,n_data,4)
            #expand data_batch dimensions to (k,n_data,4)
            if iteration == 0:
                clusters = np.expand_dims(clusters,axis=1)
                clusters = np.tile(clusters,(1,batch_size,1))
                
            data_batch = np.expand_dims(data_batch,axis=0)
            data_batch = np.tile(data_batch,(self.k,1,1))
            
            #Convert clusters and data to corner coordinates
            clusters_diag = np_cvt_coord_to_diagonal(clusters)
            data_batch_diag = np_cvt_coord_to_diagonal(data_batch)
            #compute iou
            iou = np_intersection_over_union(clusters_diag,data_batch_diag)
            dist = 1.0 - iou
            min_dist_idx = np.argmin(dist,axis=0)
            
            avg_dist = 0.0
            avg_iou = 0.0
            for i in range(self.k):
                a = (i*np.ones(data_batch_diag.shape[1])).astype(int)
                b = np.argwhere(np.equal(a,min_dist_idx)==True)
                if b.shape[0] == 0:
                    continue
                #get boxes in center coordinate form
                boxes = data_batch[i,b,:]
                new_centroid = np.mean(boxes,axis=0)
                clusters[i,:,:] = new_centroid
                avg_dist += dist[i,b].sum()
                avg_iou += iou[i,b].sum()

            avg_dist = avg_dist / batch_size
            avg_iou = avg_iou / batch_size
            iteration+=1
            
            if iteration_hist:
                print('Iteration {}, AvgError: {}, AvgIou: {}'.format(iteration,
                  avg_dist,avg_iou))
            
        self.cluster_vectors = clusters[:,0,2:4]
        
        avg_dist, avg_iou = self.evaluate(data)
        print('Data has been fitted\n AvgError: {}, AvgIou: {}'.format(
                  avg_dist, avg_iou))
        return avg_dist,avg_iou
        
    def evaluate(self,data):
        batch_size = data.shape[0]
        
        b = np.zeros(shape=(self.k,2))
        clusters = np.concatenate((b,self.cluster_vectors),axis=1)
        
        b = np.zeros(shape=(batch_size,2))
        data_batch = np.concatenate((b,data),axis=1)
        
        clusters = np.expand_dims(clusters,axis=1)
        clusters = np.tile(clusters,(1,batch_size,1))
            
        data_batch = np.expand_dims(data_batch,axis=0)
        data_batch = np.tile(data_batch,(self.k,1,1))
        
        #Convert clusters and data to corner coordinates
        clusters_diag = np_cvt_coord_to_diagonal(clusters)
        data_batch_diag = np_cvt_coord_to_diagonal(data_batch)
        #compute iou
        iou = np_intersection_over_union(clusters_diag,data_batch_diag)
        dist = 1.0 - iou
        min_dist_idx = np.argmin(dist,axis=0)
        
        avg_dist = 0.0
        avg_iou = 0.0
        for i in range(self.k):
            a = (i*np.ones(data_batch_diag.shape[1])).astype(int)
            b = np.argwhere(np.equal(a,min_dist_idx)==True)
            #get boxes in center coordinate form
            boxes = data_batch[i,b,:]
            new_centroid = np.mean(boxes,axis=0)
            clusters[i,:,:] = new_centroid
            avg_dist += dist[i,b].sum()
            avg_iou += iou[i,b].sum()

        avg_dist = avg_dist / batch_size
        avg_iou = avg_iou / batch_size
        return avg_dist, avg_iou
    
    def fit(self,data):
        batch_size = data.shape[0]
        
        b = np.zeros(shape=(self.k,2))
        clusters = np.concatenate((b,self.cluster_vectors),axis=1)
        
        b = np.zeros(shape=(batch_size,2))
        data_batch = np.concatenate((b,data),axis=1)
        
        clusters = np.expand_dims(clusters,axis=1)
        clusters = np.tile(clusters,(1,batch_size,1))
            
        data_batch = np.expand_dims(data_batch,axis=0)
        data_batch = np.tile(data_batch,(self.k,1,1))
        
        #Convert clusters and data to corner coordinates
        clusters_diag = np_cvt_coord_to_diagonal(clusters)
        data_batch_diag = np_cvt_coord_to_diagonal(data_batch)
        #compute iou
        iou = np_intersection_over_union(clusters_diag,data_batch_diag)
        dist = 1.0 - iou
        min_dist_idx = np.argmin(dist,axis=0)
        

        return min_dist_idx   