'''
Created on Sep 4, 2020

'''
import os
import random
import argparse
# from lib.utils import GRUI_train_dir

import torch

class ReadImputedData:
    def __init__(self, dataPath ):

        self.files = os.listdir(dataPath)
        self.dataPath=dataPath
        self.count=int(len(self.files)/3)
        
    def load(self):
        count=int(self.count)
        self.x=[]
#         self.y=[]
        self.delta=[]
        self.x_lengths=[]
        self.m=[]
        for i in range(1,count+1):
            file_x=open(os.path.join(self.dataPath,"batch"+str(i)+"x"))
            file_m=open(os.path.join(self.dataPath,"batch"+str(i)+"m"))
            file_delta=open(os.path.join(self.dataPath,"batch"+str(i)+"delta"))
            this_x,this_lengths=self.readx(file_x)
            this_m,_=self.readdelta(file_m)
            
            self.x.extend(this_x)
            self.x_lengths.extend(this_lengths)
#             self.y.extend(self.ready(file_y))
            this_delta,_=self.readdelta(file_delta)
            self.delta.extend(this_delta)
            self.m.extend(this_m)
            file_x.close()
#             file_y.close()
            file_delta.close()
        self.maxLength=len(self.x[0])
        
        
    def readx(self,x):
        this_x=[]
        this_lengths=[]
        count=1
        for line in x.readlines():
            if count==1:
                words=line.strip().split(",")
                for w in words:
                    if w=='':
                        continue
                    this_lengths.append(int(w))
            else:
                if "end" in line:
                    continue
                if "begin" in line:
                    d=[]
                    this_x.append(d)
                else:
                    words=line.strip().split(",")
                    oneclass=[]
                    for w in words:
                        if w=='':
                            continue
                        oneclass.append(float(w))
                    this_x[-1].append(oneclass)
            count+=1
        return this_x,this_lengths
    
    def ready(self,y):
        this_y=[]
        for line in y.readlines():
            d=[]
            words=line.strip().split(",")
            for w in words:
                if w=='':
                    continue
                d.append(int(w))
            this_y.append(d)
        return this_y
    
    def readdelta(self,delta):
        this_delta=[]
        this_m=[]
        for line in delta.readlines():
            if "end" in line:
                continue
            if "begin" in line:
                d=[]
                this_delta.append(d)
                t=[]
                this_m.append(t)
            else:
                words=line.strip().split(",")
                oneclass=[]
                onem=[]
                for i in range(len(words)):
                    w=words[i]
                    if w=='':
                        continue
                    oneclass.append(float(w))
                    if i==0 or float(w) >0:
                        onem.append(1.0)
                    else:
                        onem.append(0.0)
                this_delta[-1].append(oneclass)
                this_m[-1].append(onem)
        return this_delta,this_m
    
    def shuffle(self,batchSize=128,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            c = list(zip(self.x,self.y,self.m,self.delta,self.x_lengths))
            random.shuffle(c)
            self.x,self.y,self.m,self.delta,self.x_lengths=zip(*c)
            
    def nextBatch(self):
        i=1
        while i*self.batchSize<=len(self.x):
            x=[]
            y=[]
            m=[]
            delta=[]
            x_lengths=[]
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                x.append(self.x[j])
                y.append(self.y[j])
                m.append(self.m[j])
                delta.append(self.delta[j])
                x_lengths.append(self.x_lengths[j])
            i+=1
            yield  x,y,[0.0]*len(self.x[0][0]),m,delta,x_lengths,x,0,0,0
            
            
# if __name__ == '__main__':
# 
#     parser = argparse.ArgumentParser(description="parse args")
#     
#     dataset_name = 'mimic3_17'
#     
#     datapath = '../' + GRUI_train_dir + "/" + dataset_name
#     
#     print(datapath) 
#     
#     imputed_data = ReadImputedData(datapath)
#     
#     imputed_data.load()
#     
#     x_train = torch.tensor(imputed_data.x)
#     
#     mask_train = torch.tensor(imputed_data.m)
#     
#     delta_t_train = torch.tensor(imputed_data.delta)
#     
#     print(x_train.shape)
    
    
    
    