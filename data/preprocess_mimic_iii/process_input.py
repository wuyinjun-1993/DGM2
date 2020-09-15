'''
Created on Jul 22, 2020

'''

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

import numpy as np

file_path="~/pg_data/physionet.org/files/mimiciii/1.4/"

adm_3 = pd.read_csv(file_path+"Admissions_processed.csv")

inputs=pd.read_csv(file_path+"INPUTEVENTS_MV.csv")
#Restrict the dataset to the previously selected admission ids only. join admissions_processed with input
adm_ids=list(adm_3["HADM_ID"])
inputs=inputs.loc[inputs["HADM_ID"].isin(adm_ids)]

#Inputs_small only contains the columns of interest.
inputs_small=inputs[["SUBJECT_ID","HADM_ID","STARTTIME","ENDTIME","ITEMID","AMOUNT","AMOUNTUOM","RATE","RATEUOM","PATIENTWEIGHT","ORDERCATEGORYDESCRIPTION"]]
print(inputs_small.head())

print("Number of patients remaining in the database: ")
print(inputs_small["SUBJECT_ID"].nunique())




#item_id 
item_id=pd.read_csv(file_path+"D_ITEMS.csv")
item_id_1=item_id[["ITEMID","LABEL"]]
print(item_id_1.head())

#We merge the name of the item administrated.
inputs_small_2=pd.merge(inputs_small,item_id_1,on="ITEMID")
print(inputs_small_2.head())
print("Number of patients remaining in the database: ")
print(inputs_small_2["SUBJECT_ID"].nunique())

#For each item, evaluate the number of patients who have been given this item.
pat_for_item=inputs_small_2.groupby("LABEL")["SUBJECT_ID"].nunique()
#Order by occurence and take the 33 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:50]

#Select only the time series with high occurence.
inputs_small_3=inputs_small_2.loc[inputs_small_2["LABEL"].isin(list(frequent_labels.index))].copy()

print("Number of patients remaining in the database: ")
print(inputs_small_3["SUBJECT_ID"].nunique())















