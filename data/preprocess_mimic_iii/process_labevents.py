'''
Created on Jul 22, 2020

'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import datetime
from datetime import timedelta

file_path="~/pg_data/physionet.org/files/mimiciii/1.4/"
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 300)

adm=pd.read_csv(file_path+"Admissions_processed.csv")

lab=pd.read_csv(file_path+"LABEVENTS.csv")

#Restrict the dataset to the previously selected admission ids only.
adm_ids=list(adm["HADM_ID"])
lab=lab.loc[lab["HADM_ID"].isin(adm_ids)]

print("Number of patients remaining in the database: ")
print(lab["SUBJECT_ID"].nunique())


#item_id
item_id=pd.read_csv(file_path+"D_LABITEMS.csv")
item_id_1=item_id[["ITEMID","LABEL"]]
item_id_1.head()

#We merge the name of the item administrated.
lab2=pd.merge(lab,item_id_1,on="ITEMID")
lab2.head()
print("Number of patients remaining in the database: ")
print(lab2["SUBJECT_ID"].nunique())


n_best=150
#For each item, evaluate the number of patients who have been given this item.
pat_for_item=lab2.groupby("LABEL")["SUBJECT_ID"].nunique()
#Order by occurence and take the 20 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]

#Select only the time series with high occurence.
lab3=lab2.loc[lab2["LABEL"].isin(list(frequent_labels.index))].copy()

print("Number of patients remaining in the database: ")
print(lab3["SUBJECT_ID"].nunique())


#Verification that all input labels have the same amounts units.
print(lab3.groupby("LABEL")["VALUEUOM"].value_counts())

#Correct the units
lab3.loc[lab3["LABEL"]=="Calculated Total CO2","VALUEUOM"]="mEq/L"
lab3.loc[lab3["LABEL"]=="PT","VALUEUOM"]="sec"
lab3.loc[lab3["LABEL"]=="pCO2","VALUEUOM"]="mm Hg"
lab3.loc[lab3["LABEL"]=="pH","VALUEUOM"]="units"
lab3.loc[lab3["LABEL"]=="pO2","VALUEUOM"]="mm Hg"

#Only select the subset that was used in the paper (only missing is INR(PT))
subset=["Albumin","Alanine Aminotransferase (ALT)","Alkaline Phosphatase","Anion Gap","Asparate Aminotransferase (AST)","Base Excess","Basophils","Bicarbonate","Bilirubin, Total","Calcium, Total","Calculated Total CO2","Chloride","Creatinine","Eosinophils","Glucose","Hematocrit","Hemoglobin",
"Lactate","Lymphocytes","MCH","MCHC","MCV","Magnesium","Monocytes","Neutrophils","PT","PTT","Phosphate","Platelet Count","Potassium","RDW","Red Blood Cells","Sodium","Specific Gravity","Urea Nitrogen","White Blood Cells","pCO2","pH","pO2"]

lab3=lab3.loc[lab3["LABEL"].isin(subset)].copy()



lab3.groupby("LABEL")["VALUENUM"].describe()




#Glucose : mettre -1 aux résultats négatifs et supprimer les autres entrées dont la valeur numérique est NaN.
lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"].isnull())&(lab3["VALUE"]=="NEG"),"VALUENUM"]=-1
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"].isnull())].index).copy()

#Retirer les entrées avec NaN aux values et valuenum
lab3=lab3.drop(lab3.loc[(lab3["VALUENUM"].isnull())&(lab3["VALUE"].isnull())].index).copy()

#Remove the remaining NAN Values
lab3=lab3.drop(lab3.loc[(lab3["VALUENUM"].isnull())].index).copy()

#Remove anion gaps lower than 0
lab3=lab3.drop(lab3.loc[(lab3["VALUENUM"]<0)&(lab3["LABEL"]=="Anion Gap")].index).copy()

#Remove BE <-50
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Base Excess")&(lab3["VALUENUM"]<-50)].index).copy()
#Remove BE >50
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Base Excess")&(lab3["VALUENUM"]>50)].index).copy()

#Remove high Hemoglobins
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Hemoglobin")&(lab3["VALUENUM"]>25)].index).copy()

#Clean some glucose entries
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"]>2000)&(lab3["HADM_ID"]==103500.0)].index).copy()
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"]>2000)&(lab3["HADM_ID"]==117066.0)].index).copy()

#Clean toO high levels of Potassium
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Potassium")&(lab3["VALUENUM"]>30)].index).copy()

lab3.to_csv(file_path+"LAB_processed.csv")

#Merge the admission time.
adm_short=adm[["HADM_ID","ADMITTIME","ELAPSED_TIME","DEATHTAG"]]
lab4=pd.merge(lab3,adm_short,on="HADM_ID")
lab4['CHARTTIME']=pd.to_datetime(lab4["CHARTTIME"], format='%Y-%m-%d %H:%M:%S')
#lab4['ADMITTIME']=pd.to_datetime(lab4["ADMITTIME"], format='%Y-%m-%d %H:%M:%S')

#Set the reference time as the admission time for each admission.
ref_time=lab4.groupby("HADM_ID")["CHARTTIME"].min()
#ref_time=lab4.groupby("HADM_ID")["ADMITTIME"].min()
lab5=pd.merge(ref_time.to_frame(name="REF_TIME"),lab4,left_index=True,right_on="HADM_ID")
lab5["TIME_STAMP"]=lab5["CHARTTIME"]-lab5["REF_TIME"]
assert(len(lab5.loc[lab5["TIME_STAMP"]<timedelta(hours=0)].index)==0)



#Create a label code (int) for the labels.
label_dict=dict(zip(list(lab5["LABEL"].unique()),range(len(list(lab5["LABEL"].unique())))))
lab5["LABEL_CODE"]=lab5["LABEL"].map(label_dict)
lab_short=lab5[["SUBJECT_ID","HADM_ID","VALUENUM","TIME_STAMP","LABEL_CODE","DEATHTAG"]]


#Now only select values within 48 hours.
lab_short=lab_short.loc[(lab_short["TIME_STAMP"]<timedelta(hours=48))]
print("Number of patients considered :"+str(lab_short["SUBJECT_ID"].nunique()))


#We then choose a binning factor of 2. 
#Set the time as an integer. We take 2 bins per hour
lab_short["TIME_STAMP"]=round(lab_short["TIME_STAMP"].dt.total_seconds()*2/(100*36)).astype(int)
#Then sort the dataframe with order : Admission ID, Label Code and time stamps
lab_short=lab_short.sort_values(by=["HADM_ID","LABEL_CODE","TIME_STAMP"],ascending=[1,1,1])


lab_short.duplicated(subset=["HADM_ID","LABEL_CODE","TIME_STAMP"]).value_counts()

#Then save locally
lab_short.to_csv(file_path+"lab_events_short.csv")









