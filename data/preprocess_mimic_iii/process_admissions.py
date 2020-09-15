'''
Created on Jul 22, 2020

'''
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

import numpy as np

file_path="~/pg_data/physionet.org/files/mimiciii/1.4/"


adm=pd.read_csv(file_path+"ADMISSIONS.csv")
adm.head()

patients_df=pd.read_csv(file_path+"PATIENTS.csv")
patients_df["DOBTIME"]=pd.to_datetime(patients_df["DOB"], format='%Y-%m-%d')
patients_df[["SUBJECT_ID","DOBTIME"]].head()
adm_dob=pd.merge(patients_df[["SUBJECT_ID","DOBTIME"]],adm,on="SUBJECT_ID")

df=adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()
plt.hist(df,bins=100)
plt.show()
print("Number of patients with specific number of admissions : \n",df.value_counts())


subj_ids=list(df[df==1].index) #index of patients with only one visit.
adm_1=adm_dob.loc[adm_dob["SUBJECT_ID"].isin(subj_ids)] #filter out the patients with more than one visit
print("Number of patients remaining in the dataframe: ")
print(len(adm_1.index))



#We now add a new column with the duration of each stay.
adm_1=adm_1.copy()
adm_1['ADMITTIME']=pd.to_datetime(adm_1["ADMITTIME"], format='%Y-%m-%d %H:%M:%S')
adm_1['DISCHTIME']=pd.to_datetime(adm_1["DISCHTIME"], format='%Y-%m-%d %H:%M:%S')

adm_1["ELAPSED_TIME"]=adm_1["DISCHTIME"]-adm_1["ADMITTIME"]
adm_1.head()
adm_1["ELAPSED_DAYS"]=adm_1["ELAPSED_TIME"].dt.days #Elapsed time in days in ICU
plt.hist(adm_1["ELAPSED_DAYS"],bins=200)
plt.show()
print("Number of patients with specific duration of admissions in days : \n",adm_1["ELAPSED_DAYS"].value_counts())


# #Let's now report the death rate in function of the duration stay in ICU.
# adm_1["DEATHTAG"]=0
# adm_1.loc[adm_1.DEATHTIME.notnull(),"DEATHTAG"]=1
# 
# df_deaths_per_duration=adm_1.groupby("ELAPSED_DAYS")["DEATHTAG"].sum()
# df_patients_per_duration=adm_1.groupby("ELAPSED_DAYS")["SUBJECT_ID"].nunique()
# df_death_ratio_per_duration=df_deaths_per_duration/df_patients_per_duration
# plt.plot(df_death_ratio_per_duration)
# plt.title("Death Ratio per ICU stay duration")
# plt.xlabel("Duration in days")
# plt.ylabel("Death rate (Number of deaths/Nunber of patients)")
# plt.show()

adm_2=adm_1.loc[(adm_1["ELAPSED_DAYS"]<30) & (adm_1["ELAPSED_DAYS"]>2)]
print("Number of patients remaining in the dataframe: ")
print(len(adm_2.index))


# adm_2["ADMITTIME"] = pd.to_datetime(adm_2["ADMITTIME"]).dt.date

print(adm_2["ADMITTIME"].sub(adm_2["DOBTIME"]))

print((adm_2["ADMITTIME"].sub(adm_2["DOBTIME"]))/ np.timedelta64(1, 'D'))

# adm_2["DOBTIME"] = pd.to_datetime(adm_2["DOBTIME"]).dt.date

# print((adm_2["ADMITTIME"] - adm_2["DOBTIME"]).astype('int64'))


# adm_2_15=adm_2.loc[((adm_2["ADMITTIME"] - adm_2["DOBTIME"]).dt.days//365)>15].copy()

adm_2_15=adm_2.loc[((adm_2["ADMITTIME"].sub(adm_2["DOBTIME"]))/ np.timedelta64(1, 'D')//365)>15].copy()
print("Number of patients remaining in the dataframe: ")
print(len(adm_2_15.index))


#We remove the admissions with no chart events data.
adm_2_15_chart=adm_2_15.loc[adm_2_15["HAS_CHARTEVENTS_DATA"]==1].copy()
print("Number of patients remaining in the dataframe: ")
print(len(adm_2_15_chart.index))





#We now investigate the admission_type
df_type=adm_2_15_chart.groupby("ADMISSION_TYPE")["SUBJECT_ID"].count()


adm_3=adm_2_15_chart.loc[adm_2_15_chart["ADMISSION_TYPE"]!="NEWBORN"]
print("Number of patients remaining in the dataframe: ")
print(adm_3["SUBJECT_ID"].nunique())

adm_3.to_csv(file_path+"./Admissions_processed.csv")



## INPUTS EVENTS DATA







































