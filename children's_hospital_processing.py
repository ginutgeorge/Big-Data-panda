from importlib.resources import path
import os
from pathlib import Path
import pandas as pd
import re

newData = pd.DataFrame()
asDF_Cut = pd.DataFrame()
heart_data = pd.DataFrame()
diseases_list = pd.DataFrame()
gender_data = pd.DataFrame()
insurance_data = pd.DataFrame()
readFile = []
loadedFiles = []
df = pd.DataFrame()
newSym = pd.DataFrame()

MERGECOLUMN_1 = "HADM_ID"
SYMPTOM_COLUMN = "SYMPTOM_NAME"
MERGECOLUMN_2 = "SUBJECT_ID"
MERGECOLUMN_3 = "ICD10_CODE_CN"
DISEASE_COLUMN = "TITLE"
COLUMN_LIST = ["HADM_ID","SUBJECT_ID","SYMPTOM_NAME"]

#Constance
FILE_NO1 = 'ADMISSIONS.CSV'
FILE_NO2 = 'D_ICD_DIAGNOSES.CSV'
FILE_NO3 = 'EMR_SYMPTOMS.CSV'
FILE_NO4 = 'PATIENTS.CSV'


admissionFile = pd.DataFrame()
diagnosesFile = pd.DataFrame()
symptomsFile = pd.DataFrame()
patientsFile = pd.DataFrame()
symptoms = pd.DataFrame()
mergedData = pd.DataFrame()
count = pd.DataFrame()


#add code
def addColumn(dataName, column):
    newData[column] = dataName[column]
    return newData


def createFile(result, name):
    print(name)
    file_name = re.sub('[^a-zA-Z0-9 \n\.]', '', str(name))
    print (file_name)
    # file_name.encode('utf-8').decode('unicode-escape')
    os.makedirs('output', exist_ok=True)
    result.to_csv( f'output/{file_name}_Dataset.csv', index=False)

def split(column_Name, Data_file):
    # listing diseases
    data_list = Data_file[column_Name].unique()

    # creating different files each for each disease
    for data in data_list:
        new_Data = Data_file.loc[Data_file[column_Name] == data]
        del new_Data[column_Name]
        print('************' + str(data) + '**************')
        print(new_Data)
        createFile(new_Data, str(column_Name)+str(data))

def cutColumn(Data_file,Column_list):
    for column in Column_list:
        cutFile = addColumn(Data_file, column)
    return cutFile

def loadFiles(FILE_NO1,FILE_NO2,FILE_NO3,FILE_NO4):
    admissionFile = pd.read_csv(FILE_NO1)
    diagnosesFile = process(pd.read_csv(FILE_NO2),DISEASE_COLUMN)
    symptomsFile = process(pd.read_csv(FILE_NO3),SYMPTOM_COLUMN)
    symptomsFile = cutColumn(symptomsFile,COLUMN_LIST)
    patientsFile = pd.read_csv(FILE_NO4)
    return admissionFile, diagnosesFile, symptomsFile, patientsFile

def process(Data_file,column_Name):
    data_directory = pd.DataFrame()
    number_array = []
    data_list = Data_file[column_Name].unique()
    print(len(data_list))
    for value in range(len(data_list)):
        number_array.append(str(value))
    print(data_list)
    Data_file[column_Name] = Data_file[column_Name].replace(data_list,number_array)
    data_directory["Assigned_number"] = number_array
    data_directory[column_Name] = data_list
    print(data_directory)
    print(">>>>>***********..."+column_Name+"....***********<<<<<<<")
    createFile(data_directory,column_Name)
    createFile(Data_file,"Process_Debug")
    return Data_file



def mergeFiles(loadedFiles):
    asDF = pd.merge(loadedFiles[0], loadedFiles[2], on=[MERGECOLUMN_1,MERGECOLUMN_2], how='outer') # first merging the admission file and the symptoms file
    aspDF = pd.merge(asDF, loadedFiles[3], on=[MERGECOLUMN_2], how='outer') # Then merging the new file with patients
    aspdDF = pd.merge(aspDF, loadedFiles[1], on=[MERGECOLUMN_3]) # Now merging the new file with diagnoses
    return aspdDF

def cleanData(mergedData):
    # Filtering the file to have only the necessary columns
    aspdColumnName = ["SUBJECT_ID", "HADM_ID", "INSURANCE", "HOSPITAL_EXPIRE_FLAG", "SYMPTOM_NAME"
        , "GENDER", "TITLE", "EXPIRE_FLAG"]

    for column in aspdColumnName:
        aspdDF_Cut = addColumn(mergedData, column)

    # Cleaning the file and replacing missiing values
    fill_values = {'INSURANCE': 0,
                   'ICD10_CODE_CN': 0,
                   'HOSPITAL_EXPIRE_FLAG': 0,
                   'SYMPTOM_NAME': 0,
                   'GENDER': 0,
                   'DOB': 0,
                   'DOD': 0,
                   'EXPIRE_FLAG': 0,
                   'TITLE': 0}


    final_Data = aspdDF_Cut.fillna(fill_values)

    final_Data['INSURANCE'] = final_Data['INSURANCE'].replace(['Self Pay','Discount','Foundation','Medical Insurance',
                                                               'Premiums Pay','the General Card'],[0,1,2,3,4,5])
    final_Data['GENDER'] =  final_Data['GENDER'].replace(['M','F'],[0,1])



    final_Data.rename(columns={'TITLE': 'DISEASES'}, inplace=True)
    createFile(final_Data, 'final_Data')
    return final_Data


def main():
    # loading all files
    loadedFiles = loadFiles(FILE_NO1, FILE_NO2, FILE_NO3, FILE_NO4)
    #result1 = process(loadedFiles[2],SYMPTOM_COLUMN)
    #result2 = process(loadedFiles[1],DISEASE_COLUMN)
    mergedData = mergeFiles(loadedFiles)
    final_Data = cleanData(mergedData)

    split('DISEASES',final_Data)
    split('GENDER',final_Data)
    split('INSURANCE',final_Data)






main()