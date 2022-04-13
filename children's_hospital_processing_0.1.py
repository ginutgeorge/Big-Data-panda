import pandas as pd
import re

import sys
import time
import numpy  # Slicing individual rows of data.
from numpy import array
from numpy import reshape

# Scikit-Learn data preparation.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Skikit-Learn prediction algorithms.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF  # For GaussianProcessClassifier.
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Didn't use histogram-based gradient boosting.
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Scikit-Learn measures of prediction accuracy.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
COLUMN_LIST = ["HADM_ID", "SUBJECT_ID", "SYMPTOM_NAME"]
ALGO_NAMES = ["GaussianNB", "DecisionTreeClassifier", "AdaBoostClassifier", "MLPClassifier",
              "GradientBoostingClassifier", "HistGradientBoostingClassifier", "LinearDiscriminantAnalysis",
              "RandomForestClassifier",
              "DecisionTreeClassifier"]
aspdColumnName = ["SUBJECT_ID", "HADM_ID", "INSURANCE", "SYMPTOM_NAME", "GENDER", "TITLE", "EXPIRE_FLAG"]

##Constance
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


def analyse(my_dataset, alg_no, column_Name, data):

    column_names = ["Threshold", "TP", "FP", "FN", "TN", "Precision_P", "Recall_P"]
    final_df = pd.DataFrame(columns = column_names)
    num_positives = 0.001
    no_of_ones = list(my_dataset.EXPIRE_FLAG).count(1)
    if no_of_ones > 0:
        num_positives = list(my_dataset.EXPIRE_FLAG).count(1)
    num_lines = len(my_dataset.index)  # One row of headers and 510 rows of data.
    number_of_column = len(my_dataset.columns)
    column_narray = []
    for n in range(0, number_of_column):
        column_narray.append(n)
    y_list = column_narray.pop()
    X = my_dataset.iloc[:, column_narray].values  # Columns of inputs.
    y = my_dataset.iloc[:, y_list].values  # Last column is output, Yes or No.


    # Store results in a big array.
    all_results = [[0 for m in range(num_lines)] for n in range(5000)]

    # Do predictions for increasingly strict probabilities.
    # Precision equals TP / (TP + FP)
    # Recall = TP Rate TP / (TP + FN)
    print("#Starting on " + str(num_lines) + " leave-one-out trainings.", end="")
    sys.stdout.flush()  # Flush the buffer, so we can see progress.
    for line in range(0, num_lines):
        # Leave-one-out cross-validation, test data is a single row.
        X_test = array(X[line])
        X_test = X_test.reshape((1, X_test.shape[0]))

        y_test = array(y[line])
        y_test = y_test.reshape(1)

        # Training data is everything except that row of test data.
        X_train = numpy.delete(X, (line), axis=0)
        y_train = numpy.delete(y, (line), axis=0)
        # Gaussian naive Bayesian.
        algo = "Gaussian"
        if alg_no == 0:
            clf = GaussianNB()
            algo = "Gaussian"
        elif alg_no == 1:
            clf = DecisionTreeClassifier(criterion="entropy", random_state=17)
            algo = "DecisionTreeClassifier"
        elif alg_no == 2:
            clf = AdaBoostClassifier(n_estimators=100, random_state=17)
            algo = "AdaBoostClassifier"
        elif alg_no == 3:
            #clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=5000, activation='relu', solver='adam',
            #                   tol=0.000000001)
            #algo = "MPLClassifier"
            clf = GaussianNB()
            algo = "Gaussian"

        elif alg_no == 4:
            clf = GradientBoostingClassifier()
            algo = "GradientBoostingClassifier"
        elif alg_no == 5:
            clf = HistGradientBoostingClassifier()
            algo = "HistGradientBoostingClassifier"
        elif alg_no == 6:
            clf = LinearDiscriminantAnalysis()
            algo = "LinearDiscriminantAnalysis"
        elif alg_no == 7:
            clf = RandomForestClassifier(max_depth=10, random_state=17)
            algo = "RandomForestClassifier"
        elif alg_no == 8:
            clf = DecisionTreeClassifier(random_state=17)
            algo = "DecisionTreeClassifier"

        else:
            clf = GaussianNB()



        # Use the training data to create a model.
        clf.fit(X_train, y_train)

        # Find the model's predicted probability of each output.
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Eye candy.
        if (0 == (line % 10)):
            print(".", end="")
            sys.stdout.flush()  # Flush the buffer, so we can see progress.

        # Is it sufficiently confident to venture a prediction?
        for threshint in range(5000, 10000):
            thresh = threshint / 10000.0  # Threshold goes from 0.5 to 0.999
            my_TP = 0
            my_FP = 0
            my_FN = 0
            my_TN = 0
            if (y_prob[0][0] >= thresh):  # First probability is P = churn, not stay.
                if (y_pred == y_test):
                    my_TP = 1
                    my_FP = 0
                else:
                    my_FP = 1
                    my_TP = 0
            elif (y_prob[0][1] >= thresh):  # Second probability is N = stay, not churn.
                if (y_pred == y_test):
                    my_TN = 1
                    my_FN = 0
                else:
                    my_FN = 1
                    my_TN = 0

            # Should be only one or zero predictions.
            if ((my_TP + my_FP + my_FN + my_TN) > 1):
                print("Error, should be only one prediction, or none, not ",
                      (my_TP + my_FP + my_FN + my_TN))

            # Store that set of results, as an immutable list.
            # print(thresh, line, y_prob[0][0], y_prob[0][1], my_TP, my_FP, my_FN, my_TN)
            single_result = (thresh, line, my_TP, my_FP, my_FN, my_TN)
            all_results[threshint - 5000][line] = single_result

    print(". done.")

    # So how did it go on all those leave-one-out data sets?
    print("#Thresh TP FP FN TN Precision_P    Recall_P")
    for threshint in range(5000, 10000):
        thresh = threshint / 10000.0  # Threshold goes from 0.5 to 0.999
        prev_TP = my_TP
        prev_FP = my_FP
        prev_FN = my_FN
        prev_TN = my_TN
        my_TP = 0
        my_FP = 0
        my_FN = 0
        my_TN = 0
        for line in range(0, num_lines):
            single_result = all_results[threshint - 5000][line]
            # print(single_result)
            if (1 == single_result[2]):
                my_TP += 1
            elif (1 == single_result[3]):
                my_FP += 1
            elif (1 == single_result[4]):
                my_FN += 1
            elif (1 == single_result[5]):
                my_TN += 1

        # Any different from the previous threshold?
        if ((threshint > 5000) and
                (prev_TP == my_TP) and
                (prev_FP == my_FP) and
                (prev_FN == my_FN) and
                (prev_TN == my_TN)):
            pass  # Don't print anything, same as previous line.
        else:
            # If there were any predictions, calculate precision and TP rate.
            if ((my_TP + my_FP) > 0):
                my_precis = my_TP / (my_TP + my_FP)
            else:
                my_precis = 0

            if ((my_TP + my_FN) > 0):
                my_recall = my_TP / (num_positives)  # Out of all positive cases.
            else:
                my_recall = 0

            print(str(thresh), "  ", my_TP, my_FP, my_FN, my_TN,
                  "  ", my_precis, my_recall)
            arr = [str(thresh),my_TP,my_FP,my_FN,my_TN,my_precis,my_recall]
            data_to_append = {}
            for i in range(len(final_df.columns)):
                data_to_append[final_df.columns[i]] = arr[i]
            final_df = final_df.append(data_to_append, ignore_index=True)

    # All finished, repeat the row of headers.
    print("#Thresh TP FP FN TN Precision_P    Recall_P")
    createFile(final_df,"OUTPUT" + str(algo) + str(column_Name) + str(data))


def addColumn(dataName, column):
    newData[column] = dataName[column]
    return newData


def createFile(result, name):
    print(name)
    file_name = re.sub('[^a-zA-Z0-9 \n\.]', '', str(name))
    # file_name.encode('utf-8').decode('unicode-escape')
    result.to_csv(file_name + "_Dataset.csv", index=False)


def split(column_Name, Data_file):
    # listing unique data
    data_list = Data_file[column_Name].unique()

    # creating different files each for each unique
    algo_num = [x for x in range(9)]
    for data in data_list:
        new_Data = Data_file.loc[Data_file[column_Name] == data]
        del new_Data[column_Name]
        print('****' + str(data) + '******')
        print(new_Data)
        no_of_ones = list(new_Data.EXPIRE_FLAG).count(1)
        no_of_zeros = list(new_Data.EXPIRE_FLAG).count(0)
        print("No of Ones " + str(no_of_ones))
        print("No of Zeros " + str(no_of_zeros))
        if len(new_Data) > 500 and len(new_Data) < 700 and no_of_ones > 1 and no_of_zeros > 1:
            print("******  " + data + " *****")
            analyse(new_Data, 0, column_Name, data)
            for n in algo_num:
                print("\n******  " + ALGO_NAMES[n] + " *****\n")
                #analyse(new_Data, n, column_Name,data)
        createFile(new_Data, str(column_Name) + str(data))


def cutColumn(Data_file, Column_list):
    for column in Column_list:
        cutFile = addColumn(Data_file, column)
    return cutFile


def loadFiles(FILE_NO1, FILE_NO2, FILE_NO3, FILE_NO4):
    admissionFile = pd.read_csv(FILE_NO1)
    diagnosesFile = process(pd.read_csv(FILE_NO2), DISEASE_COLUMN)
    symptomsFile = process(pd.read_csv(FILE_NO3), SYMPTOM_COLUMN)
    symptomsFile = cutColumn(symptomsFile, COLUMN_LIST)
    patientsFile = pd.read_csv(FILE_NO4)
    return admissionFile, diagnosesFile, symptomsFile, patientsFile


def process(Data_file, column_Name):
    data_directory = pd.DataFrame()
    number_array = []
    data_list = Data_file[column_Name].unique()
    print(len(data_list))
    for value in range(len(data_list)):
        number_array.append(str(value))
    print(data_list)
    Data_file[column_Name] = Data_file[column_Name].replace(data_list, number_array)
    data_directory["Assigned_number"] = number_array
    data_directory[column_Name] = data_list
    print(data_directory)
    print(">>>>>****..." + column_Name + "....****<<<<<<<")
    createFile(data_directory, column_Name)
    createFile(Data_file, "Process_Debug")
    return Data_file


def mergeFiles(loadedFiles):
    asDF = pd.merge(loadedFiles[0], loadedFiles[2], on=[MERGECOLUMN_1, MERGECOLUMN_2],
                    how='outer')  # first merging the admission file and the symptoms file
    aspDF = pd.merge(asDF, loadedFiles[3], on=[MERGECOLUMN_2], how='outer')  # Then merging the new file with patients
    aspdDF = pd.merge(aspDF, loadedFiles[1], on=[MERGECOLUMN_3])  # Now merging the new file with diagnoses
    return aspdDF


def cleanData(mergedData):
    # Filtering the file to have only the necessary columns


    for column in aspdColumnName:
        aspdDF_Cut = addColumn(mergedData, column)

    # Cleaning the file and replacing missiing values


    final_Data = aspdDF_Cut.fillna(0)

    final_Data['INSURANCE'] = final_Data['INSURANCE'].replace(
        ['Self Pay', 'Discount', 'Foundation', 'Medical Insurance',
         'Premiums Pay', 'the General Card'], [0, 1, 2, 3, 4, 5])
    final_Data['GENDER'] = final_Data['GENDER'].replace(['M', 'F'], [0, 1])

    final_Data.rename(columns={'TITLE': 'DISEASES'}, inplace=True)
    createFile(final_Data, 'final_Data')
    return final_Data



def main():
    # loading all files
    sys_time = time.time()

    print(sys_time)
    loadedFiles = loadFiles(FILE_NO1, FILE_NO2, FILE_NO3, FILE_NO4)
    # result1 = process(loadedFiles[2],SYMPTOM_COLUMN)
    # result2 = process(loadedFiles[1],DISEASE_COLUMN)
    mergedData = mergeFiles(loadedFiles)
    final_Data = cleanData(mergedData)

    split('GENDER', final_Data)
    split('INSURANCE', final_Data)
    split('DISEASES', final_Data)
    print(time.time() - sys_time)


main()