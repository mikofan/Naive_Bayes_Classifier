import csv
import os
import math
import operator
import random


'''
External Reference used:-
https://realpython.com/welcome/
https://codereview.stackexchange.com/questions/9222/calculating-population-standard-deviation/9224
https://www.coursera.org/learn/advanced-machine-learning-signal-processing/lecture/9D2Pz/bayesian-inference-in-python
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''

#Loading Training data sets from inputdata.txt file
def loadTrainingDataSet():
    dataPoints = []
    fileh = open(os.getcwd() + "//inputdata.txt",'r')
    try:
        csv_reader = csv.reader(fileh,delimiter=',')
        for row in csv_reader:
            tempList = []
            tempList.append(float(row[0])) # Height
            tempList.append(float(row[1])) # Weight
            tempList.append(float(row[2])) # Age
            tempList.append(row[3]) # Class Label
            dataPoints.append(tempList)
    except:
        pass
    fileh.close()
    return dataPoints

#Loading Test data sets from sampledata.txt file
def loadTestData():
    sampleData = []
    fileSecondh = open(os.getcwd() + "//sampledata.txt")

    try:
        csv_reader = csv.reader(fileSecondh,delimiter=',')
        for row in csv_reader:
            tempList = []
            tempList.append(float(row[0])) # Height
            tempList.append(float(row[1])) # Weight
            tempList.append(float(row[2])) # Age
            #tempList.append(row[3]) # Class Label
            sampleData.append(tempList)
    except:
        pass
    fileSecondh.close()
    return sampleData

#Get labels from training data set
def getLabels(inputDataList):
    labels = []
    for x in range(len(inputDataList)):
        inputList = inputDataList[x]
        labels.append(inputList[3])
    return labels

#calculating prior for label class
def getPriors(labels):
    priors = {}
    N = len(labels)
    countM = 0.0
    countW = 0.0
    for className in labels:
        if className == 'W':
            countW+=1
        elif className == 'M':
            countM+=1

    priorForM = float(countM/N)
    priorForW = float(countW/N)
    priors['M'] = priorForM
    priors['W'] = priorForW
    return priors

#computing mean and standard deviation for each feature and for each class types
def computeMeanSdForTraining(trainingDataList,uniquelabels):
    dict_elems = {}
    super_list = []
    count = 0
    while count < 3:
        subListForW = []
        subListForM = []
        tempBigList = []
        for i in range(len(trainingDataList)):
            value = trainingDataList[i][-1]
            if value == 'W':
                subListForW.append(trainingDataList[i][count])
            else:
                subListForM.append(trainingDataList[i][count])

        meanM = getmean(subListForM)
        meanW = getmean(subListForW)
        sdM = getstddev(subListForM)
        sdW = getstddev(subListForW)
        newSubListM = []
        newSubListW = []
        newSubListM.append(meanM)
        newSubListM.append(sdM)
        newSubListW.append(meanW)
        newSubListW.append(sdW)
        #first list for M
        tempBigList.append(newSubListM)
        tempBigList.append(newSubListW)
        dict_elems[count] = tempBigList
        count+=1

    return dict_elems


#calculating Likelihood value using x(ith) feature of the test data set
def getLikelihood(x,featureIndex,model,className):
    f = 0
    try:
        classStatistics = model[className]
        mean = classStatistics[featureIndex][0]
        std = classStatistics[featureIndex][1]
        first_part = 1/math.sqrt(2*3.14*std**2)
        second_part = math.exp(-(x-mean)**2)/(2*std**2)
        f =  first_part * second_part
    except:
        pass
    #print(f)
    return float(f)

#calculating posteriors using model and prior. model is dictionary contains mean and standard deviations of all features per class.
def getPosterior(x,model,priors):
    posteriors = {}
    for className in priors:
        p = 1.0
        for featureIndex in range(len(x)):
            p = p * (getLikelihood(x[featureIndex],featureIndex,model,className) * priors[className])
        posteriors[className] = p
    return posteriors

#getting label of class having highest posterior probablity value
def classify(x,model,priors):
    posteriors = getPosterior(x,model,priors)
    tempList = []
    for key,val in posteriors.items():
        tempList.append(float(val))
    tempList.sort()
    value = tempList[1]
    for key ,val in posteriors.items():
        if val == value:
            return key

#calculating mean of list of items
def getmean(lst):
    result = 0.0
    try:
        result = sum(lst) / len(lst)
    except:
        pass

    return result

#calculating standard deviation of list of items
def getstddev(lst):
    variance = 0
    mn = getmean(lst)
    for e in lst:
        variance += (e-mn)**2
    try:
        variance = variance + variance/len(lst)
    except:
        pass
    return math.sqrt(variance)

#main function contains main entry for the programme
def main():
    trainingDataList  = loadTrainingDataSet()
    testDataList = loadTestData()
    labels = getLabels(trainingDataList)
    priors = getPriors(labels)
    uniquelabels = list(set(labels))
    statsForM = []
    statsForW = []
    statsData = computeMeanSdForTraining(trainingDataList,uniquelabels) #存放着所有相关元素的均值和标准差
    for key,value in statsData.items():
        tempList = value
        statsForM.append(tempList[0])
        statsForW.append(tempList[1])
    model = {}
    model['M'] = statsForM
    model['W'] = statsForW

    predictions = [classify(x,model,priors) for x in testDataList]
    print predictions


main()