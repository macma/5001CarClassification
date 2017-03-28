import os
import label_loader as ll
import numpy as np

def matrix_loader():
    holiday = ["2016-4-30","2016-5-1","2016-5-2","2016-5-7","2016-5-8"]
    workday = ["2016-4-29","2016-5-3","2016-5-4","2016-5-5","2016-5-6","2016-5-9","2016-5-10","2016-5-11"]
    labelList = ll.label_loader()
    print labelList
    allFiles = os.listdir("matrix")
    print len(labelList)
    training = []
    for file in allFiles:
        mat = np.loadtxt("matrix/"+file)
        splits = file.split(".")
        vin = splits[0]
        date = splits[1]
        dict = {"data":mat,"label":labelList[vin]}
        training.append(dict)
        if(date in workday):
            training.append(dict)
            training.append(dict)
    return training