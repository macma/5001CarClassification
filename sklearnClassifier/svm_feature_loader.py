import label_loader as ll
def svm_feature_loader():
    allLabel = ll.label_loader()
    featureList = []
    labelList = []
    with open("feature_extracted.txt",'r') as f:
        for line in f: #read in record
            tuples = line.split(":")
            vin = tuples[0]
            featurestr = tuples[1].rstrip()
            features = map(lambda x:float(x),featurestr.split(","))

            label = allLabel[vin]
            featureList.append(features)
            labelList.append(label)
    return {"featureList":featureList,"labelList":labelList}