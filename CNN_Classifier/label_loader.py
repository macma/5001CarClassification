'''
return a dictionary of [{vin:,label:}]
Uber - 0
private - 1
'''
def label_loader():
    allSeries = {}
    with open("label.csv",'r') as f:
        for line in f: #read in record
            tuples = line.split(",")
            key = tuples[0]
            val = 0 if(tuples[1][0]=='U')else 1
            allSeries[key]= val
    return allSeries