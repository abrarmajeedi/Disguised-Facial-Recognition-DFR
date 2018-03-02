import pandas as pd
import numpy as np
import math

df =  pd.read_csv("predictions.csv",header = None)
images = df.iloc[:,-1]
points = df.iloc[:,:-1]
points = pd.DataFrame(points)
names = []
for image in images:
	image = image.split('_')[0]
	names.append(image)
df2 = pd.DataFrame(names)
x = pd.concat([points, df2], axis = 1)
x.to_csv("predictions.csv",header =None, index = None)
#


df =  pd.read_csv("train.csv",header = None)
images = df.iloc[:,-1]
points = df.iloc[:,:-1]
points = pd.DataFrame(points)
names = []
for image in images:
	image = image.split('_')[0]
	names.append(image)
df2 = pd.DataFrame(names)
x = pd.concat([points, df2], axis = 1)
x.to_csv("train.csv",header =None, index = None)
#


df =  pd.read_csv("test.csv",header = None)
images = df.iloc[:,-1]
points = df.iloc[:,:-1]
points = pd.DataFrame(points)
names = []
for image in images:
	image = image.split('_')[0]
	names.append(image)
df2 = pd.DataFrame(names)
x = pd.concat([points, df2], axis = 1)
x.to_csv("test.csv",header =None, index = None)


ratio = 1

def find_ratio(a,b,c,d):
    d1 = float(math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2))
    d2 = float(math.sqrt((c[1]-d[1])**2 + (c[0]-d[0])**2))
    #ratio = float(d1/d2)
    return float(d1/d2)


angles = []
def find_angle(a,b,c):
    num1 = b[1] - a[1]
    den1 = b[0] - a[0]
    num2 = b[1] - c[1]
    den2 = b[0] - c[0]
    if den1 == 0:
	m1 = 10000000
    else:
        m1 = float(num1/den1)
    

    if den2 ==0:
        m2 = 100000000
    else:    
    	m2 = float(num2/den2)
    

    if m1*m2 == -1:
	return 90

    m = float((m1 - m2)/ (1+m1*m2))
    return float(math.degrees(math.atan(m)))
  

def find_parameters(point):
	paramslist = []
	#find angles
	paramslist = []
	
	#find angles
	paramslist.append(find_angle(point[1],point[0],point[5]))
	paramslist.append(find_angle(point[1],point[0],point[4]))
	paramslist.append(find_angle(point[0],point[5],point[1]))
	paramslist.append(find_angle(point[0],point[1],point[6]))
	paramslist.append(find_angle(point[0],point[10],point[15]))	
	paramslist.append(find_angle(point[0],point[10],point[18]))
	paramslist.append(find_angle(point[4],point[15],point[6]))
	paramslist.append(find_angle(point[10],point[15],point[18]))
	paramslist.append(find_angle(point[10],point[15],point[12]))
	paramslist.append(find_angle(point[1],point[13],point[2]))

	paramslist.append(find_angle(point[6],point[13],point[7]))
	paramslist.append(find_angle(point[5],point[13],point[8]))
	paramslist.append(find_angle(point[18],point[13],point[19]))
	paramslist.append(find_angle(point[18],point[16],point[19]))
	paramslist.append(find_angle(point[15],point[13],point[17]))	
	paramslist.append(find_angle(point[12],point[13],point[14]))
	paramslist.append(find_angle(point[4],point[15],point[12]))
	paramslist.append(find_angle(point[14],point[17],point[9]))
	paramslist.append(find_angle(point[7],point[17],point[9]))
	paramslist.append(find_angle(point[11],point[17],point[19]))
	paramslist.append(find_angle(point[3],point[11],point[17]))
	paramslist.append(find_angle(point[3],point[11],point[19]))
	paramslist.append(find_angle(point[2],point[3],point[9]))
	paramslist.append(find_angle(point[2],point[3],point[8]))
	paramslist.append(find_angle(point[3],point[2],point[7]))
	paramslist.append(find_angle(point[0],point[0],point[1]))
	paramslist.append(find_angle(point[2],point[2],point[3]))
	paramslist.append(find_angle(point[2],point[8],point[3]))
	
	
	
	#find ratios
	
	paramslist.append(find_ratio(point[0],point[1],point[0],point[10]))
	paramslist.append(find_ratio(point[2],point[3],point[3],point[11]))
	paramslist.append(find_ratio(point[1],point[2],point[13],point[16]))
	paramslist.append(find_ratio(point[0],point[5],point[10],point[5]))
	paramslist.append(find_ratio(point[3],point[8],point[8],point[11]))
	paramslist.append(find_ratio(point[5],point[13],point[10],point[13]))
	paramslist.append(find_ratio(point[8],point[13],point[11],point[13]))
	paramslist.append(find_ratio(point[0],point[4],point[1],point[4]))
	paramslist.append(find_ratio(point[0],point[6],point[1],point[6]))
	paramslist.append(find_ratio(point[2],point[7],point[3],point[7]))
	paramslist.append(find_ratio(point[2],point[9],point[3],point[9]))
	paramslist.append(find_ratio(point[0],point[1],point[1],point[13]))

	paramslist.append(find_ratio(point[2],point[3],point[2],point[13]))
	
	paramslist.append(find_ratio(point[10],point[15],point[10],point[13]))
	paramslist.append(find_ratio(point[11],point[17],point[11],point[13]))
	paramslist.append(find_ratio(point[0],point[5],point[10],point[15]))
	paramslist.append(find_ratio(point[3],point[8],point[11],point[17]))
	paramslist.append(find_ratio(point[1],point[5],point[5],point[13]))
	paramslist.append(find_ratio(point[2],point[8],point[8],point[13]))
	paramslist.append(find_ratio(point[0],point[1],point[13],point[16]))
	paramslist.append(find_ratio(point[2],point[3],point[13],point[16]))

	paramslist.append(find_ratio(point[0],point[4],point[0],point[5]))
	paramslist.append(find_ratio(point[1],point[5],point[1],point[6]))
        paramslist.append(find_ratio(point[2],point[7],point[2],point[8]))
        paramslist.append(find_ratio(point[3],point[8],point[3],point[9]))
        
	paramslist.append(find_ratio(point[0],point[10],point[10],point[13]))
	paramslist.append(find_ratio(point[3],point[11],point[11],point[13]))
	paramslist.append(find_ratio(point[12],point[13],point[13],point[16]))
	paramslist.append(find_ratio(point[13],point[14],point[13],point[16]))
	
	paramslist.append(find_ratio(point[10],point[18],point[10],point[13]))
	paramslist.append(find_ratio(point[11],point[19],point[11],point[13]))
	
	paramslist.append(find_ratio(point[0],point[10],point[10],point[18]))
	paramslist.append(find_ratio(point[3],point[11],point[11],point[19]))
	
	paramslist.append(find_ratio(point[13],point[16],point[18],point[19]))
	paramslist.append(find_ratio(point[12],point[15],point[15],point[18]))
	paramslist.append(find_ratio(point[14],point[17],point[17],point[19]))
	
	paramslist.append(find_ratio(point[1],point[13],point[0],point[10]))
        paramslist.append(find_ratio(point[2],point[13],point[3],point[11]))
        
	paramslist.append(find_ratio(point[4],point[6],point[13],point[16]))
	paramslist.append(find_ratio(point[7],point[9],point[13],point[16]))


	paramslist.append(find_ratio(point[4],point[9],point[10],point[11]))
	paramslist.append(find_ratio(point[0],point[3],point[4],point[9]))
	paramslist.append(find_ratio(point[1],point[18],point[13],point[16]))
	paramslist.append(find_ratio(point[2],point[19],point[13],point[16]))
	
	paramslist.append(find_ratio(point[15],point[17],point[18],point[19]))
	paramslist.append(find_ratio(point[10],point[18],point[13],point[16]))
	paramslist.append(find_ratio(point[11],point[19],point[13],point[16]))
	paramslist.append(find_ratio(point[0],point[4],point[13],point[16]))
    	paramslist.append(find_ratio(point[1],point[6],point[13],point[16]))
	paramslist.append(find_ratio(point[2],point[7],point[13],point[16]))
        paramslist.append(find_ratio(point[3],point[9],point[13],point[16]))
	paramslist.append(find_ratio(point[10],point[11],point[13],point[16]))
	paramslist.append(find_ratio(point[1],point[16],point[13],point[16]))
	paramslist.append(find_ratio(point[2],point[16],point[13],point[16]))
    	paramslist.append(find_ratio(point[1],point[13],point[13],point[16]))
	paramslist.append(find_ratio(point[2],point[13],point[13],point[16]))
	paramslist.append(find_ratio(point[5],point[8],point[13],point[16]))
	paramslist.append(find_ratio(point[6],point[7],point[13],point[16]))
	paramslist.append(find_ratio(point[3],point[14],point[13],point[16]))
	paramslist.append(find_ratio(point[0],point[12],point[13],point[16]))
	paramslist.append(find_ratio(point[5],point[18],point[13],point[16]))
    	paramslist.append(find_ratio(point[8],point[19],point[13],point[16]))
	paramslist.append(find_ratio(point[12],point[14],point[18],point[19]))
	paramslist.append(find_ratio(point[15],point[17],point[13],point[16]))
	paramslist.append(find_ratio(point[13],point[15],point[13],point[16]))
	paramslist.append(find_ratio(point[13],point[17],point[13],point[16]))

    	return paramslist


tempcsv = []
df = pd.read_csv("predictions.csv",header = None)
for i in range(0,50):
	parameters = []
	point = []
	col = df.iloc[i]
	col = col.values
	name = col[40]
	print i
	#convert coords to 20 tuples, and append them to a list of points
	for j  in range(0,39,2):
        	t = tuple((float(col[j]/ratio),float(col[j+1]/ratio)))
        	point.append(t)
	#gets a list of parameters from the list of points passed in
	parameters = find_parameters(point)
	parameters.append(name)
	tempcsv.append(parameters)
	
df2 = pd.DataFrame(tempcsv)
df2.to_csv("anglesratiospredictions.csv",header = None,index =None)

#

tempcsv = []
df = pd.read_csv("train.csv",header = None)
for i in range(0,3500):
	parameters = []
	point = []
	col = df.iloc[i]
	col = col.values
	name = col[40]
	
	#convert coords to 20 tuples, and append them to a list of points
	for j  in range(0,39,2):
        	t = tuple((float(col[j]/ratio),float(col[j+1]/ratio)))
        	point.append(t)
	#gets a list of parameters from the list of points passed in
	parameters = find_parameters(point)
	parameters.append(name)
	tempcsv.append(parameters)
	
df2 = pd.DataFrame(tempcsv)
df2.to_csv("anglesratiostrain.csv",header = None,index =None)

#

tempcsv = []
df = pd.read_csv("test.csv",header = None)
for i in range(0,500):
	parameters = []
	point = []
	col = df.iloc[i]
	col = col.values
	name = col[40]
	
	#convert coords to 20 tuples, and append them to a list of points
	for j  in range(0,39,2):
        	t = tuple((float(col[j]/ratio),float(col[j+1]/ratio)))
        	point.append(t)
	#gets a list of parameters from the list of points passed in
	parameters = find_parameters(point)
	parameters.append(name)
	tempcsv.append(parameters)
	
df2 = pd.DataFrame(tempcsv)
df2.to_csv("anglesratiostest.csv",header = None,index =None)


