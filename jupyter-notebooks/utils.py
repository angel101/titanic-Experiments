from tflearn.datasets import titanic
from tflearn.data_utils import load_csv

import pandas
import numpy as np

class Utils():
	
	def downloadDataset():
		titanic.download_dataset()
		data, labels = load_csv('train.csv', target_column=1,
                       categorical_labels=True, n_classes=2,
                       columns_to_ignore=[0,8,10])

		return data,labels
	
	def preprocess(titles=True):

		data,labels = Utils.downloadDataset()

		dict_novelties=[]
		dict_salidas=[]
		#feed titles dictionary 
		for item in data:
			if not item[1].split(',')[1].strip().split(' ')[0] in dict_novelties:
				dict_novelties.append(item[1].split(',')[1].strip().split(' ')[0])
			if not item[7] in dict_salidas:
				dict_salidas.append(item[7])


		print(dict_novelties)
		print(dict_salidas)
		dataset = []
	    #get dummy variables for the name titles ('mr,ms,doctor, etc')
		dummiesDataFrame = pandas.get_dummies(dict_novelties)
		dummiesSalida = pandas.get_dummies(dict_salidas)
	    #parce dataset to numeric values 
		for item in data:
			tmp = []
			tmp.append(int(item[0]))
			if titles:
				tmp.append(dummiesDataFrame.index[dummiesDataFrame[item[1].split(',')[1].strip().split(' ')[0]]==True].tolist()[0])

			tmp.append(1 if item[2]=='female' else 0)
			tmp.append(float(item[3] if item[3]!= '' else 0))
			tmp.append(int(item[4]))
			tmp.append(int(item[5]))
			tmp.append(item[6])
			tmp.append(dummiesSalida.index[dummiesSalida[item[7]]==True].tolist()[0])
			dataset.append(tmp)
		dataset = np.array(dataset)


		Y=[]
		#reshape in a not that fancy way label array 
		for label in labels:
		    if label[0] == 1.0:
		        Y.append(1)
		    else:
		        Y.append(0)



		return dataset, Y

