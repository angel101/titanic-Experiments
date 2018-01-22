from tflearn.datasets import titanic
from tflearn.data_utils import load_csv

import pandas

class Utils():
	
	def downloadDataset():
		titanic.download_dataset()
		data, labels = load_csv('titanic_dataset.csv', target_column=0,
                       categorical_labels=True, n_classes=2)

		return data,labels
	
	def preprocess():

		data,labels = Utils.downloadDataset()

		dict_novelties=[]
		
		#feed titles dictionary 
		for item in data:
			if not item[1].split(',')[1].strip().split(' ')[0] in dict_novelties:
				dict_novelties.append(item[1].split(',')[1].strip().split(' ')[0])


		dataset = []
	    #get dummy variables for the name titles ('mr,ms,doctor, etc')
		dummiesDataFrame = pandas.get_dummies(dict_novelties)
	    #parce dataset to numeric values 
		for item in data:
			tmp = []
			tmp.append(int(item[0]))

			tmp.append(dummiesDataFrame.index[dummiesDataFrame[item[1].split(',')[1].strip().split(' ')[0]]==True].tolist()[0])

			tmp.append(1 if item[2]=='female' else 0)
			tmp.append(float(item[3]))
			tmp.append(int(item[4]))
			tmp.append(int(item[5]))
			tmp.append(float(item[7]))
			dataset.append(tmp)
		dataset = pandas.DataFrame(dataset)


		Y=[]
		#reshape in a not that fancy way label array 
		for label in labels:
		    if label[0] == 1.0:
		        Y.append(1)
		    else:
		        Y.append(0)



		return dataset, Y

