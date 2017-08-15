import numpy as np
import pandas as pd

batch_size=32

def create_datasets():
	train_val = pd.read_csv('Data/MNIST/mnist_train.csv',header=None)
	test = pd.read_csv('Data/MNIST/mnist_test.csv',header=None)

	# Shuffle training
	index_s=np.random.permutation(np.arange(train_val.shape[0]))
	train_val=train_val.iloc[index_s]

	#train and validation split
	train = train_val.iloc[0:int(train_val.shape[0]*.8)]
	val = train_val.iloc[int(train_val.shape[0]*.8):]

	# Preprocessing
	x_train, y_train = train.as_matrix()[:,1:],train.as_matrix()[:,0].reshape((len(train),1))
	x_val, y_val = val.as_matrix()[:,1:],val.as_matrix()[:,0].reshape((len(val),1))
	x_test, y_test = test.as_matrix()[:,1:],test.as_matrix()[:,0].reshape((len(test),1))

	# One-hot encode target
	num_classes = 10

	y_train=(np.arange(num_classes) == y_train[:,]).astype(np.float32)
	y_val=(np.arange(num_classes) == y_val[:,]).astype(np.float32)
	y_test=(np.arange(num_classes) == y_test[:,]).astype(np.float32)

	return [(x_train,y_train),(x_val,y_val),(x_test,y_test)]

datasets = create_datasets()

class Batch(object):
    def __init__(self,x,y,batch_size):
        self.x = x
        self.y = y
        self.cursor = 0
        self.batch_size = batch_size
        self.size = len(x)
        assert len(x) % batch_size == 0
    def next_batch(self):
        xr=self.x[self.cursor:self.cursor+self.batch_size,:]
        yr=self.y[self.cursor:self.cursor+self.batch_size,:]
        self.cursor = (self.cursor+self.batch_size)%self.size
        return xr,yr

def get_train():
	return datasets[0]

def get_validation():
	return datasets[1]

def get_test():
	return datasets[2]


def get_trainer():
	x_train,y_train=datasets[0]
	b = Batch(x_train,y_train,batch_size)
	return b

def accuracy(y,y_hat):
    return np.sum((np.argmax(y,axis=1) == np.argmax(y_hat,axis=1)).astype(np.float32))/len(y)


