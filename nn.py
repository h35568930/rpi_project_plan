# -*- coding:UTF-8 -*-
import sys
import numpy as np
from numpy import savetxt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model
import random
# command: pythonx.x nn.py 3_3_17.csv

def main(argv):
	# data preprocessing (5 input(Temp, Vdd, Vi, Vq, Freq), 1 output(IRR))
	#data      = np.loadtxt(str(sys.argv[1]), delimiter=',')
	data      = np.loadtxt("3_3_17.csv", delimiter=',')
	dataset   = np.random.permutation(data)
	n_train   = int(len(dataset)*0.8)    # change 0.8 to any value you want in [0, 1]
	train_in  = dataset[:n_train,:5]
	train_out = dataset[:n_train, 5]
	#test_in   = dataset[n_train:,:5]
	#test_out  = dataset[n_train:, 5]

	# Vi, Vq set used in plotting
	Vi_Vq_set   = data[8232, :5].reshape(1,5)
	IRR = data[8232,5].reshape(1,1)
	for i in range(120):
		Vi_Vq_set = np.append(Vi_Vq_set, data[8232+17*(i+1),:5].reshape(1,5), axis=0)
		IRR = np.append(IRR, data[8232+17*(i+1),5].reshape(1,1), axis=0)
	
	# build the neural network
	model = Sequential()

	model.add(Dense(15, input_dim = 5, activation="relu", use_bias=True))
	model.add(Dense(15, activation="relu", use_bias=True))
	model.add(Dense(5, activation="relu", use_bias=True))
	model.add(Dense(1, activation="linear"))
	model.compile(loss="mae", optimizer="nadam", metrics=["accuracy"])

	plot_model(model, show_shapes=True, to_file="model_shape.png")
	# train the model with small batch (update the network for each batch)
	batch_size = 16
	check_point = n_train//10

	for i in range( (n_train+16-n_train%16) // 16):
		"""if ((i+1)*batch_size>n_train):
			batch_in  = train_in[i*batch_size:(i+1)*batch_size-(16-n_train%16), :5] 
			batch_out = train_out[i*batch_size:(i+1)*batch_size-(16-n_train%16)]
		else:"""
		batch_in  = train_in[i*batch_size:(i+1)*batch_size, :5] 
		batch_out = train_out[i*batch_size:(i+1)*batch_size]
		model.fit(batch_in, batch_out, validation_split=0.2, epochs=15, batch_size=batch_size)

	"""# uncomment the following section to check the accuracy while training is still going	
	if (i*batch_size >= 2*check_point):
			predictions = model.predict(test_in)
			test_out    = test_out.reshape(len(test_out), 1)
			error  = (predictions-test_out)/test_out*100
			result = np.append(predictions, test_out, axis=1)
			result = np.append(result, error, axis=1)
			pd.DataFrame(result).to_csv("test.csv", header=["predicted value", "actual value", "error(%)"], index=None)
	print(np.average(abs(error)))"""

	# plot predictions of defferent Vi, Vq set (temp=25, vdd=1.0, freq=38)
	predictions = model.predict(Vi_Vq_set)

	fig = plt.figure()
	ax_actual_surface   = fig.add_subplot(221, projection="3d")
	ax_test_surface     = fig.add_subplot(222, projection="3d")
	ax_actual_scatter   = fig.add_subplot(223, projection="3d")
	ax_test_scatter     = fig.add_subplot(224, projection="3d")
	
	actual_IRR  = np.zeros([11,11])
	predict_IRR = np.zeros([11,11])
	result = np.array([[0,0,0,0.0]])
	for i in range(11):
		for j in range(11):
			index = 11*i + j
			ax_test_scatter.scatter(Vi_Vq_set[index,2], Vi_Vq_set[index,3], predictions[index], color='g')
			ax_actual_scatter.scatter(Vi_Vq_set[index,2], Vi_Vq_set[index,3], IRR[index], color='r')
			actual_IRR[i,j]  = IRR[index]
			predict_IRR[i,j] = predictions[index]
			#print(Vi_Vq_set[index,2], Vi_Vq_set[index,3], predictions[index], IRR[index])
			result = np.append(result, np.array([[Vi_Vq_set[index,2], Vi_Vq_set[index,3], predictions[index,0], IRR[index,0]]]), axis=0)
			
	savetxt("result_relu_20_4.csv", result, delimiter=',')
	X = np.outer(np.linspace(-1, 1, 11), np.ones(11))
	Y = X.T
	ax_test_surface.plot_surface(X, Y, predict_IRR[:], color='g')
	ax_actual_surface.plot_surface(X, Y, actual_IRR[:], color='r')
	plt.show()

if __name__ == "__main__":
	main(sys.argv)