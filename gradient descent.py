import numpy
from scipy import io
class neural_network_layer:

	def __init__(self, W, b, learning_rate=0.1):
		self.learning_rate = learning_rate
		self.W = W
		self.b = b

	def predict(self, x):
		z = numpy.empty(0)
		z = (numpy.dot(self.W, x) + self.b)
		y_pred =  numpy.exp(z)/numpy.sum(numpy.exp(z))
		return y_pred

	def loss(self, y_pred, class_label):
		print(-numpy.log(y_pred[class_label-1]))
		return -numpy.log(y_pred[class_label-1])

	def gradient_descent(self, y_pred, x, class_label):
		gdw = -(y_pred[class_label-1] - y_pred[class_label-1]*y_pred[class_label-1])*x/y_pred[class_label-1]
		print(gdw.shape)
		gdb = -(y_pred[class_label-1] - y_pred[class_label-1]*y_pred[class_label-1])/y_pred[class_label-1]
		self.W[class_label-1] -= self.learning_rate*gdw.ravel()
		self.b[class_label-1] -= self.learning_rate*gdb.ravel()


data = io.loadmat("data.mat")
b = io.loadmat("bias.mat")
W = io.loadmat("weight.mat")
W = (W["softmax_weight"])
b = (b['softmax_bias'])
x = (data['asample'])



W = numpy.reshape(W, [100,20])
b = numpy.reshape(b, [20, 1])

W = W.T


x = numpy.reshape(x, [100,1])


model = neural_network_layer(numpy.copy(W), numpy.copy(b))
y_pred = model.predict(x)

loss = model.loss(y_pred, 1)


model.gradient_descent(y_pred, x, 1)
y_pred = model.predict(x)

loss = model.loss(y_pred, 1)

W_diff = model.W - W
B_diff = model.b - b

increase_w = 0
decrease_w = 0
increase_b = 0
decrease_b = 0
no_change_w = 0
no_change_b = 0

for e in W_diff.ravel():
	if(e > 0.001):
		increase_w += 1
	else:
		if(e < -0.001):
			decrease_w += 1
		else:
			no_change_w += 1

for e in B_diff.ravel():
	if(e > 0.001):
		increase_b += 1
	else:
		if(e < -0.001):
			decrease_b += 1
		else:
			no_change_b += 1 
print("Increased Weights",increase_w, "  decreased Weights" ,decrease_w, "  Weights no changed",no_change_w)
print("Increased biases",increase_b, "  decreased biases",decrease_b,"  biases no changed", no_change_b)
