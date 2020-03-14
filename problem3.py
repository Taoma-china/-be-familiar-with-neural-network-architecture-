import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import math
import matplotlib.pyplot as plt

def function(z):
    return math.sin(3*math.pi*z/4)

x = np.linspace(-1,1,200)
function2 = np.vectorize(function)
y=function2(x)

model = Sequential()
model.add(Dense(20,input_dim=1,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='SGD')
model.fit(x,y,epochs=7000,batch_size=20,validation_data=(x,y))

x_predict = np.linspace(-1,1,200)
predictions=model.predict(x_predict)
function3 = np.vectorize(function)
y_real=function3(x_predict)
print np.square(np.transpose(predictions)-y_real)
print np.amax(abs(np.transpose(predictions)-y_real))

plt.subplot(2,1,1)
plt.scatter(x,y,s=1)
plt.title('y=$sin(3pi*x/4)$')
plt.ylabel('Real y')
plt.subplot(2,1,2)
plt.scatter(x_predict,predictions,s=1)
plt.xlabel('x')
plt.ylabel('Approximated y')
plt.show()
