import matplotlib.pyplot as plt
import numpy as np

def sigmoid():

    x=np.linspace(-10,10,100)
    y=np.exp(-x)/pow((1+np.exp(-x)),2)

    plt.plot(x,y)
    plt.show()

def linear():
    x=np.linspace(0,10)
    y=np.linspace(1,1)


    plt.plot(x,y)

    x=np.linspace(-10,0)
    y=np.linspace(0,0)
    plt.plot(x,y)
    
    plt.show()

def piece():
    x=np.linspace(-10,-1)
    y=np.linspace(0.05,0.05)
    plt.plot(x,y)
    x=np.linspace(-1,1)
    y=np.linspace(1,1)
    plt.plot(x,y)
    x=np.linspace(1,10)
    y=np.linspace(0.05,0.05)
    plt.plot(x,y)
    plt.show()


def swish():
    x=np.linspace(-10,10)
    y=(5*np.exp(-5*x))/pow((1+np.exp(-5*x)),2)


    plt.plot(x,y)
    plt.show()
def elu():
    x=np.linspace(0,10)
    y=np.linspace(1,1)
    plt.plot(x,y)
    x=np.linspace(-10,0)
    y=0.5*np.exp(x)
    print x 
    print y
    plt.plot(x,y)
    plt.show()

linear()
sigmoid()
piece()
swish()
elu()
