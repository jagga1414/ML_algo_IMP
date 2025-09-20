

import numpy as np 

class LinearRegression():
    def __init__(self,numFeatures):
        self.W = np.random.rand(numFeatures)
        self.B = 0

    def fit(self,x,y,lr,itr):

        n_samples = x.shape

        for i in range(itr):
            ypred = np.dot(x,self.W) + self.B

            loss = (1/n_samples)* np.sum(np.square(y-ypred),axis=-1)

            dw = (-2/n_samples)*np.dot(x.T,(y-ypred))
            db = (-2/n_samples)*np.sum(y-ypred,axis=-1)

            self.W = self.W - lr*dw
            self.B = self.B - lr*db

    def predict(self,X):

        return np.dot(X,self.W) + self.B        
    