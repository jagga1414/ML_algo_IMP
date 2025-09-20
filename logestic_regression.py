

import numpy as np 

class LogesticRegression():
    def __init__(self,numFeatures):
        self.W = np.random.rand(numFeatures)
        self.B = 0
    def sigmoid(self,y):

        return (1/(1+np.exp(-y)))
    def fit(self,x,y,lr,itr):

        n_samples = x.shape

        for i in range(itr):
            ypred = self.sigmoid(np.dot(x,self.W) + self.B)

            # loss = (1/n_samples)* np.sum(np.square(y-ypred),axis=-1)
            loss = -np.sum((y*np.log(1e-8+ypred))+(1-y)*np.log(1-ypred+1e-8),axis=-1)/n_samples

            dw = np.dot(x.T,(ypred-y))/n_samples
            # dw = (-2/n_samples)*np.dot(x.T,(y-ypred))
            # db = (-2/n_samples)*np.sum(y-ypred,axis=-1
            db = np.sum(ypred-y)/n_samples

            self.W = self.W - lr*dw
            self.B = self.B - lr*db

    def predict(self,X):

        return self.sigmoid(np.dot(X,self.W) + self.B)     
    