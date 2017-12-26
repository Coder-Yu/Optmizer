from config import *
from random import choice,shuffle
class Optimizer(object):
    def __init__(self):
        self.lastLoss=0
        self.loss=0
        self.iteration=0
        self.lossRecord = []
    def readParameters(self,filename):
        pass
    def update(self,**params):
        pass
    def run(self,**params):
        pass

class SGD(Optimizer):
    "Stochastic Gradient Descent Method"
    def readParameters(self,filename):
        settings = LineConfig(Config(fileName=filename)['SGD'])
        self.epoch = int(settings['-epoch'])
        self.step = float(settings['-step'])
        self.reg = float(settings['-reg'])
        self.name = 'SGD'

    def update(self,**params):
        data = params['data']
        o2_weights = params['o2_weights']
        o1_weights = params['o1_weights']
        bias = params['bias']
        while self.iteration <= self.epoch:
            self.lastLoss = self.loss
            self.loss = 0
            for i in range(len(data)):
                instance = choice(data)
                groundTruth = instance[-1]
                features = instance[0:-1]
                error = (o2_weights.dot(features*features)+o1_weights.dot(features)+bias-groundTruth)
                o2_weights-=self.step*(error*features*features+self.reg*o2_weights)
                o1_weights -= self.step * (error * features + self.reg*o1_weights)
                bias-=self.step*(error+self.reg*bias)
                self.loss+=0.5*error**2
            if self.iteration>=1:
                print 'epoch: %d, training loss: %.4f, delta loss: %.4f, step size: %.4f' \
                      %(self.iteration, self.loss, self.lastLoss-self.loss, self.step)
            self.iteration+=1
            self.loss+=o2_weights.dot(o2_weights)+o1_weights.dot(o1_weights)+bias**2
            self.lossRecord.append(self.loss)
        return o2_weights,o1_weights,bias

class BGD(Optimizer):
    "Batch Gradient Descent Method"

    def readParameters(self, filename):
        settings = LineConfig(Config(fileName=filename)['BGD'])
        self.epoch = int(settings['-epoch'])
        self.step = float(settings['-step'])
        self.reg = float(settings['-reg'])
        self.batch = int(settings['-batch'])
        self.name = 'BGD'

    def update(self, **params):
        data = params['data']
        o2_weights = params['o2_weights']
        o1_weights = params['o1_weights']
        bias = params['bias']
        while self.iteration <= self.epoch:
            self.lastLoss = self.loss
            self.loss = 0

            for i in range(len(data)/self.batch):
                batchData = data[i*self.batch:(i+1)*self.batch]
                o2_gradients = 0
                o1_gradients = 0
                bias_gradients = 0
                for instance in batchData:
                    groundTruth = instance[-1]
                    features = instance[0:-1]
                    error = (o2_weights.dot(features * features) + o1_weights.dot(features) + bias - groundTruth)
                    o2_gradients += error*(features*features)+ self.reg * o2_weights
                    o1_gradients += error*features + self.reg * o1_weights
                    bias_gradients += error + self.reg * bias
                    self.loss += 0.5*error**2

                o2_weights -= self.step * (o2_gradients)
                o1_weights -= self.step * (o1_gradients)
                bias -= self.step * bias_gradients

            if self.iteration >= 1:
                print 'epoch: %d, training loss: %.4f, delta loss: %.4f, step size: %.4f' \
                      % (self.iteration, self.loss, self.lastLoss - self.loss, self.step)
            self.iteration += 1
            self.loss += o2_weights.dot(o2_weights) + o1_weights.dot(o1_weights) + bias ** 2
            self.lossRecord.append(self.loss)
            shuffle(data)
        return o2_weights, o1_weights , bias

class Momentum(Optimizer):
    "Stochastic Gradient Descent Method With Momentum"
    def readParameters(self,filename):
        settings = LineConfig(Config(fileName=filename)['Momentum'])
        self.epoch = int(settings['-epoch'])
        self.step = float(settings['-step'])
        self.gamma = float(settings['-gamma'])
        self.reg = float(settings['-reg'])
        self.name = 'Momentum'
        self.momentum_o2 = 0
        self.momentum_o1 = 0
        self.momentum_b = 0


    def update(self,**params):
        data = params['data']
        o2_weights = params['o2_weights']
        o1_weights = params['o1_weights']
        bias = params['bias']
        while self.iteration <= self.epoch:
            self.lastLoss = self.loss
            self.loss = 0
            for i in range(len(data)):
                instance = choice(data)
                groundTruth = instance[-1]
                features = instance[0:-1]
                error = (o2_weights.dot(features*features)+o1_weights.dot(features)+bias-groundTruth)
                o2_gradient = self.step*((error*features*features+self.reg*o2_weights)+0.9*self.momentum_o2)
                o1_gradient = self.step *((error * features + self.reg*o1_weights)+0.9*self.momentum_o1)
                bias_gradient = self.step*((error+self.reg*bias)+0.9*self.momentum_b)
                self.momentum_o2 = o2_gradient
                self.momentum_o1 = o1_gradient
                self.momentum_b = bias_gradient
                o2_weights-= o2_gradient
                o1_weights -= o1_gradient
                bias-= bias_gradient
                self.loss+=0.5*error**2
            if self.iteration>=1:
                print 'epoch: %d, training loss: %.4f, delta loss: %.4f, step size: %.4f' \
                      %(self.iteration, self.loss, self.lastLoss-self.loss, self.step)
            self.iteration+=1
            self.loss += o2_weights.dot(o2_weights) + o1_weights.dot(o1_weights) + bias ** 2
            self.lossRecord.append(self.loss)
        return o2_weights,o1_weights,bias


class NAG(Optimizer):
    "Stochastic Gradient Descent Method With Momentum"
    def readParameters(self,filename):
        settings = LineConfig(Config(fileName=filename)['NAG'])
        self.epoch = int(settings['-epoch'])
        self.step = float(settings['-step'])
        self.gamma = float(settings['-gamma'])
        self.reg = float(settings['-reg'])
        self.name = 'NAG'
        self.momentum_o2 = 0
        self.momentum_o1 = 0
        self.momentum_b = 0


    def update(self,**params):
        data = params['data']
        o2_weights = params['o2_weights']
        o1_weights = params['o1_weights']
        bias = params['bias']
        while self.iteration <= self.epoch:
            self.lastLoss = self.loss
            self.loss = 0
            for i in range(len(data)):
                instance = choice(data)
                groundTruth = instance[-1]
                features = instance[0:-1]
                error = ((o2_weights-self.gamma*self.momentum_o2).dot(features*features)+
                         (o1_weights-self.gamma*self.momentum_o1).dot(features)+(bias-self.gamma*self.momentum_b)-groundTruth)
                o2_gradient = self.step*((error*features*features+self.reg*o2_weights)+self.gamma*self.momentum_o2)
                o1_gradient = self.step *((error * features + self.reg*o1_weights)+self.gamma*self.momentum_o1)
                bias_gradient = self.step*((error+self.reg*bias)+self.gamma*self.momentum_b)
                self.momentum_o2 = o2_gradient
                self.momentum_o1 = o1_gradient
                self.momentum_b = bias_gradient
                o2_weights-= o2_gradient
                o1_weights -= o1_gradient
                bias-= bias_gradient
                self.loss+=0.5*error**2
            if self.iteration>=1:
                print 'epoch: %d, training loss: %.4f, delta loss: %.4f, step size: %.4f' \
                      %(self.iteration, self.loss, self.lastLoss-self.loss, self.step)
            self.iteration+=1
            self.loss += o2_weights.dot(o2_weights) + o1_weights.dot(o1_weights) + bias ** 2
            self.lossRecord.append(self.loss)
        return o2_weights,o1_weights,bias