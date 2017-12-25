from optimizers import *
import numpy as np
from math import sqrt
from random import random,gauss
from time import time
class Regression(object):
    def __init__(self):
        self.optimizer = None
        self.trainingSet = []
        self.testSet = []

    def readData(self,filename):
        #preproccessing
        maxVal = [-1000]*5
        minVal = [1000]*5
        data = []
        with open(filename) as f:
            for line in f:
                features = line.strip().split(',')
                for ind,item in enumerate(features[:-1]):
                    if float(item)>maxVal[ind]:
                        maxVal[ind]=float(item)
                    if float(item)<minVal[ind]:
                        minVal[ind]=float(item)
                data.append(features)

            for features in data:
                for ind, item in enumerate(features[:-1]):
                    features[ind] = (float(item)-minVal[ind])/(maxVal[ind]-minVal[ind])
                if random()>0.2:
                    self.trainingSet.append(np.array(features,dtype=float))
                else:
                    self.testSet.append(np.array(features,dtype=float))


    def getOptimizer(self,optimizer):
        self.optimizer = optimizer

    def run(self):
        start = time()
        o2_weights,o1_weights,bias = self.optimizer.update(data=self.trainingSet, o2_weights=np.ones(5)/10, o1_weights=np.ones(5)/10, bias=0.1)
        #prediction
        error = 0
        for instance in self.testSet:
            features = instance[0:-1]
            groundTruth = instance[-1]
            error+=(groundTruth-o2_weights.dot(features*features)-o1_weights.dot(features)-bias)**2
        averageError = sqrt(error/(len(self.testSet)+0.0))
        end = time()
        print 'Second Order Weights',o2_weights
        print 'One Order Weights', o1_weights
        print 'bias', bias
        print 'Task finished.  Optimizer: %s,  Average Error: %.4f' % (self.optimizer.name, averageError)
        print "Run time: %f s" % (end - start)
        print '='*80


if __name__ == '__main__':
    #generate data
    # o3_weights = np.random.rand(5) * 5
    # o2_weights = np.random.rand(5)*5
    # o1_weights = np.random.rand(5)*5
    # bias = random()*5
    # print 'Original Third Order Weights', o3_weights
    # print 'Original Second Order Weights', o2_weights
    # print 'Original One Order Weights', o1_weights
    # print 'Original bias', bias
    #
    # samples = []
    # for i in range(10000):
    #     sample = np.random.rand(5)*3
    #     y = o3_weights.dot(sample*sample*sample)+o2_weights.dot(sample*sample)+o1_weights.dot(sample)+bias+gauss(3,3)
    #     line = list(np.array(sample,dtype='str'))
    #     line.append(str(y))
    #     samples.append(','.join(line)+'\n')
    # with open('data.txt','w') as f:
    #     f.writelines(samples)

    #SGD
    task1 = Regression()
    task1.readData('data.txt')
    sgd = SGD()
    sgd.readParameters('config.conf')
    task1.getOptimizer(sgd)
    task1.run()
    #BGD
    task2 = Regression()
    task2.readData('data.txt')
    bgd = BGD()
    bgd.readParameters('config.conf')
    task2.getOptimizer(bgd)
    task2.run()
    #Momentum
    task2 = Regression()
    task2.readData('data.txt')
    momentum = Momentum()
    momentum.readParameters('config.conf')
    task2.getOptimizer(momentum)
    task2.run()