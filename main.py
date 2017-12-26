from optimizers import *
import numpy as np
from math import sqrt
from random import random,gauss
from time import time

featuresCount = 8
class Regression(object):
    def __init__(self):
        self.optimizer = None
        self.trainingSet = []
        self.testSet = []

    def readData(self,filename):
        #preproccessing
        maxVal = [-1000]*featuresCount
        minVal = [1000]*featuresCount
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
        o2_weights,o1_weights,bias = self.optimizer.update(data=self.trainingSet, o2_weights=np.ones(featuresCount)/10, o1_weights=np.ones(featuresCount)/10, bias=0.1)
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
    # o3_weights = np.random.rand(5)*5
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
    #     y = o3_weights.dot(sample*sample*sample)+o2_weights.dot(sample*sample)+o1_weights.dot(sample)+bias+gauss(10,3)
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
    task3 = Regression()
    task3.readData('data.txt')
    momentum = Momentum()
    momentum.readParameters('config.conf')
    task3.getOptimizer(momentum)
    task3.run()
    #NAG
    task4 = Regression()
    task4.readData('data.txt')
    nag = NAG()
    nag.readParameters('config.conf')
    task4.getOptimizer(nag)
    task4.run()


    #plot

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns


    def drawLine(x, y, labels, xLabel, yLabel, title):
        f, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

        # f.tight_layout()
        #sns.set(style="darkgrid")

        palette = [ 'red', 'green', 'purple', '#0F90FF','pink','blue', 'orange',]


        for ydata, lab, c in zip(y, labels, palette):
            ax.plot(x, ydata, color=c, label=lab,linewidth=3)
        # ind = np.arange(0, 60, 10)
        # ax.set_xticks(ind)
        # ax.set_xticklabels(x)
        ax.set_xlabel(xLabel, fontsize=22)
        ax.set_ylabel(yLabel, fontsize=22)
        ax.tick_params(labelsize=20)
        # ax.tick_params(axs='y', labelsize=20)

        ax.set_title(title, fontsize=24)
        #ax.set_ylim(30000,140000)
        plt.grid(True)
        handles, labels1 = ax.get_legend_handles_labels()

        # ax[i].legend(handles, labels1, loc=2, fontsize=20)
        # ax.legend(loc=2,
        #        ncol=6,  borderaxespad=0.,fontsize=20)
        # ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=20)
        ax.legend(loc='upper right', fontsize=20, shadow=True)
        plt.show()
        plt.close()


    labels = ['SGD', 'BGD', 'Momentum','NAG' ]
    xlabel = 'Iteration'
    ylabel = 'Loss'
    x = [i for i in range(sgd.epoch+1)]
    y = [sgd.lossRecord,bgd.lossRecord,momentum.lossRecord,nag.lossRecord]
    drawLine(x, y, labels, xlabel, ylabel, 'Comparison of Different Optimization Methods')