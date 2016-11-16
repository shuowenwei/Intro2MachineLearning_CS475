# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:56:38 2016

@author: mygao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:11:16 2016

@author: mygao

@email: mygao90@gmail.com
"""

import os
import argparse
import sys
import pickle
import numpy
#import scipy.stats
import math
import scipy.stats

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

def load_data(filename):
    instances = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")
                
            #######convert class label 0 to -1
           # if int_label == 0:
            #    int_label = -1

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
           
            ####### deal with feature vector, into (index,value)
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    # TODO This is where you will add new command line options
    parser.add_argument("--clustering-training-iterations", type =int, help=  "The number of clustering iterations",
                        default=10)
    parser.add_argument("--num-clusters", type = int, help=  "The number of clusters in Naive Bayes clustering.",
                        default=3)
    parser.add_argument("--cluster-lambda", type = float, help=  "The value of lambda in lambda-means.",
                        default=0.0)                
   
    ##########                        
    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")

##############
# compute the mean of all instances
def data_mean(instances):
    sumx = [0]*maxFeature
    for e in instances:
        for f in e._feature_vector._data:
            sumx[f[0]-1] += f[1]
    mu = [a/len(instances) for a in sumx]
    return mu
            
# euclidean distance between x and mu     
def EuclideanDist(feature_vector1,mu, num_feature):
    x1 = numpy.zeros(num_feature)
    x2 = numpy.asarray(mu)
    for f in feature_vector1._data:
        x1[f[0]-1] = f[1]
    dist = numpy.linalg.norm(x1-x2)
    return dist

# compute the cluster mean
def findmu(instances,r,k):
    sumk= [0] * maxFeature
    mem = 0
    for e in range(len(instances)):
        if r[e] == k:
            mem += 1
            for f in instances[e]._feature_vector._data:
                sumk[f[0]-1] += f[1]
    if mem >0:
        mu = [ a/float(mem) for a in sumk]
        return mu
    else:
        print k                   
                      
##################
class lambda_means(Predictor):
    def __init__(self):
        self._mu = [] 
        
    def train(self, instances):
        # cluster init
        K=1
        self._mu.append(data_mean(instances))
        r = [0]*len(instances)    
        for i in range(clustering_training_iterations):
            ### E-step
            for e in range(len(instances)):
                distmin = float("inf")
                for k in range(K):
                    dist = EuclideanDist(instances[e]._feature_vector, self._mu[k], maxFeature)**2
                    if dist < distmin:
                        r[e] =k
                        distmin = dist
               # print distmin
                if distmin > cluster_lambda:
                    K += 1
                    r[e] = K-1
                    x1 = numpy.zeros(maxFeature)
                    for f in instances[e]._feature_vector._data:
                        x1[f[0]-1] = f[1]
                    self._mu.append(x1)
            print K
            #print r
            ### M-step
            for k in range(K):
                self._mu[k] = findmu(instances,r,k)
                

    def predict(self,instance):
        res = 0
        distmin = float("inf")
        for k in range(len(self._mu)):
            dist = EuclideanDist(instance._feature_vector, self._mu[k], len(self._mu[k]))
            if dist < distmin:
                res = k
                distmin = dist     
        return res                                  
###########################  

def findvar(instances,y,k,mu):
    sumvar = [0]*maxFeature
    mem = 0.0
    for e in range(len(instances)):
        if y[e] == k:
            mem += 1
            x1 = numpy.zeros(maxFeature)
            for f in instances[e]._feature_vector._data:
                x1[f[0]-1] = f[1]
            for j in range(maxFeature):
                sumvar[j] += (x1[j]-mu[j])**2
    if mem >1:
        var = [a/float(mem-1) for a in sumvar]
        for e in range(maxFeature):
            if var[e] <Sj[e]:
                var[e] = Sj[e]
    else:
        var = Sj    
    return var

def findphi(y,k):
    mem = 0.0
    for i in range(len(y)):
        if y[i] == k:
            mem += 1
    phi = (mem+1)/(len(y)+num_clusters)
    return phi

def lognormpdf(x, mean, var):
    pi = 3.1415926
    res = -0.5*math.log(2*pi*var) -(float(x)-float(mean))**2/(2*var)
    return res


def loglikeli(feature_vector, mu, var, phi):
    x = numpy.zeros(len(mu))
    loglike = 0.0
    for f in feature_vector._data:
        x[f[0]-1] = f[1]
    for i in range(len(mu)):
        loglike += lognormpdf(x[i],mu[i],var[i])
    loglike += math.log(phi)
    return loglike

###########################3          
class nb_clustering(Predictor):
    def __init__(self):
        self._mu = [-1]*num_clusters 
        self._var = [-1]*num_clusters
        self._phi = [-1]*num_clusters
    
    def train(self, instances):
        # cluster init
        K = num_clusters
        y = [0] * len(instances)
        for e in range(len(instances)):
            y[e] = e % K
        for k in range(K):
            self._mu[k] = findmu(instances,y,k)
            self._var[k] = findvar(instances,y,k,self._mu[k])
            self._phi[k] = findphi(y,k)
        
        ####
        for i in range(clustering_training_iterations):
            ### E-step
            for e in range(len(instances)):
                likemax = -float("inf")
                for k in range(K):
                    like = loglikeli(instances[e]._feature_vector, self._mu[k], self._var[k], self._phi[k])
                    if like > likemax:
                        y[e] =k
                        likemax = like 
                #print likemax
            #print y
        # M step
            for k in range(K):
                self._phi[k] = findphi(y,k)
                self._mu[k] = findmu(instances,y,k)
                self._var[k] = findvar(instances,y,k,self._mu[k])
                #print self._mu[k]
                #print self._var[k]

                
    def predict(self,instance):
        res = 0 
        likemax = -float("inf")
        for k in range(len(self._mu)):
            like = loglikeli(instance._feature_vector, self._mu[k], self._var[k], self._phi[k])
            if like > likemax:
                res = k
                likemax = like     
        return res 
            
##########################

def train(instances, algorithm):
    # TODO Train the model using "algorithm" on "data"
    # TODO This is where you will add new algorithms that will subclass Predictor
     
    if algorithm == "lambda_means":
        sol = lambda_means()
        sol.train(instances)
        return sol  
        
    if algorithm == "nb_clustering":
         sol = nb_clustering()
         sol.train(instances)
         return sol 

####################

def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
                
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)

def ComputeMaxFeature(instances):
    maxfeature = 0 
    for e in instances:
        for f in e._feature_vector._data:
            if f[0] > maxfeature:
                maxfeature = f[0]
    return maxfeature 

def ComputeLambda(instances):
    sumlambda = 0   
    for e in instances:
        sumlambda +=  EuclideanDist(e._feature_vector, dataMean, maxFeature)**2
    clusterlambda = sumlambda / len(instances)
    return clusterlambda     

def findSj(instances):
    sumvar = [0]*maxFeature    
    for e in range(len(instances)):
        for f in instances[e]._feature_vector._data:
            sumvar[f[0]-1] += (f[1]-dataMean[f[0]-1])**2
    sj = [a/float(len(instances)-1)*0.01 for a in sumvar]
    return sj
            
def main():
    args = get_args()
    global maxFeature
    global clustering_training_iterations
    global cluster_lambda
    global num_clusters
    global Sj
    global dataMean
    
    if args.mode.lower() == "train":
        # Load the training data.
        instances = load_data(args.data)
        maxFeature = ComputeMaxFeature(instances)
       
        num_clusters = args.num_clusters
        dataMean = data_mean(instances)

        clustering_training_iterations = args.clustering_training_iterations
        if args.cluster_lambda == 0.0 :
            cluster_lambda = ComputeLambda(instances)
        else:
            cluster_lambda = args.cluster_lambda
        print cluster_lambda 
        Sj = findSj(instances)
        # Train the model.
        predictor = train(instances, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        instances = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)

        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()


#python classify4.py --mode train --algorithm lambda_means --model-file speech.lambda_means.model --data speech.train --clustering-training-iterations 1
#python classify4.py --mode test --model-file speech.lambda_means.model --data speech.dev --predictions-file speech.dev.predictions

#python classify4.py --mode train --algorithm nb_clustering --model-file speech.nb_clustering.model --data speech.train --clustering-training-iterations 1
#python classify4.py --mode test --model-file speech.nb_clustering.model --data speech.dev --predictions-file speech.dev.predictions
#python cluster_accuracy.py speech.dev speech.dev.predictions
#python number_clusters.py speech.dev.predictions

