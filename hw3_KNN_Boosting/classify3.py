# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 11:29:12 2016

@author: mygao
"""
import os
import argparse
import sys
import pickle
import numpy

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor, InstanceKnn

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

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
            ####label is a string, feature_vector is a list, instances is a list
           
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
    parser.add_argument("--online-learning-rate", type =float, help=  "The learning rate for perceptton",
                        default=1.0)
    parser.add_argument("--online-training-iterations", type = int, help=  "The number of training iternations for online methods.",
                        default=5)
    parser.add_argument("--pegasos-lambda", type = float, help=  "The regularization parameter for Pegasos.",
                        default=1e-4)                
    parser.add_argument("--knn", type = int, help=  "The value of K for KNN classification.",
                        default=5)       
    parser.add_argument("--num-boosting-iterations", type = int, help=  "The value of boosting iteratons to run.",
                        default=10)    
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

####################
def EuclideanDistance(feature_vector1, feature_vector2, maxFeature):
    x1 = numpy.zeros(maxFeature)
    x2 = numpy.zeros(maxFeature)
    for f in feature_vector1._data:
        x1[f[0]-1] = f[1]
    for f in feature_vector2._data:
        x2[f[0]-1] = f[1]
    dist = numpy.linalg.norm(x1-x2)
    #print dist
    return dist

 
def knn_GetNeighbors(instances, testInstance, k, maxFeature):
    distances = []
    for e in instances:
        dist0 = EuclideanDistance(e._feature_vector, testInstance._feature_vector, maxFeature)
        Dist = InstanceKnn(dist0,e._feature_vector,e._label)
        distances.append(Dist)
    sortedDist = sorted(distances, key=lambda x : x._dist)
    neighbors = []
    for x in range(k):
        mem = Instance(sortedDist[x]._feature_vector,sortedDist[x]._label)
        neighbors.append(mem)
    return neighbors
       
def dwKnn_GetNeighbors(instances, testInstance, k, maxFeature):
    distances = []
    for e in instances:
        dist0 = EuclideanDistance(e._feature_vector, testInstance._feature_vector, maxFeature)
        Dist = InstanceKnn(dist0,e._feature_vector,e._label)
        distances.append(Dist)
    sortedDist = sorted(distances, key=lambda x : x._dist)
    neighbors = []
    for x in range(k):
        mem = Instance(sortedDist[x]._dist, sortedDist[x]._feature_vector,sortedDist[x]._label)
        neighbors.append(mem)
    return neighbors

 
def knn_GetLabel(neighbors):
    classVotes = {}
    for e in neighbors:
        response = str(e._label)
        #print type(response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #print classVotes
    sortedVotes = sorted(classVotes, key=classVotes.get, reverse=True)     
    #print sortedVotes
    pred = sortedVotes[0]
    return int(pred)

def dwKnn_GetLabel(neighbors):
    classVotes = {}
    for e in neighbors:
        response = str(e._label)
        if response in classVotes:
            classVotes[response] += 1.0/(e._dist**2+1)
        else:
            classVotes[response] = 1.0/(e._dist**2+1)
    sortedVotes = sorted(classVotes, key=classVotes.get, reverse=True)     
    pred = sortedVotes[0]
    return int(pred)


##################
class knn(Predictor):
    def train(self, instances, k):
        self._instances = instances
        self._k= k
        maxFeature = ComputeMaxFeature(instances)
        self._maxFeature = maxFeature
               
    def predict(self,instance):
        neighbors = knn_GetNeighbors(self._instances,instance, self._k,self._maxFeature)
        label = knn_GetLabel(neighbors)
        return label        
###########################            
class distanceWeighted_knn(Predictor):
    def train(self, instances, k):
        self._instances = instances
        self._k= k
        maxFeature = ComputeMaxFeature(instances)
        self._maxFeature = maxFeature      

    def predict(self,instance):
        neighbors = dwKnn_GetNeighbors(self._instances,instance, self._k,self._maxFeature)
        label = dwKnn_GetLabel(neighbors)
        return label


###############
### this function finds the x_{kj} for every sample, j feature
def getFeatureValues(instances, feature_index):
    xkj = []    # rename 'xkj' to 'instancesFeature'
    for e in instances:
        val = 0
        for f in e._feature_vector._data:
            if f[0] == feature_index:
                val = f[1]
        xkj.append(val)
    return xkj      # rename 'xkj' to 'instancesFeature'

######## this function finds the h_{j,c} value given x_{jk}    
def hjc(xkjList, cutoff):   # rename 'hjc' to 'getStumpLabel', '' to ''
    """
    greater = []    
    for e in xkjList:
        if e > cutoff:
            greater.append(1)
        else:
            greater.append(0)
    return greater 
    """
    # list comprehension is just a faster way to construct an object
    return [1 if e > cutoff else 0 for e in xkjList]

#### this function finds the best cutoff c in feature j
def htj(instances, weights, feature_index):     # rename 'htj' to 'getCutoff'
    xkj = getFeatureValues(instances, feature_index)    # rename 'xkj' to 'instancesFeature'
    xkj_sorted = sorted(xkj)                            # rename 'xkj_sorted' to 'instancesFeature_sorted'
    """
    cutoffList = []
    for i in range(sampleSize-1):
        c = 0.5*(xkj_sorted[i]+xkj_sorted[i+1])
        cutoffList.append(c)    
    """
    cutoffList = [ 0.5*(xkj_sorted[i] + xkj_sorted[i+1]) for i in range(sampleSize-1)] 
    cutoffList = list(set(cutoffList))
    error = float("inf")
    res = []
    for c in range(len(cutoffList)):
        cand_err = 0
        hjcx = hjc(xkj, cutoffList[c])  # rename 'hjc' to 'getStumpLabel', 'hjcx' to 'stumpLabel'
        for i in range(sampleSize):
            cand_err += weights[i] * int(str(hjcx[i]) != str(instances[i]._label))  # rename 'hjc' to 'getStumpLabel', 'hjcx' to 'stumpLabel'
        if cand_err < error:
            error = cand_err
            res = [feature_index, cutoffList[c], error, hjcx]   # rename 'hjcx' to 'stumpLabel'
    return res
      
##########this function finds the best j,c         
def ht(instances, weights):     # rename 'ht' to 'updateWeights'
    error = float("inf")
    result =[]
    for j in range(maxFeature):
        candidate = htj(instances, weights, j+1)
        if candidate[2] < error:
            result = candidate
            error = candidate[2]
    return result
            
# convert integer labels when needed and codes more reader friendly  
def labelConvert(label):
    if label == 1 or label == '1':
        return 1
    if label == 0 or label == '0':
        return -1
    if label == -1 or label == '-1':
        return 0

#############################
class adaboost(Predictor):
    def __init__(self):
        #self._weights = numpy.ones((sampleSize, T)) * (1.0/sampleSize)
        self._weights = [1.0/sampleSize] * sampleSize
        self._res = []
        
    def train(self, instances):
        import math
        for t in range(iterations):
            everyth = ht(instances, self._weights)  # rename 'ht' to 'updateWeights', 'everyth' to 'newWeights'
            eps = everyth[2]
            htv = everyth[3]   # rename 'htv' to 'stumpLabel'
            if eps <= 0.000001:
                break
            else:
                if eps == 1:
                    alp = -float("inf")
                else:
                    alp = 0.5 * math.log((1-eps)/eps)
                
                Dt = [0]*sampleSize
                for i in range(sampleSize): 
                    lab = labelConvert(int(str(instances[i]._label)))
                    Dt[i] = self._weights[i] * numpy.exp(-alp * lab * labelConvert(htv[i]))  # rename 'htv' to 'stumpLabel' 
                                
                # you can do this if you like :) 
                #Dt = [self._weights[i] * numpy.exp(-alp * labelConvert(str(instances[i]._label)) * labelConvert(htv[i])) for i in range(sampleSize)]
                Dtsum = sum(Dt)
                self._weights = [x/Dtsum for x in Dt]
            
            self._res.append( [alp, everyth[0], everyth[1], everyth[2]] ) # rename 'everyth' to 'newWeights'
        return self._res
        
    def predict(self, instance):
        cand0 = 0
        cand1 = 0
        #print [self._res[0][0], self._res[0][1], self._res[0][2]]
        for i in range(len(self._res)):
            alp = self._res[i][0]
            feature_index = self._res[i][1]
            cutoff = self._res[i][2]
            ind = 0
            for f in instance._feature_vector._data:
                if f[0] == feature_index and f[1] > cutoff:
                    ind = 1
            
            if ind == 0:
                cand0 = cand0 + alp
            else:
                cand1 = cand1 + alp
                
        if cand0 >= cand1:
            return 0
        else:             
            return 1
        

def train(instances, algorithm, k):
    # TODO Train the model using "algorithm" on "data"
    # TODO This is where you will add new algorithms that will subclass Predictor
     
    if algorithm == "knn":
        sol = knn()
        sol.train(instances,k)
        return sol  
        
    if algorithm == "distance_knn":
         sol = distanceWeighted_knn()
         sol.train(instances, k)
         return sol 

    if algorithm == "adaboost":
         sol = adaboost()
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
  
            
def main():
    args = get_args()
    global maxFeature
    global sampleSize
    global iterations
    iterations = args.num_boosting_iterations

    if args.mode.lower() == "train":
        # Load the training data.
        instances = load_data(args.data)
        maxFeature = ComputeMaxFeature(instances)
        sampleSize = len(instances)
        # Train the model.
        predictor = train(instances, args.algorithm, args.knn)
        #print predictor
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


#python classify3.py --mode train --algorithm knn --model-file speech.mc.knn.model --data speech.mc.train
#python classify3.py --mode test --model-file speech.mc.knn.model --data speech.mc.dev --predictions-file speech.mc.dev.predictions

#python classify3.py --mode train --algorithm distance_knn --model-file easy.distance_knn.model --data easy.train
#python classify3.py --mode test --model-file easy.distance_knn.model --data easy.dev --predictions-file easy.dev.predictions

#python classify3.py --mode train --algorithm adaboost --model-file easy.adaboost.model --data easy.train
#python classify3.py --mode test --model-file easy.adaboost.model --data easy.dev --predictions-file easy.dev.predictions


#python classify3.py --mode train --algorithm adaboost --model-file easy.adaboost.model --data easy.train --num-boosting-iterations 10
#python classify3.py --mode test --model-file easy.adaboost.model --data easy.dev --predictions-file easy.dev.predictions

#python compute_accuracy.py easy.dev easy.dev.predictions

