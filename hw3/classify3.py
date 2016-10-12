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
import math

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

 
def GetNeighbors(instances, testInstance, k, maxFeature):
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
       
def GetNeighbors2(instances, testInstance, k, maxFeature):
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

 
def GetLabel(neighbors):
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

def GetLabel2(neighbors):
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
        neighbors = GetNeighbors(self._instances,instance, self._k,self._maxFeature)
        label = GetLabel(neighbors)
        return label        
###########################            
class distance_knn(Predictor):
    def train(self, instances, k):
        self._instances = instances
        self._k= k
        maxFeature = ComputeMaxFeature(instances)
        self._maxFeature = maxFeature      

    def predict(self,instance):
        neighbors = GetNeighbors2(self._instances,instance, self._k,self._maxFeature)
        label = GetLabel2(neighbors)
        return label


###############
### this function finds the x_{kj} for every sample, j feature
def findValue(instances, feature_index):
    J = feature_index
    xkj = []
    for e in instances:
        val = 0
        for f in e._feature_vector._data:
            if f[0]-1 == J:
                val = f[1]
        xkj.append(val)
   # print xkj
    return xkj
    
########this function finds the h_{j,c} value given x_{jk}    
def hjc(xkjList, cutoff):
    C = cutoff
    greater = []
    for e in xkjList:
        if e > C:
            greater.append(1)
        else:
            greater.append(0)
    return greater

#### this function finds the best c in feature j
def htj(instances, weights, feature_index):
   # y = []
    J = feature_index
    xkj = findValue(instances, feature_index)
    #for i in range(sampleSize):
     #   mem = Instance(xkj[i-1],sampleSize[i-1]._label)
      #  y.append(mem)
    #sortedy = sorted(y, key=lambda x : x._feature_vector)   
    sortedy = sorted(xkj) 
    Clist = []
    for i in range(sampleSize-1):
        c = 0.5*(sortedy[i-1]+sortedy[i])
        Clist.append(c)
    Cset = set(Clist)
    Clist = list(Cset)
    #print Clist
    ht = float("inf")
    res = []
    for l in range(len(Clist)):
        cand =0
        hjcx = hjc(xkj, Clist[l-1])
        for i in range(sampleSize):
            cand = cand + weights[i-1] * int(str(hjcx[i-1]) != str(instances[i-1]._label))
        if cand < ht:
            ht = cand
            res = [J, Clist[l-1], ht, hjcx]
    return res

##########this function finds the best j,c         
def ht(instances, weights):     
    ht = float("inf")
    result0 =[]
    for j in range(maxFeature):
        cand = htj(instances, weights, j)
        if cand[2] < ht:
            result0 = cand
            ht = cand[2]
#    print [result0[0], result0[1], result0[2]]
    return result0


#############################
class adaboost(Predictor):
    def __init__(self):
        #self._weights = numpy.ones((sampleSize, T)) * (1.0/sampleSize)
        self._weights = []
        self._res = []
        
    def train(self, instances):
        self._weights = [1.0/sampleSize]*sampleSize
        for t in range(iterations):
            everyth = ht(instances, self._weights)
            eps = everyth[2]
            #print eps
            htv = everyth[3]
            if eps <= 0.000001:
                break
            else:
                if eps == 1:
                    alp = -float("inf")
                else:
                    alp = 0.5 * math.log((1-eps)/eps)
                    #print alp
                Dt = [0]*sampleSize
                for i in range(sampleSize): 
                    lab = int(str(instances[i-1]._label))
                    Dt[i-1] = self._weights[i-1]* numpy.exp(-alp * (lab*2-1)*htv[i-1])
            
                Dtsum = sum(Dt)
                self._weights = [x/Dtsum for x in Dt]
            
            ###everyth[0] =J, everyth[1] =c
            store = [alp, everyth[0],everyth[1], everyth[2]]
            self._res.append(store) 
        return self._res
        
    def predict(self,instance):
        #htx = []
        cand0 = 0
        cand1 =0
        #print [self._res[0][0], self._res[0][1], self._res[0][2]]
        for i in range(len(self._res)):
            alp = self._res[i-1][0]
            J = self._res[i-1][1]
            C = self._res[i-1][2]
           #h = self._res[i-1][3]
            ind = 0
            m = 0
            for f in instance._feature_vector._data:
                if f[0]-1 ==J and f[1] >C:
                    ind = 1
                    #print [C, f[1]]
                    m = m +1
                
            #print ind
            #htx.append(ind)
            if ind == 0:
                #print [C, f[1]]
                cand0 = cand0 + alp
            else:
                cand1 = cand1 + alp
        print [cand0,cand1]
        if cand0 >=cand1:
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
         sol = distance_knn()
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


#python classify3.py --mode train --algorithm adaboost --model-file easy.adaboost.model --data easy.train --num-boosting-iterations 5
#python classify3.py --mode test --model-file easy.adaboost.model --data easy.dev --predictions-file easy.dev.predictions

#python compute_accuracy.py easy.dev easy.dev.predictions

