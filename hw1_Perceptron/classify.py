import os
import argparse
import sys
import pickle

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
    parser.add_argument("--online-learning-rate", type =float, help=  "The learning rate for perceptton",
                        default=1.0)
    parser.add_argument("--online-training-iterations", type = int, help=  "The number of training iternations for online methods.",
                        default=5)
                        
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
    
def update(w, atheta, label, feature_vector):
    if label == '0':
        for f in feature_vector._data:
           w[f[0]-1] = w[f[0]-1] - atheta* f[1]
        return w
    if label == '1':
        for f in feature_vector._data:
            w[f[0]-1] = w[f[0]-1] + atheta* f[1]
        return w 
        
def decision(w, feature_vector):
    res = 0
    for f in feature_vector._data:
        res = res + w[f[0]-1] * f[1]
    if res >= 0:
        return 1
    else:
        return 0

class perception_train(Predictor):
    def __init__(self):
        self._weights = [0]* maxFeature
        
    def train(self, instances):
        I = iterations
        for i in range(I): 
            for e in instances:
                if str(decision(self._weights, e._feature_vector)) != str(e._label):
                    self._weights = update(self._weights, atheta, str(e._label), e._feature_vector)     
            #print(self._weights)
    
    def predict(self, instance): 
        res = 0             
        for f in instance._feature_vector._data:            
            if f[0] <= len(self._weights):
                res = res + self._weights[f[0]-1] * f[1]
        #print(res)
        if res >= 0:
            return 1
        else:
            return 0

class average_perception_train(Predictor):
    def __init__(self):
        self._weights =[0]* maxFeature
        
    def train(self, instances):
        I = iterations 
        length = len(self._weights)
        add_weights = [0]* maxFeature
        for i in range(I):              
            for e in instances:
                if str(decision(self._weights, e._feature_vector)) != str(e._label):
                    self._weights = update(self._weights, atheta, str(e._label), e._feature_vector)
                for j in range(length):
                    add_weights[j] += self._weights[j] 
        self._weights = [ a/float(I*len(instances)) for a in add_weights ]
        
        
    def predict(self,instance):
        res=0
        for f in instance._feature_vector._data:
            if f[0] <= len(self._weights):
                res=res + self._weights[f[0]-1] *f[1]
        if res >=0:
            return 1
        else:
            return 0
 

def train(instances, algorithm):
    # TODO Train the model using "algorithm" on "data"
    # TODO This is where you will add new algorithms that will subclass Predictor
     
    if algorithm == "perceptron":
        sol = perception_train()
        sol.train(instances)
        return sol 
        
    if algorithm == "averaged_perceptron":
        sol = average_perception_train()
        sol.train(instances)
        return sol 


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
    global atheta
    global iterations
    atheta = args.online_learning_rate 
    iterations = args.online_training_iterations

    if args.mode.lower() == "train":
        # Load the training data.
        instances = load_data(args.data)
        maxFeature = ComputeMaxFeature(instances)
        #print maxFeature

        # Train the model.
        predictor = train(instances, args.algorithm)
        #print predictor
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            print(pickle.PickleError)
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


#python classify.py --mode train --algorithm perceptron --model-file finance.perceptron.model --data finance.train
#python classify.py --mode test --model-file easy.perceptron.model --data easy.dev --predictions-file easy.dev.predictions

#python classify.py --mode train --algorithm averaged_perceptron --model-file easy.averaged_perceptron.model --data easy.train
#python classify.py --mode test --model-file easy.averaged_perceptron.model --data easy.dev --predictions-file easy.dev.predictions
#python compute_accuracy.py easy.dev easy.dev.predictions

#python classify.py --mode train --algorithm averaged_perceptron --model-file bio.averaged_perceptron.model --data bio.train
#python classify.py --mode test --model-file bio.averaged_perceptron.model --data bio.dev --predictions-file bio.dev.predictions
#python compute_accuracy.py bio.dev bio.dev.predictions

