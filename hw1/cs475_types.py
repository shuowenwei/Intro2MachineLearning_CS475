from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # TODO 
        self._label = label
        
    def __str__(self):
        # TODO
        return(str(self._label))
 
class FeatureVector:
    def __init__(self):
        # TODO
        self._data = []   
        
    def add(self, index, value):
        # TODO
        self._data.append( [index,value] )
        
    def get(self, index):
        # TODO
        for i in self._data:
            if index == self._data[i][0]: 
                return self._data[i][1]
                    
class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
