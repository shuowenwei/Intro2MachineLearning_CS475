from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # TODO
        pass
        
    def __str__(self):
        # TODO
        pass

class FeatureVector:
    def __init__(self):
        # TODO
        pass
        
    def add(self, index, value):
        # TODO
        pass
        
    def get(self, index):
        # TODO
        return 0.0
        

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

       
