from abc import ABCMeta, abstractmethod

class BaseModel(metaclass=ABCMeta):

    def __init__(self, feature_extractor):

        self.feature_extractor = feature_extractor


    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

