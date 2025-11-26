from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """Base evaluator class for unified evaluation pipeline"""
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()
