from abc import ABC, abstractmethod
from torch import nn, Tensor
from typing import List, Dict

class Subject(ABC):
    '''
        Abstract class representing a subject (animal or network)
        tasked with a visual stimuli and generating a set of
        activations for multiple layers.
        
    '''
    
    @abstractmethod
    def foo(self):
        # TODO discuss what is a "Subject"
        pass
    
    
    
class NetworkSubject(Subject, nn.Module):
    '''
        Abstract class representing a network involved in
        a visual task experiment as an artificial counterpart 
        of an animal. A network subject has a layer indexing that
        to access each architecture component with a unique mapping. 
        
        NOTE The class also has the abstractmethod 'forward' from the nn.Module
    '''
    
    @property
    @abstractmethod
    def layer_names(self) -> List[str]:
        '''
        Return layers names in the network architecture.
        
        :return: List of layers names.
        :rtype: List[str]
        '''
        pass
    
    @abstractmethod
    def get_layer(self, layer_name: str) -> nn.Module:
        '''
        Return the network layer matching the name in input.
        NOTE The layer is expected to have attribute "name" which
            is its identifier in layer indexing
        
        :param layer_name: Layer name in the architecture.
        :type layer_name: str
        :return: Network layer.
        :rtype: nn.Module
        '''
        pass
    


class AlexNet(NetworkSubject): #Copied from 
    '''
    AlexNet CNN model architecture
    NOTE Same as it was in torchvision/models/alexnet.py
    NOTE works with input batches of size (batch_size, 224, 224, 3)
    '''
    
    def __init__(self, num_classes: int = 1000):
        
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def foo(self):
        # TODO 
        return None
    
    def get_layer(self, layer_name: str) -> nn.Module:
        '''
        Return the network layer matching the name in input.
        
        :param layer_name: Layer name in the architecture.
        :type layer_name: str
        :return: Network layer.
        :rtype: nn.Module
        '''
        
        layer_idx = self._names_layer_mapping[layer_name]
        layer = self.features[layer_idx]
        
        setattr(layer, "name", layer_name)
        
        return layer
    
    @property
    def layer_names(self) -> List[str]:
        '''
        Return layers names in the network architecture.
        
        :return: List of layers names.
        :rtype: List[str]
        '''
        
        return [v for v in self._names_layer_mapping]
        
    @property
    def _names_layer_mapping(self) -> Dict[str, int]:
        
        # TODO - temporary workaround before indexing layers by name
        #        we simply explicitly define the index of convolutional layers in
        #        the sequence
        return {
            "conv1": 0,
            "conv2": 3,
            "conv3": 6,
            "conv4": 8,
            "conv5": 10,
        }
        
    def forward(self, x: Tensor) -> Tensor:
                
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x