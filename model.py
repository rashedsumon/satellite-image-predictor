import torch
import torch.nn as nn
import torchvision.models as models

def get_eurosat_model(num_classes=10):
    """
    Constructs a ResNet18 model configured for the 10 EuroSAT land cover classes.
    """
    # Load weights from a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify the final Fully Connected (FC) layer to output 10 target classes instead of 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Put model into evaluation mode for inference tasks
    model.eval()
    return model