import torch

model_ids = ['mobilenet_v2', 'vgg11', 'vgg11_bn', 'alexnet', 'vgg16', 'vgg16_bn', 'resnet18',
             'densenet161', 'inception_v3',
             'googlenet', 'resnext50_32x4d', 'mnasnet1_0',
             'resnet50']


def load_model(model_identifier):
    return torch.hub.load('pytorch/vision:v0.6.0', model_identifier, pretrained=True, verbose=False).eval()
