# Coding ResNet-50 from scratch and training it on ImageNet

This is a purely training exercise. I wanted to build a relatively large CNN from scratch without making any use of anybody else's code. I also wanted to learn how to handle training on a relatively large dataset.

### Preprocessing data and training the model

ImageNet training set consists of close to 1.3 mln images of different sizes. The model accepts fixed size 224x224 RGB images as input. At very minimum, before an image can be fed to the model it needs to be croped to 224x224 size if the shortest side is at least 224px, or it needs to be re-sized first and then cropped if it isn't.
