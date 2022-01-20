# Coding ResNet-50 from scratch and training it on ImageNet

This is a purely training exercise. I wanted to build a relatively large CNN from scratch without making any use of anybody else's code. I also wanted to learn how to handle training on a relatively large dataset.

### Preprocessing data and training the model

ImageNet training set consists of close to 1.3 mln images of different sizes. The model accepts fixed size 224x224 RGB images as input. At very minimum, before an image can be fed to the model it needs to be cropped to 224x224 size if the shortest side is at least 224px, or it needs to be re-sized first and then cropped if it originally isn't. In theory one can preprocess all the images in the dataset first, save their 224x224 versions, and then load those preprocessed images during each training epoch when needed. Although this approach saves preprocessing  effort - for each image there is no need to do cropping as many times as there are epochs - it is deeply flawed. Benefits of data augmentation which comes with random cropping (different cropping for each epoch) are just way too significant to give up. The whole preprocessing procedure I'm using here is even a bit more elaborated. Here are the preprocessing steps I'm doing each epoch for each image before it is added to a batch:

1. pick a random number from this list \[256, 263, 270, 277, ... , 473, 480\] \(integers from 256 to 480 with step 7\)
2. re-size an image so the shortest side is equal to the integer chosen at step 1
3. randomly crop to 224x224 size
4. normalize resulting image (for each RGB channel apply channel specific mean and standard deviation calculated in advance for all images in the dataset)
5. with 50% probability flip the image horizontally, with 50% probability do nothing
6. apply slight random color augmentation

The preprocessing steps above have to be done for each image before the image can be included in a batch. Next epoch the same image will have to be preprocessed again. The preprocessing procedure described above takes more time than actual GPU training and would slow down training from one week (I was using one 8Gb RTX-2080 GPU) to a few weeks if GPU had to wait each time the next batch is formed.
