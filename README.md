# DeepArts
One of the projects of style transfer technology. And not the best.
A paper on this work will be published on March 13 (LaTeX file format)

### Method
[Main Paper about style transfer](https://arxiv.org/abs/1508.06576/ "Come on, Click on this button")

### Instruments
* VGG19, Keras Application. 
It represents deep learning model that are made available alongside pre-trained weights. Keras apps can be used for prediction, feature extraction, and fine-tuning.

VGG19 is a model for image classification with weights trained on [ImageNet](http://www.image-net.org/ "Come on, Click on this button")

### Arguments
* include_top: whether to include the 3 fully-connected layers at the top of the network.
* weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
* input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
* input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
* pooling: Optional pooling mode for feature extraction when include_top is False.
None means that the output of the model will be the 4D tensor output of the last convolutional block.
'avg' means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
'max' means that global max pooling will be applied.
* classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.

### Loss
We’ll pass the network both the desired content image and our base input image. This will return the intermediate layer outputs from model. Then we take the euclidean distance between the two intermediate representations of those images.  

