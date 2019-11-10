# ConditionalLearnToPayAttention
A tensorflow version implementation of ConditionalLearnToPayAttention:designing a conditional attention mechanism to solve sequential visual task such as multiple objects recognition and image caption. 


## SVHN dataset
SVHN is obtained from house numbers in Google Street View images. The dataset can be download [here(format1)](http://ufldl.stanford.edu/housenumbers/).</br>

## Training data samples
![](https://github.com/caoquanjie/ConditionalLearnToPayAttention/raw/master/samples/sample.jpg)

## COCO dataset 
Mscoco is a dataset built by Microsoft, which includes detection, segmentation, keypoints and other tasks. The dataset can be download [here](http://cocodataset.org/#people)

## Requirements
python 3.6</br>
tensorflow 1.4.0</br>
numpy 1.15.0</br>
matplotlib 2.0.0</br>
skimage 0.15.0



## In MultipleObjectsRecognition</br>
### Training details
We generate images with bounding boxes, and resize the images to 64×64. 
We then use the similar data augmentation which crops a 54×54 pixel image from a random location within the 64×64 pixel image in [Goodfellow et al. (2013)](https://arxiv.org/pdf/1312.6082).</br>
In order to verify the universality of the model, we directly resize the orginal images in SVHN dataset without bounding boxes, and the results outperform than the method in [Goodfellow et al. (2013)](https://arxiv.org/pdf/1312.6082)</br>
Also we use multiple scale attention features to improve performance, and for different attention scales, the method of training model is the same.</br>
Run `python convert_to_tfrecords.py`, you can get three tfrecords files(train,val,test) with bounding box.</br>
Run `python main.py`


## In weaklySvhnRecognition</br>
We have only reprocessed the data, and the structure and training mode of the model have not changed, so we only need to run 'python convert_to_tfrecords.py' to generate new weakly labeled data.</br>
Run `python convert_to_tfrecords.py`, you can get three data tfrecords files(train,val,test) without bounding box.</br>


## Image Caption

## Result
The crop svhn recognition accuracy of this soft attention model is reached 97.15% than baseline CNN model 96.04% [here](https://github.com/caoquanjie/SVHN-multi-digits-recogniton).
The weakly svhn recognition accuracy of the soft attention model is reached 80.45% than baseline CNN model 70.58%
All qualitative and quantitative results are all exported to the svhn.log, you can print some other results to the logs if you are interested.
You also can view results in tensorboard.</br>

Run `tensorboard --logdir=logs`


## Visualization attention map 
Attention maps from conditional attention model trained on SVHN dataset with/without bounding box, or valisualization of image caption can be seen in our paper.
