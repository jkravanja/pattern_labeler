# pattern_labeler
An LSTM network used for detecting and labeling parts of projected pattern for use in active triangulation systems for depth image acquisition.

The code is a modified version of https://github.com/oxford-cs-ml-2015/practical6

train_set.t7 should contain fields:

data : FloatTensor - size: nr_images x nr_subimages x 400
labels : FloatTensor - size: nr_images x nr_subimages

The code is used for labeling projected patterns in active triangulation systems and thus solving the correspondence problem. The system it was tested on was composed of a camera and a projector of 11 parallel lines. The goal of the network is to detect and label (from 1-11 and additional label 12 for background/no pattern) parts of the image where the projected pattern is present.

The image below shows an example of the projected pattern on the scene.

![Projected pattern on the scene](https://github.com/jkravanja/pattern_labeler/blob/master/img/00000.png)

The input image is split into smaller (20x20 pixels) subimages and flattened to 400-dimensional vector.

![Subimages](https://github.com/jkravanja/pattern_labeler/blob/master/img/a.png)

This results in a sequence (subimages taken column-wise from the original input image) of 400-dimensional vectors which are fed to the input of LSTM. In each timestep, the LSTM produceces a probability distribution over labels at the output, one for each subimage at the input. The labels with max value are taken.

An example of labeled image is shown below.

![Labeled image](https://github.com/jkravanja/pattern_labeler/blob/master/img/1.jpg)

And the groundtruth

![Ground truth](https://github.com/jkravanja/pattern_labeler/blob/master/img/l_00001.jpg)

It is also interesting to observe, what kind of filters the network learned at the first (input) layer (see image below).

![filters](https://github.com/jkravanja/pattern_labeler/blob/master/img/filters_noise_0.png)
