# pattern_labeler
An LSTM network used for detecting and labeling parts of projected pattern for use in active triangulation systems for depth image acquisition.

The code is a modified version of https://github.com/oxford-cs-ml-2015/practical6

The code is used for labeling projected patterns in active triangulation systems and thus solving the correspondence problem. The system it was tested on was composed of a camera and a projector of 11 parallel lines. The goal of the network is to detect and label (from 1-11 and additional label 12 for background/no pattern) parts of the image where the projected pattern is present.

The image below shows an example of the projected pattern on the scene.

![https://github.com/jkravanja/pattern_labeler/blob/master/img/00000.png]
