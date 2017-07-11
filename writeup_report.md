# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualize_nn.png "Model Visualization"
[image2]: ./examples/center.jpg "Center lane driving"
[image3]: ./examples/center_acw.jpg "Center lane driving in the opposite direction"
[image4]: ./examples/from_left_to_right_1.jpg "Recovery Image"
[image5]: ./examples/from_left_to_right_2.jpg "Recovery Image"
[image6]: ./examples/from_left_to_right_3.jpg "Recovery Image"
[image7]: ./examples/center_image.jpg "Normal Image"
[image8]: ./examples/center_image_flipped.jpg "Flipped Image"
[image9]: ./examples/left.jpg "Left Image"
[image10]: ./examples/center.jpg "Center Image"
[image11]: ./examples/right.jpg "Right Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64 (model.py lines 93-105) 

The model includes RELU layers to introduce nonlinearity (code line 93-94, 98-99, 103-104), and the data is normalized in the model using a Keras lambda layer (code line 92). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 96, 101, 111). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 123).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in the opposite direction. Also to get higher accuracy in left and right turns I have added snippets of dat recored for multiple left and right turns into a training set.a 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the VGG and NVIDIA architectures. I thought this model might be appropriate because VGG is an image classfier and NVIDIA was scpecifically designed to address behavioral driving problem. I ended up having a bit of both as a result of trial and error.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it had multiple Dropout layers with 0.5 dropout rate, that helped to improve the model a lot. Another techqnique to improve overfitting was using max pool layers between hierarchies of 16, 32 and 64 deps layers.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like one spot right after the bridge where some parts of a dividing lane is missing. To improve the driving behavior in these cases, I had to exclude certain portion of data from training set,  more specifically I discovered that my model was severly biased toward driving straight that eventually would get a car off the track  so I had to exclude all images with steering angle ~0.0. I also added a couple of driving snippets to a training data set specifically for driving around these areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 89-113) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also recored one lap of a driving in opposite direction:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover car's position after turn back to a center line driving mode. These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would overcome a left turn heavy bias in track1. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

I also added left and right camera images to the training set with corrected stearing angle +0.3 for the left camera image and -0.3 for the right one. That provided more data points for NN. 0.3 correction angle was picked experimentally and helped a lot during solving a  bias towards center line driving of neural network.

![alt text][image9]{:width="80%"}
![alt text][image10]{:width="80%"}
![alt text][image11]{:width="80%"}


After the collection process, I had about 7500 data points. I then preprocessed this data by converting the images into RGB color space from BGR, then cropping the image by removing top part with the sky and bottom part with mostly car's hood as insignificant for training and finally normalized pixels to [0, 1] interval. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by validation loss being increased after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
