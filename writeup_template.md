# **Behavioral Cloning** 

[//]: # (Image References)

[image2]: ./examples/center_2017_07_13_23_45_58_498.png "new track"
[image3]: ./examples/center_2017_07_13_19_20_34_746.png "Recovery Image"
[image4]: ./examples/center_2017_07_13_19_20_35_448.png "Recovery Image"
[image5]: ./examples/center_2017_07_13_19_20_35_826.png "Recovery Image"
[image6]: ./examples/center_2017_07_13_18_54_58_883.png "Normal Image"
[image7]: ./examples/center_2017_07_13_18_54_58_883_flip.png "Flipped Image"
 
---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural network with 7x7, 5x5 and 3x3 filter sizes and depths between 24 and 96 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

The model is similiar to the NVDIA neural network.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers and l2 regularization in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an rmsprop optimizer, the default learning rate is 0.001 and I change that to 0.0001 (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road; and some additional data to reinforce the model to learn how to drive in special road texture. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVDIA model I thought this model might be appropriate because in NVDIA paper it has demonstrated its capabilities.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting and the model does not generalize well.

To combat the overfitting, first I record the data from the second track. By introducing new data, the model can be forced to learn a more generalized pattern. I modified the model by introducing dropout and regularization. By doing those steps, the model performs better than before. 

The final step was to run the simulator to see how well the car was driving around track one. However, at some sharp corners, the vehicle does not steer enough and fell out of the track. In order to improve the driving behavior in these cases, This implies the model do not have large variance. Therefore, I increase the nodes number to make the model more complex. This turns out to be very effective.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 53-76) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   				    	|
| crop layer         	| (50, 10)                                      | 
| Convolution 7x7     	| 7x7x24 size, 2x2 stride, valid padding        |
| RELU					|												|
| Convolution 5x5     	| 5x5x36 size, 2x2 stride, valid padding        |
| RELU					|												|
| Convolution 3x3     	| 3x3x48 size, 2x2 stride, valid padding        |
| RELU					|												|
| Convolution 3x3     	| 3x3x64 size, 1x1 stride, valid padding        |
| RELU					|												|
| Convolution 3x3     	| 3x3x96 size, 1x1 stride, valid padding        |
| RELU					|												|
| Fully connected		| 19584 input, 1743 output       				|
| Fully connected		| 1743 input, 200 output 						|
| Fully connected		| 100 input, 20 output     				    	|
| Fully connected		| 20 input, 1 output     				    	|
|						|												|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and two laps in the reverse direction. Here is an example image of center lane driving:

![alt text][image6]

In order to make the model generalize, I recorded one lap on the new track. Here is an example image of center lane driving in new track:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the model not bias on either direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I preprocessed this data by randomly remove 50% image with steering angle of 0.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the validation loss. I used an rmsprop optimizer and reduce learning rate on plateau so that manually training the learning rate wasn't necessary.
