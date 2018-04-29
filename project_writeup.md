# **Traffic Sign Recognition** 

## Project Writeup

### This project involved the implementation the LeNet CNN architecture to classify German traffic signs.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project included the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/training_images_distribution.png "Training Images Distribution"
[image2]: ./output_images/unprocessed_images.png "Unprocessed Images"
[image3]: ./output_images/preprocessed_images.png "Preprocessed Images"
[image4]: ./output_images/downloaded_traffic_signs.png "Downloaded Traffic Signs"
[image5]: ./output_images/softmax_plot.png "Probability Plot"

## Rubric Points
### Here the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually listed to describe how each point was implementated.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This document describes the solution approach, here is a link to the final [project code](https://github.com/DrBoltzmann/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

In thye data exploration and summary I used numpy, pandas, and matplotlib to build basic summary statistics of the traffic signs data set and visualized class distributions:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Basic distributions visualizations of the images and classes were created:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In the preprocessing development, various transformations were applied and their influence on training dataset loss was evaluated. Initally only grayscale conversion and normalization was used, and only about a 60% accuracy could be achieved. After evaluation of added preprocessing steps and rerunning the model, the following preprocessing pipeline was finalized:

* Convert to grayscale in order to reduce the image size and focus on building contrast between light and dark regions of the images.

* The image histogram was equalized (cv2.equalizeHist), which stretches the histogram to either ends of the light to dark range, which ideally improves contrast of the images.

* Gaussian blur was applied (cv2.GaussianBlur) to reduce image noise.

* Shapening was applied (cv2.addWeighted) to ideally improve segmentation in the images to differentate different features.

* Normalization was applied using cv2.normalize.

Here is an example the traffic signs before preprocessing:

![alt text][image2]

Here are the images after running the preprocessing. The features are easier to discern, and the test accuracy increased to ~90%:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the classic LeNet architecture,  the final output is equal to the number of sign classifications:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, 28x28x6 output, VALID padding		|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 14x14x6 output 					|
| Convolution 3x3		| 1x1 stride, 10x10x6 output, VALID padding		|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 5x5x16 output 					|
| Flatten				| 400 output									|
| Fully connected		| 120 output									|
| RELU					| 												|
| Add Dropout			|												|
| Fully connected		| 84 output										|
| RELU					|												|
| Fully connected		| 43 output										|
|						|												|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Model training included 20 epochs, a learning rate of 0.001, and a batch size of 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 95%
* test set accuracy of 91%

Solution approach:
* The LeNet architecture was implemented to provide a basic approach and see what accuracy it would provide (it was recommended to use in the problem description). The initial accuracy was ~60% with grayscale and normalization preprocessing. Modifing learning rate didn't have much influence. I then went back to the preprocessing function and included additonal functions including histogram normalization, Gaussian blur, sharpening, and normalization with OpenCV. This dramatically increased the train accuracy to ~90%. I applied dropout in the architecture, first with a keep_prob = 0.5, and then increased to 0.7, which lead to a an increase of a few % in accuracy. At this point I had been only using 5 epochs to evaluate the inital changes, after I saw the train accruacy was above 90% from the first few epochs, I increased the total epochs to 20, which provided enough iterations to achieve a 95% test accuracy. I had considered performing data augmentation to generate more training data, but since the test accuracy was above 93%, I decided to not augment the data set at this time.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found the following five German traffic signs via Google:

![alt text][image4]

Generally speaking, the test images all reflected features similar to the features distribution found in the test data set, therefore, I would not expect there to be any serious issues with classifing them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Prediction results and Softmax value:

| Image					| Prediction			| Softmax Output	| 
|:---------------------:|:---------------------:|:-----------------:| 
| Stop Sign				| Stop sign   			| 0.52				|
| 30 km/h				| 30 km/h 				| 0.46				|
| Road Work				| Road Work				| 0.44				|
| Right Turn Ahead 		| Right Turn Ahead		| 1.0				|
| 70 km/h				| 70 km/h      			| 0.99				|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top five soft max probabilities were:

| Image					| Softmax Output	| 
|:---------------------:|:-----------------:| 
| Stop Sign				| 0.52				|
| 30 km/h				| 0.46				|
| Road Work				| 0.44				|
| Right Turn Ahead 		| 1.0				|
| 70 km/h				| 0.99				|

When plotted, the Softmax probability distributions are the following:

![alt text][image5]

The model predicted the traffic signs well. The quality of the images was rather good, each appears to have been taken in good lighting, which provided a high quailty data set to evaluate. It would be possible that for other images of lower light quality, additional operations would be needed in order to provide better detail in the image before evaluation with the model.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


