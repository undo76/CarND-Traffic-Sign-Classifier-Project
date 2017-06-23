# Traffic Sign Recognition

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/fig-1.jpg "Visualization"
[image2]: ./imgs/fig-2.jpg "Histogram"
[image3]: ./imgs/fig-3.jpg "Preprocessing"
[image4]: ./imgs/fig-4.jpg "Learning curves"
[image5]: ./imgs/fig-5.jpg "Confusion matrix"
[image6]: ./imgs/fig-6.jpg "Real images"
[image7]: ./imgs/fig-7.jpg "Cropped images"
[image8]: ./imgs/fig-8.jpg "Prediction of real images"
[image9]: ./imgs/fig-9.jpg "Prediction of test images"
[image10]: ./imgs/fig-10.jpg "Visualization of layer activation"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/undo76/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here there are some labeled random samples from the training set. We can appreciate that some of them are challenging even for humans, as their illumination conditions are not optimal

![image1]

I plot the frequency of the traffic signs in the different data sets.

![image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to standarize every image to have mean = 0 and std = 1. This will center and scale the range of the features (pixels) to be uniform. This will help the training process in two ways: 1) It will equalized the inputs reducing the sensibility to different light conditions and contrast. 2) It will unskew the distribution of data, reducing the oscillations due to the different gradients magnitudes. (This is a bit less important as I am using the Adam optimizer that has individual learning rates for each parameter and can compensate automatically for disparity of feature scales). 

I also experimented with grayscale images, but the result didn't varied a lot.

<pre>
<code>
def preprocess_images(X, greyscale=False, standarize=True):
    if greyscale:
        X = np.dot(X, [0.299, 0.587, 0.114])[..., np.newaxis]
    if standarize: 
        X = (X - np.mean(X, axis=(1,2), keepdims=True)) / np.std(X, axis=(1,2), keepdims=True)    
    return X
</code>
</pre>

Here is an example of a traffic sign image after standarizing.

![image3]

I also experimented with augmentation of the data set adding noise, but the accuracy didn't improve a lot, while making the training slower. In addition, through simple inspection, it seems that the training dataset has already been augmented with noise and affine transformations, so it doesn't make too much sense to add similar transformations again. Check the notebook for additional details.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have generated a convolutional neural network combining convolutions, max-pooling, ReLUs and dropout layers. The final layers are comprised of two full connected units.

<pre>
<code>
def net(x, keep_prob, n_classes):   
    x = tf.layers.conv2d(x, 128, (5, 5), padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
    
    x = tf.layers.conv2d(x, 256, (5, 5), padding='same', activation=tf.nn.relu)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
    
    x = tf.reshape(x, shape=(-1, 8*8*256))
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    
    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    x = tf.nn.dropout(x, keep_prob=keep_prob)

    logits = tf.layers.dense(x, n_classes)
    tf.identity(logits, 'logits')
    return logits
</code>
</pre>

My final model consisted of the following layers:

| Layer              		|     Description	                           					| 
|:---------------------:|:---------------------------------------------------:| 
| Input         		    | 32x32x3 RGB image         			            				| 
| Convolution 5x5     	| 5x5 stride, same padding, outputs 32x32x128, ReLU 	|
| Max pooling	         	| 2x2 stride, outputs 16x16x128         		      		|
| Convolution 5x5     	| 5x5 stride, same padding, outputs 16x16x256, ReLU 	|
| Dropout               |                                                     |
| Max pooling	         	| 2x2 stride, outputs 8x8x256                         |
| Dropout               |                                                     |
| Dense                	| outputs 256, ReLU                                  	|
| Dropout               |                                                     |
| Dense                	| outputs #classes (43), Linear                       |
| Softmax				        | (Calculated with the loss)                          |
 

#### 3. Describe how you trained your model.

In order to train the model I used an Adam optimizer to minimize the multiclass cross-entropy of the Softmax of the logits.

<pre><code>def losses(logits, y):
    batch_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    losses = tf.reduce_mean(batch_losses)
    tf.summary.scalar('loss', losses)
    return losses

def optimize(loss, learning_rate, beta1=0.9, beta2=0.95):
    return tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss)</code></pre>

I used the following hyperparameters.

<pre><code>progress = fit(model, X_train_pp, y_train, X_valid_pp, y_valid,
    n_epochs=25, 
    learning_rate=0.0001, 
    keep_prob=0.45,
    batch_size=64,
   )</code></pre>

These are the learning curves during training.

![image4]


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:

* training set accuracy of 100%
* validation set accuracy of 97%
* test set accuracy of 97%

In order to analyse the performance of the model I calculated the confusion matrix and the precision, recall and F1-score over the test dataset.

![image5]

<pre><code>
                                                    precision    recall  f1-score   support

                              Speed limit (20km/h)       1.00      0.76      0.87        59
                              Speed limit (30km/h)       0.93      0.99      0.96       720
                              Speed limit (50km/h)       0.96      1.00      0.98       750
                              Speed limit (60km/h)       0.94      0.95      0.94       450
                              Speed limit (70km/h)       0.99      0.98      0.98       660
                              Speed limit (80km/h)       0.94      0.98      0.96       629
                       End of speed limit (80km/h)       1.00      0.88      0.94       149
                             Speed limit (100km/h)       0.98      0.86      0.92       448
                             Speed limit (120km/h)       0.91      0.94      0.92       448
                                        No passing       0.96      0.98      0.97       480
      No passing for vehicles over 3.5 metric tons       0.99      0.99      0.99       658
             Right-of-way at the next intersection       0.90      0.96      0.93       419
                                     Priority road       0.99      0.99      0.99       688
                                             Yield       0.99      1.00      0.99       719
                                              Stop       1.00      1.00      1.00       270
                                       No vehicles       0.97      1.00      0.98       210
          Vehicles over 3.5 metric tons prohibited       1.00      0.99      1.00       150
                                          No entry       1.00      0.98      0.99       360
                                   General caution       0.98      0.94      0.96       388
                       Dangerous curve to the left       0.95      1.00      0.98        60
                      Dangerous curve to the right       0.97      0.98      0.97        90
                                      Double curve       0.97      0.82      0.89        90
                                        Bumpy road       0.99      0.99      0.99       120
                                     Slippery road       0.97      0.99      0.98       150
                         Road narrows on the right       1.00      0.82      0.90        90
                                         Road work       0.94      0.96      0.95       479
                                   Traffic signals       0.87      0.84      0.86       180
                                       Pedestrians       0.89      0.53      0.67        60
                                 Children crossing       0.97      0.97      0.97       150
                                 Bicycles crossing       0.97      1.00      0.98        90
                                Beware of ice/snow       0.86      0.82      0.84       150
                             Wild animals crossing       0.98      1.00      0.99       268
               End of all speed and passing limits       0.91      1.00      0.95        60
                                  Turn right ahead       1.00      1.00      1.00       209
                                   Turn left ahead       0.98      0.99      0.99       120
                                        Ahead only       0.99      0.96      0.98       388
                              Go straight or right       0.99      0.97      0.98       120
                               Go straight or left       0.97      0.97      0.97        60
                                        Keep right       0.99      0.99      0.99       690
                                         Keep left       1.00      0.94      0.97        89
                              Roundabout mandatory       0.98      0.91      0.94        90
                                 End of no passing       1.00      0.83      0.91        60
End of no passing by vehicles over 3.5 metric tons       0.98      1.00      0.99        90

                                       avg / total       0.97      0.96      0.96     12608
</code></pre>

In order to get this result I have tested different NN architectures and hyperparameters. I tried deeper architectures, but they weren't substancially better. I also tried a hinge loss (as in a SVM) as objective function of the optimiser with promising results (around 98% accuracy), but I discarded it for not being a heterodox aproach. At the beginning, the model was overfitting, so I had to apply some agressive dropout. I also tried to compensate the unbalanced dataset adding weights to the losses proportional to the inverse of their frequency, but it didn't improve the final result.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image6] 


We see that we need to center and crop these images in order to be used by our NN. This step is done manually. These are the challenges that the classifier will have to face:

1. Low resolution of the sign. Dirt.
2. This one has almost perfect conditions. It shouldn't be problematic.
3. Background elements. Bad lighting conditions. 
4. The sign is not completely vertical. There is also a sticker on it.
5. Not completely vertical. There is a shadow that could be problematic.

![image7]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

All the downloaded images are predicted perfectly giving an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Here I display the top-5 results for each image. We can see that the model is very certain for these images.

![image8]

For comparison, here I display some cases in the test dataset.

![image9]

###  Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here we display the feature maps activation of the first convolutional layer. We can observe that most of the feature maps learn to detect borders at different angles.

![image10]


