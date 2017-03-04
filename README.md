# carnd_traffic_sign_classifier
This is a German traffic signs classifier written in TensorFlow, which I created as part of Udacity's Self-Driving Car Engineer Nanodegree (carnd).

##The phases of this project are the following:

* Loading the dataset
* Dataset summary, exploration and visualization
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

### Loading the dataset
First thing's first: download the [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
```
$ wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
```
The zip file contains 3 pickle files with training, validation and test images, all resized to 32x32 pixels.<br>
The zip file also contains a CSV file (signnames.csv) with the first column containing the class ID (an integer spanning 0-42), and the second column containing a descriptive name of the sign<br>
Here are the first 4 rows:

| ClassId| SignName    |
| :-----:|-------------|
| 0      | Speed limit (20km/h) |
| 1      | Speed limit (30km/h) |
| 2      | Speed limit (50km/h) |
| 3      | Speed limit (60km/h) |

A quick examination of the data sets yields these stats:
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```
Each class represents a differnet traffic sign.<br>
I think it is odd that the test set is larger than the validation set and that it is <i>so</i> large: it is 1/3 the size of the training set and 3 times the size of the validation set.  The validation set is in fact only about 12.7% of the training set, while 20% is recommended.  When I used part of the training set for validation (instead of using the supplied validation set), I received very good validation results.  However, I chose to use the suuplied validation set, as it seemed more appropriate.
```
X_train, X_valid, y_train, y_valid = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)
```
### Dataset summary, exploration and visualization
I plotted the distribution of the training samples between the classes
![](datasets_distribution.png)
It is evident that the classes do not have equal representation in the training dataset.  Because there are 43 classes, had the samples been distributed equally, we would have 2.33% (100%/43) of the samples in each class.  However, in the actual dataset distribution, most classes comprise less than 2%.  Class 2 has the largest number of samples (5.78%), and classes 0 and 19 have the lowest representation (0.52% each).  I also plotted a histogram of the percent-of-samples per class.  This graph provides another look at how the samples distribute between the classes and it also shows that most classes have a very low representation.<br>
The validation dataset also doesn't distribute the samples between the classes in an equal manner.  
It is interesting to look at how well the validation dataset represents the training set, and that's the reason I plotted the distributions of the training and validation set in the same plot.  Looking at the plot we see q large resemblence in the distributions of the two sets, but I wanted to make sure this is the case.  I divided the percent representation of each class in the training set, by its representation in the validation set and called this the "ratio"  A ratio close to 1 indicates that there about the same fraction of validation samples as training samples, in the specific class.  In the iPython notebook I print the "ratio" value for each class, but here it sufficient to note that the median=1.03  and mean=0.94.  Although the two datasets distribute similarly enough, the plot below shows that there are enough outliers to raise my curiousity if the classes that have low/high ratio will show poor validation accuracy.  We shall see.<br>
![](valid_train_per_class.png)
<br>
Next, let's have a quick look at an image from each class, just to start getting familiar with the images themselves.<br>
![](class_signs.png)
<br>
We see here several very dark images, some blurry images.  Images are mostly centered, and mostly occupy the same amount of area within each image.  Images are mostly upright, but there are a few with a bit of rotation.  Backgrounds vary, but are mostly yellowish-brownish.<br>
<br>
Now I want to dig deeper into one of the classes.  It is important to spend time reviewing the finer details of the images and watch for different qualities and characteristics they may have.  This will help in deciding what augmentations may help during the training of the model.<br>
Class 0 is small and might need augmenting, so let's look at it.
![](class_0_training.png)
<br>
This is a bit surprising: I expected more variance, but it looks like there's a small set of signs that appear multiple times with some differences.  Not exactly what I was expecting.
Next we look at class 0 images from the validation dataset.  
![](class_0_validation.png)
<br>
Another surprise!  These images look almost all the same - it's as if they were taken from the window of a slow moving car.<br>
I don't have much experience, but this kind of repetitivity seems problematic since its mostly testing for a small number of features, because these images are highly corrolated.  On the other hand, I suppose that in a real driving scenario, this is exaclty what we would expect to see: the same sign with some small variance due to the car movement.
<br>
I also examined class 19, since it is also small enough to display (180 training samples).  It exhibited similar chracteristics, but the training set has almost 30 very dark images (~15% of the images).  
![](class_19_training.png)
<br>
Even when I look at these closer (larger) they look extremely dark.
![](class_19_training_dark_closeup.png)
<br>
I converted them to grayscale, and look at the improvement!  Grayscale is just a dot product of [R,G,B] and [0.299, 0.587, 0.114], so I imagine this "feature" can be learned by the network.  I.e. the network can learn to exract the luminance feature if it determines it is important enough.  If in the pre-processing phase I will convert the dataset images to grayscale, I will bascially make the decision that the chroma channels are not important and that the luminance data is sufficient to represent the inputs.  These dark images give me confidence that there's a lot of information in the luminance data.
<br>
![](class_19_training_dark_grayscale.png)
<br>
###Design, train and test a model architecture
The code for preprocessing the image datasets is in the iPython cell which has "Preprocessing utilities" as the first remark.<br>
I first convert the RGB images to grayscale.  There's little color information in the images, and I think that the structure of the signs is enough. Yann LeCun reported in his paper that this helped increase the predcition accuracy by a small bit, while decreasing the complexity of the model (we only have to deal with one third of the input features).<br>
Next, I perform [min-max scaling](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling) (also known as scale normalization) to rescale the pixels to the 0-1 floating-point range.  It is common practice to do this in cases where the input features have different scales, but in 8-bit integer grayscale images the pixels are already within the same value range (0-255), so it would seem unnecessary to do this rescaling.  Finally, I standardized the data using [Z-score normalization](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html) which centers (mean=0) and normalizes (std=1) all of the images.<br>
As a sanity check, I choose a few random images from the pre-processed image set and display them:<br>
![](class_19_training_preprocessing_.png)
