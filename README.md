# carnd_traffic_sign_classifier
This is a German traffic signs classifier written in TensorFlow, which I created as part of Udacity's Self-Driving Car Engineer Nanodegree (carnd).

##The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
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

