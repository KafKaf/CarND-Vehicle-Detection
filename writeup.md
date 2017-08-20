**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[video1]: .output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code to get hog features is contained in the first code cell of the notebook in method `get_hog_features()`.

I explored many combinations of parameters which I'll explain later on.

I read from disk all vehicle and non-vehicles images and extracted HOG features(and color features) before feeding it to the classifier for training(second cell@notebook).

For an example using `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, please check the visualize image pipeline code cell in the notebook.

####2. Explain how you settled on your final choice of HOG parameters.

My final choice:
    Color Space = LUV
    Orientiation Bins = 8
    Pixels Per Cell = 8 (8,8)
    Cells Per Block = 2 (2,2)

I tried to maximize the trade off between:
1) Test set accuracy
2) Classifier generalizes well - correct predictions on new images and not many false positives
3) Computation time(high correlation with training time)

Computation time was less of a factor for me(I have a slow laptop with no gpu), as I know it can be trained in real time with strong gpus.

I tested a lot of combinations of different color spaces, orientations, pixels per cell and cells per block, and it gave me a good intuition on the best hog parameters and the tradeoffs between them.

orientation = orientation histogram bins:
    higher - more accurate, slower
    lower - less accurate, faster

pixel per cell = pixels per cell, on each cell we compute the gradient histogram.
    higher - less accurate, faster
    lower - more accurate, slower
    
cells per block = how many nearby cells(block) are normalized together
    optional - for hog descripor to be lightening invarient.
    
high dependencies - pixel per cell & orientation:
    higer pixel per cell value needs more accurate orientation binning and the other way around.
    
My goal was to get to the best results in terms of recall and precision on the test set and more importantly on the videos images, and then reduce the accuracy for faster computation until I get to a satisfactory accuracy and speed trade off.

It was all trail and error.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color and shape features: hog features, color histogram features and spatial color binning features.

You can see the code in the notebook under "Classifier":
1) Read all vehicle and non vehicle images path to 2 lists and shuffle them.
2) Configure all the parameters of feature extraction.
3) Extract all vehicle feature, non vehicle features, stack them together and scale them, stack all labels as well.
Scaling is important as hog features, color histogram features, spatial color binning features all have different scales.
4) Split training set and test set randomally.
5) Train a linear svm classifier(Linear Support Vector Classifcation).
6) Show mean prediction accuracy,precision,recall and f1 score on the test set.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding multi scale window search is implemented in `find_cars()` method in the notebook.

First, we need to convert to chosen color space and resize the image according to scale.

We run a sliding window horizontally and vertically for given space within an image.
The space is chosen for the purpose of minimizing the sliding window search where it's not needed because of scale and camera distortion.

For every step we extract a patch of 64,64 of the image, create a feature vector from all features, scale it and run a prediction on the classifier, if prediction is positive, we add the patch points to a list.

This runs with different scales and different x,y start and stop points with respect to region of interest.
What is our region of interest here? upper region of bottom half show cars as small objects around the middle of the image and lower region of bottom half show cars as big objects around full width.
So smallest scale will be upper y region and middle x region and it will grow down and to the sides with the scale.

Since this is computationally expensive process, I tried to optimize it as much as possible.
I tried many combinations on test images, I think this is the best trade off between run time and accuracy.

Since we don't want to miss a vehicle, we have overlapping windows/patches, for me it's 75% overlap. Setting the value lower than that caused false negatives and setting it higher was too slow, didn't add new information and even hurt the anomaly detection that happens later.

You can check the notebook for a view of the search window and overlap.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
You can check the notebook for images examples with all the scales and the thresholding in the end.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](.output_videos/project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I joined all the positive detections of the last 10 frames(where there is not a lot of relative movement between cars) in a fifo queue.
From these detections I created a heatmap to identify the hot spots.
After it, I thresholded the heatmap(detection should be "hot" in multiple scales/frames) in order to find vehicle positions and remove outliers(false positives).
I used `scipy.ndimage.measurements.label()` to identify cars in the heatmap, it ignores zero values and finds indivudal objects from heatmap.
I added bounding boxes around each label detected.

This aggeregation technique of frames serves 2 purposes:
1) remove outliers
2) find vehicles where the classifier predicted false result(false negative)

Please refer to the notebook for a visualization of the process.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most acute problem is prediction accuracy and speed with the sliding window search technique.
Since we want it to run the pipeline in real time, we trade off accuracy(multi scale and overlap)for processing time which in turn causes many false positives/negatives.
The classifier itself can be improved as well, with more augmeneted data or a different classifier, the current one is overfitting the dataset, and therefore not generalizing well.
The frame positive predictions aggregation handles the false positives outliers nicely but it introduces other problems like small lags and bouncing windows.

My pipeline will fail in different lightening conditions(shade, etc) or where the data is much different than the training set.
Low scale patches are misclassified many times as well.

I can improve the classifier(or replace it with convlutional one), that would help a lot.
With enough time, I think I can filter out most of the noise from image with image processing techniques and then only search small nearyby areas.
After that, I can improve the sliding window search to take into consideration the last position of a vehicle, and start with the larger scales(dependce on location) so I won't have to search with the lower scales if I have a "match".
In real world scenario, we can add into our calculations information from other sensors, so we can minimize the search area and have better tracking after a detection.
