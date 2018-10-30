# Advanced Computer Vision and Deep Learning

### Advanced CNN Architectures

#### Major Computer Vision Tasks:

* Object Classification, 
* Object Detection, 
* Image Segmentation.

#### Introduction

Object detection can be divided into two phases 1. Region Proposal. 2. RoI classification with Bounding Box prediction.

Region Proposal is important phase because whatever RoI we are going to generate in this phase they all should incorporate all the objects. Otherwise there is no way that further phases can detect objects properly. That’s why it’s extremely important for the region proposals to have a high recall. And that’s achieved by generating very large numbers of proposals \(e.g., a few thousands per frame\). Most of them will be classified as background in the second stage of the detection algorithm.

**Can be considered:** We also need to balance the validity of proposals with the number of RoIs. More RoIs means lots of processing. This makes real time object detection difficult to implement.

* Often the objects in an image are dispersed all over the place or partially hidden. We need more advanced technques to localize the objects in such images.
* We many also need to undertsand the relationship between objects in the image which helps in scene understanding. 
* There might be multiple objects, some of them overlapping, some poorly visible or occluded. Moreover, for such an algorithm, performance can be a key issue. In particular for autonomous driving we have to process tens of frames per second.
* RCNN, F-RCNN, YOLO are the CNN architectures which helps in fast object detection.
* **Task:** Classification of objects and localizing their location by putting a bounding box.
* Object detection with localization is a task of classification and regression for predicting the bounding box measure which includes the center \($x, y$\) coordinates along with height and width of the bounding box.

![Screenshot from 2018-08-11 13-19-39.png](:storage/b613c45d-360d-41e9-9a67-720cf038bd3e/2a0b0c05.png%20=800x400)

* Regression Loss functions: L1 Loss, L2 Loss, Smooth L1 Loss\(Best of L1 and L2\)
* To **predict bounding boxes**, we train a model to take an image as input and output coordinate values: $\(x, y, w, h\)$. This kind of model can be extended to any problem that has coordinate values as outputs! One such example is human pose estimation.

**Weighted Loss Functions**

You may be wondering: _how can we train a network with two different outputs \(a class and a bounding box\) and different losses for those outputs?_

We know that, in this case, we use categorical cross entropy to calculate the loss for our predicted and true classes, and we use a regression loss \(something like smooth L1 loss\) to compare predicted and true bounding boxes. But, we have to train our whole network using one loss, so how can we combine these?

**There are a couple of ways to train on multiple loss functions**, and in practice, we often use a **weighted sum of classification and regression losses** $\(weight\_1 _cross\_entropy\_loss + weight\_2_ L\_1loss\)$ the result is a single error value with which we can do backpropagation. **This does introduce a hyperparameter: the loss weights**. We want to weight each loss so that these losses are balanced and combined effectively, and in research we see that another regularization term is often introduced to help decide on the weight values that best combine these losses.

#### Region Proposals

When there are many objects in an image the main challange is to localize all the objects in an image and classify them. Image is disintegrated into many regions and each region is then passed through the CNN for classification.

Disadvantage: So many cropped images is CPU consuming. All cropped images may not have objects or partial objects.

![screen-shot-2018-05-03-at-1.04.27-pm.png](:storage/b613c45d-360d-41e9-9a67-720cf038bd3e/c2f64c58.png%20=300x400)

**Strategy to segregate multiple objects in a single image -**

The regions we want to analyze are those with complete objects in them. We want to get rid of regions that contain image background or only a portion of an object. So, two common approaches are suggested: 1. Identify similar regions using feature extraction or a clustering algorithm like k-means, as you've already seen; these methods should identify any areas of interest. 2. Add another layer to our model that performs a binary classification on these regions and labels them: object or not-object; this gives us the ability to discard any non-object regions!

**Region Proposal Algorithm**: Produces limited set of cropped images which have high chances of object in them. These are called Region of Interest \(RoI\).

Segmentation algorithms which clalssify each region produced by either sliding window or some other algorithm. Then based on class scores selects the RoI which have high chances of objects in them. These are then passed on to

**R-CNN**: Propose regions. Classify proposed regions on at a time. Produce Output label and Bounding box coordinates. **Fast R-CNN**: Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions. **Faster R-CNN**: Use convolutional network to propose regions.

#### R-CNN \(Region Convolutional Neural Network\)

* Least sophisticated region based architecture
* Its the basis of understanding how multiple object recognition algorithms work.
* It outputs a class score and a bounding box coordinated for every input RoI.
* RoI can be of different sizes so every RoI is warped to be a standard size. 
* R-CNN produces bounding box coordinates to reduce localization errors; so a region comes in, but it may not perfectly surround a given object and the output coordinates \(x,y,w,h\) aim to perfectly localize an object in a given region.
* R-CNN, unlike other models, does not explicitly produce a confidence score that indicates whether an object is in a region, instead it cleverly produces a set of class scores for which one class is "background". This ends up serving a similar purpose, for example, if the class score for a region is Pbackground = 0.10, it likely contains an object, but if it's Pbackground = 0.90, then the region probably doesn't contain an object.

  [\[paper\] Rich feature hierarchies for accurate object detection and semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) 

#### Fast R-CNN

* Instead of proposing regions from the image, proposals comes from the feature maps from certain Convolutional layer and then fed further for object classification and bounding box prediction.
* Feature maps are comparitively smller than the original image and contains sufficient information to classify the objects. Now since proposals can be of different sizes and to make them all of same size we perform **RoI Pooling** [Link: Region of interest pooling explained](https://deepsense.ai/region-of-interest-pooling-explained/).

![Screenshot from 2018-08-11 15-48-25.png](:storage/b613c45d-360d-41e9-9a67-720cf038bd3e/4881c065.png%20=800x400) ![Screenshot from 2018-08-11 15-49-27.png](:storage/b613c45d-360d-41e9-9a67-720cf038bd3e/983741ae.png%20=800x400)

**Region Of Interest Pooling**

**Why we need RoI Pooling**: There are certain problems with the conventional region proposal mechanism in R-CNN where many RoI are selected to make sure that each object present in the image is covered by the RoIs. So this is achieved by generating thousands of proposals from each image/frame\(in case of video\). Problem is that

* Large number of proposals leads to performance issue. No real time detection implementation possible.
* Suboptimal implementation in terms of processing speed.

These problems can be solved by Region of Interest Pooling.

**What is RoI Pooling**: Its an neural network layer used in object detection tasks. It was first proposed by Ross Grishick paper [Fast R-CNN](https://arxiv.org/abs/1504.08083). It speeds up the training and test time and also maintain high detection accuracy.

**To be continued, read Region of interest pooling explained link**

#### Faster R-CNN

The most time consuming part in the object detection is the generation of region proposals. To speedup faster r-cnn takes another approach and generates roi/proposals by itself.

**Steps in Faster R-CNN**

* Image passed to the CNN once.
* CNN generates the feature maps.
* Faster R-CNN has extension in the form of extra layers which form Region Proposal Network\(RPN\).
* RPN extracts the regions which are rich in basic features like edges, corners from feature maps of CNN.
* RPN then does a quick classification to check if an proposal has object in it or not. Rejected proposals are discarded.

![Screenshot from 2018-08-14 22-44-48.png](:storage/b613c45d-360d-41e9-9a67-720cf038bd3e/0e1f100a.png%20=600x300) ![Screenshot from 2018-08-14 23-01-41.png](:storage/b613c45d-360d-41e9-9a67-720cf038bd3e/48072f46.png%20=600x300)

The region proposal network \(RPN\) works in Faster R-CNN in a way that is similar to YOLO object detection, which you'll learn about in the next lesson. The RPN looks at the output of the last convolutional layer, a produced feature map, and takes a sliding window approach to possible-object detection. It slides a small \(typically 3x3\) window over the feature map, then for each window the RPN:

* Uses a set of defined anchor boxes, which are boxes of a defined aspect ratio \(wide and short or tall and thin, for example\) to generate multiple possible RoI's, each of these is considered a region proposal.
* For each proposal, this network produces a probability, Pc, that classifies the region as an object \(or not\) and a set of bounding box coordinates for that object.
* Regions with too low a probability of being an object, say Pc &lt; 0.5, are discarded.

**Training the Region Proposal Network**

**Since, in this case, there are no ground truth regions, how do you train the region proposal network?** The idea is, for any region, you can check to see if it overlaps with any of the ground truth objects. That is, for a region, if we classify that region as an object or not-object, which class will it fall into? For a region proposal that does cover some portion of an object, we should say that there is a high probability that this region has an object init and that region should be kept; if the likelihood of an object being in a region is too low, that region should be discarded.

I'd recommend this blog post [Deep Learning for Object Detection: A Comprehensive Review](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9) if you'd like to learn more about region selection.

**Speed Bottleneck**

Now, for all of these networks including Faster R-CNN, we've aimed to improve the speed of our object detection models by reducing the time it takes to generate and decide on region proposals. You might be wondering: is there a way to get rid of this proposal step entirely? And in the next section we'll see a method that does not rely on region proposals to work!

