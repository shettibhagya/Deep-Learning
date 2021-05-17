# Object Detection using YOLOv3
Object detection is an important and challenging ﬁeld in computer vision, one which has been the subject of extensive research. It has been widely used in autonomous driving, pedestrian detection, medical imaging, robot vision, intelligent video surveillance, etc. Even though there exist many detection methods, the accuracy, rapidity, and efficiency of detection are not good enough. YOLOv3 is an algorithm that uses deep convolutional neural networks to perform object detection. YOLOv3  is an improvement over previous  detection networks, it features multi-scale detection, a stronger feature extractor network, and some changes in the loss function. In this project, we will build a real-time object detector based on the  model and then load the model to make predictions for both images and videos. 

This repository implements Yolov3 using TensorFlow 2.0

# Description of Dataset:
A labeled dataset is needed to train the YOLOv3 algorithm. We will use the COCO dataset which consists of 80 labels. The COCO dataset is a large scale object detection, segmentation, and captioning dataset that has 330k images ( >200k labeled), 1.5 million object instances,  80 object categories, 91 stuff categories, 5 captions per image and 250,000 people with key points. The pre-trained weights of  with the COCO dataset are uploaded as initial weights for the model.

#  Description of Model:
“You Only Look Once” is an algorithm that uses convolutional neural networks for object detection. In comparison to recognition algorithms, a detection algorithm does not only predict class labels but detects locations of objects as well. So, it not only classifies the image into a category, but it can also detect multiple objects within an Image. This algorithm applies a single neural network to the full image. It means that this network divides the image into regions/grids and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

# Installation
To check the working of our model, you can follow the steps below;<br/>
1.Clone repository. <br/>
  git clone https://github.com/shettibhagya/Deep-Learning.git<br/>
2.Intall the requirements (TensorFlow - CPU)<br/>
  pip install -r requirements.txt<br/>
3.Download the pre-trained weights and save it in weights folder.<br/>
  wget https://pjreddie.com/media/files/yolov3.weights <br/>
4.To save YOLOv3 weights in TensorFlow format, run<br/>
  python load_weights.py<br/>
5.Running the model on images and videos.<br/>
  python detect.py <br/>
  python detect_video.py<br/>
  
YOLOv3_on_colab.ipynb implements the above installation steps.

# Results
![image](https://user-images.githubusercontent.com/75746037/118541741-10957980-b720-11eb-9f5e-811de6d552dd.png)
![image](https://user-images.githubusercontent.com/75746037/118541777-1a1ee180-b720-11eb-89d8-31eb2dcf1541.png)

# Conclusion
We constructed and compiled  model in TensorFlow and Keras. We transferred weights from original Darknet weights to constructed model and then tested the model to make predictions on images and videos. All the image and video predictions are uploaded in detections folder repository. It takes approximately 70ms for detection using  on Windows. Predictions at different scales or aspect ratios for same objects improved because of the addition of feature pyramid like method. Our model is fast and accurate and this makes it the best model to choose in applications where speed is important either because the objects need to be real-time or the data is just too big.

