# Object Detection using YOLOv3
Object detection is an important and challenging ﬁeld in computer vision, one which has been the subject of extensive research. It has been widely used in autonomous driving, pedestrian detection, medical imaging, robot vision, intelligent video surveillance, etc. Even though there exist many detection methods, the accuracy, rapidity, and efficiency of detection are not good enough. YOLOv3 is an algorithm that uses deep convolutional neural networks to perform object detection. YOLOv3  is an improvement over previous  detection networks, it features multi-scale detection, a stronger feature extractor network, and some changes in the loss function. In this project, we will build a real-time object detector based on the  model and then load the model to make predictions for both images and videos. 

This repository implements Yolov3 using TensorFlow 2.0

# Installation
To check the working of our model, you can follow the steps below;
1.Clone repository. <br/>
  git clone https://github.com/shettibhagya/Deep-Learning.git
2.Intall the requirements (TensorFlow - CPU)
  pip install -r requirements.txt
3.Download the pre-trained weights and save it in weights folder.
  wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
4.To save YOLOv3 weights in TensorFlow format, run
  python load_weights.py
5.Running the model on images and videos.
  python detect.py 
  python detect_video.py
  
YOLOv3_on_colab.ipynb implements the above installation steps.
