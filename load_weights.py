# -*- coding: utf-8 -*-
"""load_weights.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ljg3U5n191iBGnrgbv-1PgTdRouUzJqi
"""

import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from YOLOv3.models import YoloV3, YoloV3Tiny
from YOLOv3.utils import load_darknet_weights

flags.DEFINE_string('weights', 'weights/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', 'weights/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.summary()
    logging.info('Model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('Weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('Sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('Weights saved')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass