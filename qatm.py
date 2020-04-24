import numpy as np
import cv2
import os, sys, shutil, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import progressbar
from models2 import QATM, MyNormLayer
from utils import compute_score, locate_bbox
from pathlib import Path
import tensorflow.keras.backend as K
# from tensorflow.keras.applications import resnet
import math


class qatm():
    def __init__(self):
        resnet = tf.keras.applications.resnet.ResNet50(include_top = False, weights = 'imagenet')
        input_ = resnet.input
        conv1_2 = resnet.get_layer('conv2_block2_out').output
        conv3_4  = resnet.get_layer('conv2_block3_out').output
        # conv1_2 = resnet.get_layer('conv4_block2_out').output
        # conv3_4  = resnet.get_layer('conv4_block4_out').output
        conv3_4 = Lambda( lambda x: tf.image.resize( x[0], size=(tf.shape(x[1])[1], tf.shape(x[1])[2]),method='bilinear'), name='resized_image' )( [conv3_4, conv1_2] )
        concat = Concatenate()([conv1_2, conv3_4])
        featex = Model([input_], [concat], name='featex')

        input_ = resnet.input
        conv1_2 = resnet.get_layer('conv2_block1_1_relu').output
        conv3_4 = resnet.get_layer('conv3_block3_out').output
        conv1_2 = Lambda(
            lambda x: tf.image.resize(x[1], size=(tf.shape(x[0])[1], tf.shape(x[0])[2]), method='bilinear'),
            name='resized_image')([conv3_4, conv1_2])
        concat = Concatenate()([conv1_2, conv3_4])
        featex2 = Model([input_], [concat], name='featex2')

        self.model = create_model(featex, alpha=25)
        self.model_bkup = create_model(featex2, alpha=25)

        # exit()
        # vgg19 = tf.keras.applications.vgg19.VGG19( include_top = False, weights = 'imagenet' )
        # input_ = vgg19.input
        # conv1_2 = vgg19.get_layer('block1_conv2').output
        # conv3_4 = vgg19.get_layer('block3_conv4').output
        # conv3_4 = Lambda( lambda x: tf.image.resize( x[0], size=(tf.shape(x[1])[1], tf.shape(x[1])[2]),method='bilinear'), name='resized_image' )( [conv3_4, conv1_2] )
        # concat = Concatenate()( [conv1_2, conv3_4] )
        # featex = Model( [input_], [concat], name='featex' )
        # input_ = vgg19.input
        # conv1_2 = vgg19.get_layer('block1_conv2').output
        # conv3_4 = vgg19.get_layer('block3_conv4').output
        # conv1_2 = Lambda( lambda x: tf.image.resize( x[1], size=(tf.shape(x[0])[1], tf.shape(x[0])[2]), method='bilinear'), name='resized_image' )( [conv3_4, conv1_2] )
        # concat = Concatenate()( [conv1_2, conv3_4] )
        # featex2 = Model( [input_], [concat], name='featex2' )
        #
        # self.model = create_model( featex , alpha=25)
        # self.model_bkup = create_model( featex2 , alpha=25)

    def get_qatm_boxes(self, image_T, image_S, ws_list=[64,128]):
        print('QATM')
        template = image_T
        image = image_S
        # ws_list = [64,128]
        boxes = []
        scores = []
        for ws in ws_list:
            print ("ws in qatm:", ws)
            # image_plot = image.copy()
            start = time.time()
            template = cv2.resize(template,(ws,ws))
            template_ = np.expand_dims(preprocess_input(template), axis=0)
            image_ = np.expand_dims(preprocess_input(image), axis=0)
            w,h,_ = template.shape
            if w * h <= 4000:
                val = self.model.predict([template_, image_])
                print (val.shape)
            else:
                # used when image is too big
                val = self.model_bkup.predict([template_, image_])
            # compute geometry average on score map
            val = tf.math.log(val)
            val = tf.image.resize(val, size=(image.shape[1], image.shape[0]))
            # val = np.log(val)
            # gray = val[0, :, :, 0]
            # gray = cv2.resize(gray, (image.shape[1], image.shape[0]))
            score = compute_score_tf(val, w, h)
            score[score > -1e-7] = score.min()
            score = np.exp(score / (h * w))  # reverse number range back after computing geometry average
            max_score = np.max(score)
            print(max_score)
            x, y, w, h = locate_bbox(score, w, h)
            x = int(x)
            y = int(y)
            boxes.append([x,y,w,h,max_score])
            scores.append(score)
        return boxes

def compute_score_tf( x, w, h ):
    # score of response strength
    # k = np.ones( (h, w) )
    k = tf.ones((h,w,1,1))
    score = tf.keras.backend.conv2d(x,k,padding='same')
    score = score[0, :, :, 0].numpy()
    # score = convolve( x, k, mode='wrap' )
    score[:, :w//2] = 0
    score[:, math.ceil(-w/2):] = 0
    score[:h//2, :] = 0
    score[math.ceil(-h/2):, :] = 0
    return score
def create_model(featex, alpha=1.):
    T = Input((None, None, 3), name='template_input')
    I = Input((None, None, 3), name='image_input')
    T_feat = featex(T)
    I_feat = featex(I)
    I_feat, T_feat = MyNormLayer(name='norm_layer')([I_feat, T_feat])
    dist = Lambda(lambda x: tf.einsum("xabc,xdec->xabde", K.l2_normalize(x[0], axis=-1), K.l2_normalize(x[1], axis=-1)),
                  name="cosine_dist")([I_feat, T_feat])
    conf_map = QATM(alpha, name='qatm')(dist)
    return Model([T, I], [conf_map], name='QATM_model')



