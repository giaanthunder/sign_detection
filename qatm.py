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
from tensorflow.keras.applications import resnet


class qatm():
    def __init__(self):
        vgg19 = tf.keras.applications.vgg19.VGG19( include_top = False, weights = 'imagenet' )

        # src = '/home/anhuynh/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5_tf2'
        # dst = '/home/anhuynh/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        # shutil.copy(src,dst)
        # extractor = resnet.ResNet50()
        # extractor = tf.keras.Model(inputs=extractor.input, outputs=extractor.get_layer('avg_pool').output)
        input_ = vgg19.input
        conv1_2 = vgg19.get_layer('block1_conv2').output
        conv3_4 = vgg19.get_layer('block3_conv4').output
        conv3_4 = Lambda( lambda x: tf.image.resize( x[0], size=(tf.shape(x[1])[1], tf.shape(x[1])[2]),method='bilinear'), name='resized_image' )( [conv3_4, conv1_2] )
        concat = Concatenate()( [conv1_2, conv3_4] )
        featex = Model( [input_], [concat], name='featex' )
        input_ = vgg19.input
        conv1_2 = vgg19.get_layer('block1_conv2').output
        conv3_4 = vgg19.get_layer('block3_conv4').output
        conv1_2 = Lambda( lambda x: tf.image.resize( x[1], size=(tf.shape(x[0])[1], tf.shape(x[0])[2]), method='bilinear'), name='resized_image' )( [conv3_4, conv1_2] )
        concat = Concatenate()( [conv1_2, conv3_4] )
        featex2 = Model( [input_], [concat], name='featex2' )

        self.model = create_model( featex , alpha=25)
        self.model_bkup = create_model( featex2 , alpha=25)

    def get_qatm_boxes(self, image_T, image_S, ws_list=[64,128]):
        print('QATM')
        template = image_T
        image = image_S
        # ws_list = [64,128]
        boxes = []
        scores = []
        for ws in ws_list:
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
            val = np.log(val)
            gray = val[0, :, :, 0]
            gray = cv2.resize(gray, (image.shape[1], image.shape[0]))
            score = compute_score(gray, w, h)
            score[score > -1e-7] = score.min()
            score = np.exp(score / (h * w))  # reverse number range back after computing geometry average
            x, y, w, h = locate_bbox(score, w, h)
            x = int(x)
            y = int(y)
            boxes.append([x,y,w,h])
            scores.append(score)
        return boxes


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



