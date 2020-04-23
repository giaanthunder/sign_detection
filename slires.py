import os, sys, time, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tensorflow as tf
import cv2

from tensorflow.keras.applications import vgg16, resnet, inception_v3, xception, densenet, nasnet
from tensorflow.keras.preprocessing import image
from tqdm import  tqdm
import sklearn
import sliwin


class slires():
    def __init__(self):
        model = resnet.ResNet50()
        self.model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
        

    def get_slires_boxes(self, img_t, img_s, ws_lst=[128, 256], thresh_lst=[0.73,0.73]):
        # thresh_lst=[0.73,0.73]

        rects = []
        for i in range(len(ws_lst)):
            ws = ws_lst[i]
            # thresh = thresh_lst[i]
            thresh = thresh_lst[i]

            # sliding window
            stride = ws // 8
            batch_size = 64

            img_1 = Image.fromarray(img_t)
            img_1 = img_1.resize((ws, ws), Image.ANTIALIAS)
            img_1 = img_1.save("temp.jpg")
            img_1 = image.load_img("temp.jpg",target_size=(224, 224))
            img_1 = image.img_to_array(img_1)
            
            ds = sliwin.sliding_generator(img_1, img_s, stride, ws, resize=224, batch_size=batch_size)
            print('win size:',ws,', num sample:', ds.patch_cnt)

            

            # batches = []
            # for next_batch in ds.gen:
            #     batches.append(next_batch)

            # # feed forward
            # for next_batch in tqdm(batches):
            for i in tqdm(range(ds.batch_cnt)):
                img1, img2, start_point, end_point = next(ds.gen)

                start_point = tf.convert_to_tensor(start_point)
                end_point = tf.convert_to_tensor(end_point)

                img1 = tf.convert_to_tensor(img1)
                img1 = tf.cast(img1, dtype=tf.float32)

                img2 = tf.convert_to_tensor(img2)
                img2 = tf.cast(img2, dtype=tf.float32)

                chosen_img, chosen_start, chosen_end, chosen_dist, chosen_sim = self.speed(img1, img2, start_point,end_point, thresh)
                chosen_img    = chosen_img.numpy()
                chosen_start = chosen_start.numpy()
                chosen_end    = chosen_end.numpy()
                chosen_dist  = chosen_dist.numpy()
                chosen_sim    = chosen_sim.numpy()

                for j in range(chosen_img.shape[0]):
                    # collect rectangle position
                    x, y = chosen_start[j].astype(np.int32).tolist()
                    rects.append([x,y,ws,ws])

        if len(rects) > 0:
            grp_rects, weights = cv2.groupRectangles(rects,1)
            if len(grp_rects) > 0:
                return grp_rects
            else:
                return rects
        else:
            return []

    @tf.function
    def speed(self, img1, img2, start_point, end_point, thresh):
        # compute target feature map
        f1 = resnet.preprocess_input(img1)
        f1 = self.model(f1)
        f1 = tf.math.l2_normalize(f1, axis=1)
        f1 = tf.keras.layers.Flatten()(f1)

        # compute patch feature map
        f2 = resnet.preprocess_input(img2)
        f2 = self.model(f2)
        f2 = tf.math.l2_normalize(f2, axis=1)
        f2 = tf.keras.layers.Flatten()(f2)

        # compute similarity
        dist = tf.reduce_sum(tf.square(f1-f2), axis=1)
        sim  = tf.reduce_sum(f1*f2, axis=1)

        # extract pass samples
        # chosen_idx = tf.where(sim > 0.73)
        chosen_idx = tf.math.logical_and(sim > thresh, dist < 0.5)
        # chosen_idx = tf.math.logical_or(sim > 0.7, dist < 0.5)
        chosen_idx = tf.where(chosen_idx)
        chosen_idx = tf.keras.backend.flatten(chosen_idx)
        chosen_img = tf.gather(img2, chosen_idx)
        chosen_start = tf.gather(start_point, chosen_idx)
        chosen_end = tf.gather(end_point, chosen_idx)

        chosen_dist = tf.gather(dist, chosen_idx)
        chosen_sim  = tf.gather(sim, chosen_idx)

        return (chosen_img, chosen_start, chosen_end, chosen_dist, chosen_sim)


