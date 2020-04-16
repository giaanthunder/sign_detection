import os, sys, time, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
plt.axis('off')
from PIL import Image, ImageOps
import tensorflow as tf
import cv2

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from tensorflow.keras.applications import resnet
from tqdm import  tqdm
import sklearn
import sliding_window as sliwin




@tf.function
def speed(img1, img2, start_point, end_point,thresh = 0.75):
   # compute target feature map
   f1 = resnet.preprocess_input(img1)
   f1 = model(f1)
   f1 = tf.math.l2_normalize(f1, axis=1)
   f1 = tf.keras.layers.Flatten()(f1)

   # compute patch feature map
   f2 = resnet.preprocess_input(img2)
   f2 = model(f2)
   f2 = tf.math.l2_normalize(f2, axis=1)
   f2 = tf.keras.layers.Flatten()(f2)

   # compute similarity
   dist = tf.reduce_sum(tf.square(f1-f2), axis=1)
   sim  = tf.reduce_sum(f1*f2, axis=1)

   # extract pass samples
   # chosen_idx = tf.math.logical_and(sim > 0.75, dist < 0.45)
   # chosen_idx = tf.where(chosen_idx)
   chosen_idx = tf.where(sim > thresh)
   chosen_idx = tf.keras.backend.flatten(chosen_idx)
   chosen_img = tf.gather(img2, chosen_idx)
   chosen_start = tf.gather(start_point, chosen_idx)
   chosen_end = tf.gather(end_point, chosen_idx)

   chosen_dist = tf.gather(dist, chosen_idx)
   chosen_sim  = tf.gather(sim, chosen_idx)

   return (chosen_img, chosen_start, chosen_end, chosen_dist, chosen_sim)





src = '/home/vy/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5_tf2'
dst = '/home/vy/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
shutil.copy(src,dst)

model = resnet.ResNet50()
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
# model.summary()

num_stop = 1000
stop_flg = False

# load image
img_1_path = '/media/vy/DATA/Liem/advertising/DeepPretrain/origin_image/hit2.jpg'
view = 5
gsv_dir = '/media/vy/DATA/Liem/advertising/DeepPretrain/sample/view{}/'.format(view)
gsv_names = os.listdir(gsv_dir)
gsv_names.sort()

out_dir = 'view{}/'.format(view)
if os.path.exists(out_dir):
   shutil.rmtree(out_dir)

result_dir = out_dir
if not os.path.exists(result_dir):
   os.makedirs(result_dir)

ws_lst  = [32, 64, 128, 256]

for name in gsv_names:
   img_2_path  = gsv_dir + name
   name = name.split('.')[0]
   rects = []

   dbg_file = open(out_dir+name+"_result.txt", "a")

   tik = time.time()
   for ws in ws_lst:
      if ws == 128 or ws == 256:
         thresh = 0.73
      else:
         thresh = 0.8
      # sliding window
      stride = ws // 8
      ds = sliwin.sliding_generator(img_1_path, img_2_path, stride, ws, resize=224, batch_size=128)
      print('name:',name,', win size:',ws,', num sample:', ds.patch_cnt)

      batches = []
      for next_batch in ds.gen:
         batches.append(next_batch)

      # feed forward
      for next_batch in tqdm(batches):
         img1, img2, start_point, end_point = next_batch

         start_point = tf.convert_to_tensor(start_point)
         end_point = tf.convert_to_tensor(end_point)

         img1 = tf.convert_to_tensor(img1)
         img1 = tf.cast(img1, dtype=tf.float32)

         img2 = tf.convert_to_tensor(img2)
         img2 = tf.cast(img2, dtype=tf.float32)

         chosen_img, chosen_start, chosen_end, chosen_dist, chosen_sim = speed(img1, img2, start_point,end_point, thresh)
         chosen_img   = chosen_img.numpy()
         chosen_start = chosen_start.numpy()
         chosen_end   = chosen_end.numpy()
         chosen_dist  = chosen_dist.numpy()
         chosen_sim   = chosen_sim.numpy()

         # save pass image
         SAVE_DIR = out_dir + '%s/ws%d/'%(name,ws)
         if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

         if len(chosen_img.shape) == 3:
            # collect rectangle position
            x, y = chosen_start.astype(np.int32).tolist()
            rects.append([x,y,ws,ws])

            # collect debug infomation
            dist = chosen_dist
            sim  = chosen_sim
            dbg_file.write('x:%d, y:%d, ws:%d, dist:%.3f, sim:%.3f\n'%(x,y,ws,dist,sim))

            # early stopping
            if len(rects) > num_stop:
               stop_flg = True
               break

            # save found image
            x, y = chosen_start
            im_2_path = SAVE_DIR+'ws%d_%d_%d.jpg'%(ws, x, y)
            print(im_2_path)
            im_2 = cv2.cvtColor(chosen_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(im_2_path, im_2)

         for j in range(chosen_img.shape[0]):
            # collect rectangle position
            x, y = chosen_start[j].astype(np.int32).tolist()
            rects.append([x,y,ws,ws])

            # collect debug infomation
            dist = chosen_dist[j]
            sim  = chosen_sim[j]
            dbg_file.write('x:%d, y:%d, ws:%d, dist:%.3f, sim:%.3f\n'%(x,y,ws,dist,sim))

            # early stopping
            if len(rects) > num_stop:
               stop_flg = True
               break

            # save found image
            x, y = chosen_start[j]
            im_2_path = SAVE_DIR+'ws%d_%d_%d.jpg'%(ws, x, y)
            print(im_2_path)
            im_2 = cv2.cvtColor(chosen_img[j], cv2.COLOR_RGB2BGR)
            cv2.imwrite(im_2_path, im_2)

      if stop_flg:
         break

   if len(rects) > 0:
      result_path = result_dir + name.split('.')[0] + '_result.jpg'
      result_img  = cv2.imread(img_2_path)
      grp_rects, weights = cv2.groupRectangles(rects,1,1.5)
      if len(grp_rects) > 0:
         draw_rects = grp_rects
      else:
         draw_rects = rects

      for rect in draw_rects:
         x, y, w, h = rect
         x_end = x + w
         y_end = y + h
         result_img  = cv2.rectangle(result_img, pt1=(x,y), pt2=(x_end,y_end), color=(0,0,255))

      cv2.imwrite(result_path, result_img)

   # if len(rects) == 1:
   #    result_path = result_dir + name.split('.')[0] + '_result.jpg'
   #    result_img  = cv2.imread(img_2_path)
   #    x, y, w, h = rects[0]
   #    x_end = x + w
   #    y_end = y + h
   #    result_img  = cv2.rectangle(result_img, pt1=(x,y), pt2=(x_end,y_end), color=(0,0,255))
   #    cv2.imwrite(result_path, result_img)
   #
   # if len(rects) > 1:
   #    result_path = result_dir + name.split('.')[0] + '_result.jpg'
   #    result_img  = cv2.imread(img_2_path)
   #    rects, weights = cv2.groupRectangles(rects,1,1.5)
   #    x, y, w, h = rects[0]
   #    x_end = x + w
   #    y_end = y + h
   #    result_img  = cv2.rectangle(result_img, pt1=(x,y), pt2=(x_end,y_end), color=(0,0,255))
   #    cv2.imwrite(result_path, result_img)
   dbg_file.close()


   tok = time.time()
   print('Duration: %.3f'%(tok-tik))



