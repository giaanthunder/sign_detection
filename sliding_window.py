import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
plt.axis('off')
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.preprocessing import image



def load_img(path, resize=None):
   img = Image.open(path)
   img = img.convert("RGB")

   if resize is not None:
      size_h = size_w = resize
      img = ImageOps.fit(img, (size_h,size_w), Image.ANTIALIAS)
   else:
      h, w =  img.size
      if h!=w:
         size = min(h, w)
         img = ImageOps.fit(img, (size,size), Image.ANTIALIAS)
   img = np.asarray(img)
   return img


def sliding_window(image, img_patches, step_size, window_size, resize=None, display=False):
   disp = plt.imshow(image)
   img_h, img_w, img_c = image.shape
   win_h, win_w = window_size

   for y in range(0, img_h-win_h, step_size):
      for x in range(0, img_w-win_w, step_size):
         x_end = x + win_h
         y_end = y + win_w
         img = np.asarray(image[y:y_end, x:x_end])
         if resize is not None:
            img = cv2.resize(img, (resize,resize), interpolation = cv2.INTER_CUBIC)
         img_patches.append((x,y,img))

         if display:
            disp_img = np.copy(image)
            disp_img = cv2.rectangle(disp_img, pt1=(x,y), pt2=(x_end,y_end), 
                  color=(0,255,0))
            disp.set_array(disp_img)
            plt.draw()
            plt.pause(0.1)

   return img_patches

def generator(img_1, img_2, patch_points, win_size, resize, batch_size):
   data_size     = len(patch_points)
   num_fullbatch = data_size//batch_size
   remain_batch  = data_size%batch_size
   # full batch
   batch_cnt = 0
   while batch_cnt < num_fullbatch:
      batch_1 = []
      batch_2 = []
      batch_3 = []
      batch_4 = []
      for i in range(batch_size):
         x_start, y_start, x_end, y_end = patch_points[batch_cnt*batch_size+i]
         img2 = np.asarray(img_2[y_start:y_end, x_start:x_end])
         img2 = cv2.resize(img2, (resize,resize), interpolation = cv2.INTER_NEAREST)
         start_point = np.asarray([x_start,y_start])
         end_point   = np.asarray([x_end,y_end])
         batch_1.append(np.copy(img_1))
         batch_2.append(img2)
         batch_3.append(start_point)
         batch_4.append(end_point)

      batch_cnt+=1
      batch_1 = np.stack(batch_1)
      batch_2 = np.stack(batch_2)
      batch_3 = np.stack(batch_3)
      batch_4 = np.stack(batch_4)
      yield (batch_1, batch_2, batch_3, batch_4)

   # last batch
   if remain_batch == 0:
      return
   batch_1 = []
   batch_2 = []
   batch_3 = []
   batch_4 = []
   for i in range(remain_batch):
      x_start, y_start, x_end, y_end = patch_points[batch_cnt*batch_size+i]
      img2 = np.asarray(img_2[y_start:y_end, x_start:x_end])
      img2 = cv2.resize(img2, (resize,resize), interpolation = cv2.INTER_NEAREST)
      start_point = np.asarray([x_start,y_start])
      end_point   = np.asarray([x_end,y_end])
      batch_1.append(np.copy(img_1))
      batch_2.append(img2)
      batch_3.append(start_point)
      batch_4.append(end_point)

   batch_cnt+=1
   batch_1 = np.stack(batch_1)
   batch_2 = np.stack(batch_2)
   batch_3 = np.stack(batch_3)
   batch_4 = np.stack(batch_4)
   yield (batch_1, batch_2, batch_3, batch_4)


class sliding_generator():
   def __init__(self, img_1_path, img_2_path, step_size, window_size, resize=None, batch_size=1):
      img_1 = image.load_img(img_1_path)
      img_1 = img_1.convert("RGB")
      img_1 = img_1.resize((window_size, window_size), Image.ANTIALIAS)
      img_1 = img_1.resize((resize,resize), Image.NEAREST)
      img_1 = image.img_to_array(img_1)


      img_2 = image.load_img(img_2_path)
      img_2 = img_2.convert("RGB")
      img_2 = image.img_to_array(img_2)

      win_h = win_w = window_size
      img_h, img_w, _ = img_2.shape

      patch_points = []
      for y in range(0, img_h-win_h, step_size):
         for x in range(0, img_w-win_w, step_size):
            x_end = x + win_h
            y_end = y + win_w
            patch_points.append((x,y,x_end,y_end))
      self.patch_cnt = len(patch_points)
      self.gen = generator(img_1, img_2, patch_points, window_size, resize, batch_size)


if __name__ == "__main__":
   gsv_dir = 'GSV_Step_1_3_7/'
   gsv_names = os.listdir(gsv_dir)
   gsv_names.sort()

   out_dir = 'out/'

   ws_lst  = [256, 128, 64, 32]

   for name in gsv_names:
      print(name)
      img_2_path  = gsv_dir + name
      img_2 = load_img(img_2_path)
      for i in range(4):
         ws     = ws_lst[i]
         stride = ws // 8

         img_patches = []
         img_2_patches = sliding_window(img_2, img_patches, stride, (ws,ws))
         num_patch = len(img_2_patches)
         print('Number of patch:', num_patch)

         SAVE_DIR = out_dir + '%s/ws%d/'%(name,ws)
         PATH  = gsv_dir + name
         if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
         for i in range(num_patch):
            im_2 = Image.fromarray(img_2_patches[i][2])
            im_2_path = SAVE_DIR+'%d_%d.jpg'%(img_2_patches[i][0],img_2_patches[i][1])
            im_2.save(im_2_path)




