import os, sys, time
import numpy as np
from PIL import Image, ImageOps
import cv2

def get_batch(img_1, img_2, patch_points, resize, batch_size, batch_cnt):
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

    batch_1 = np.stack(batch_1)
    batch_2 = np.stack(batch_2)
    batch_3 = np.stack(batch_3)
    batch_4 = np.stack(batch_4)
    batch = (batch_1, batch_2, batch_3, batch_4)
    return batch


def generator(img_1, img_2, patch_points, win_size, resize, batch_size):
    data_size      = len(patch_points)
    num_fullbatch = data_size//batch_size
    remain_batch  = data_size%batch_size
    # full batch
    batch_cnt = 0
    while batch_cnt < num_fullbatch:
        batch = get_batch(img_1, img_2, patch_points, resize, batch_size, batch_cnt)
        batch_cnt+=1
        yield batch

    # last batch
    if remain_batch == 0:
        return
    batch = get_batch(img_1, img_2, patch_points, resize, remain_batch, batch_cnt)
    yield batch

def add_win_pos(patch_points, x, y, win_w, win_h):
    x_end = x + win_w
    y_end = y + win_h
    patch_points.append((x,y,x_end,y_end))


class sliding_generator():
    def __init__(self, img_1, img_2, step_size, window_size, resize=None, batch_size=1):
        win_h = win_w = window_size
        img_h, img_w, _ = img_2.shape

        patch_points = []
        for y in range(0, img_h-win_h, step_size):
            for x in range(0, img_w-win_w, step_size):
                add_win_pos(patch_points, x, y, win_w, win_h)
            x = img_w - win_w
            add_win_pos(patch_points, x, y, win_w, win_h)
        
        y = img_h - win_h
        for x in range(0, img_w-win_w, step_size):
            add_win_pos(patch_points, x, y, win_w, win_h)
        x = img_w - win_w
        add_win_pos(patch_points, x, y, win_w, win_h)

        x = img_w - win_w
        y = img_h - win_h
        add_win_pos(patch_points, x, y, win_w, win_h)


        self.patch_cnt = len(patch_points)
        self.batch_cnt = self.patch_cnt//batch_size + (0 if (self.patch_cnt%batch_size)==0 else 1)
        self.gen = generator(img_1, img_2, patch_points, window_size, resize, batch_size)





