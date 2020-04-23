import os, sys, time, math, shutil
import qatm, slires
from pathlib import Path
import cv2
from PIL import Image
import numpy as np



def draw_boxes(image, boxes, color=(0,255,0)):
    for box in boxes:
        x, y, w, h = box
        x_end = x + w
        y_end = y + h
        image  = cv2.rectangle(image, pt1=(x,y), pt2=(x_end,y_end), color=color)
    return image

def overlap_rectangles(rect1, rect2):
    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    A = {'x1': rect1[0], 'y1': rect1[1], 'x2': rect1[0] + rect1[2], 'y2': rect1[1] + rect1[3], 'w': rect1[2], 'h': rect1[3]}
    B = {'x1': rect2[0], 'y1': rect2[1], 'x2': rect2[0] + rect2[2], 'y2': rect2[1] + rect2[3], 'w': rect2[2], 'h': rect2[3]}

    # overlap between A and B
    SA = A['w'] * A['h']
    SB = B['w'] * B['h']
    SI = np.max([0, 1 + np.min([A['x2'], B['x2']]) - np.max([A['x1'], B['x1']])]) * np.max(
        [0, 1 + np.min([A['y2'], B['y2']]) - np.max([A['y1'], B['y1']])])
    SU = SA + SB - SI
    overlap_AB = float(SI) / float(SU)
    print ('overlap between A and B: %f' % overlap_AB)
    return  overlap_AB

def add_rectangles(rect1, rect2):
    x = int(min(rect1[0], rect2[0]))
    y = int(min(rect1[1], rect2[1]))
    w = int(max(rect1[2], rect2[2]))
    h = int(max(rect1[3], rect2[3]))
    return x,y,w,h

def get_overlap_rect(boxes, rects):
    for box in boxes:
        for rect in rects:
            overlap = overlap_rectangles(box, rect)
            if rect[2] <= 128:
                if overlap > 0.1:
                    x,y,w,h = add_rectangles(box, rect)
                    return [x,y,w,h]
            else:
                x,y,w,h = rect[0], rect[1], rect[2], rect[3]
                return [x,y,w,h]
    return None



image_T_path_qatm = "hit4.jpg"
image_T_path_sliding = "hit2.jpg"

folder = '/media/vy/DATA/Liem/advertising/QATM/result_high/'
# folder = 'gsv_selenium_1/'

imgs = os.listdir(folder)
imgs.sort()

result_dir = "./result"
if not Path(result_dir).is_dir():
    os.mkdir(result_dir)


image_T = cv2.imread(image_T_path_qatm)
image_T = cv2.cvtColor(image_T, cv2.COLOR_BGR2RGB)

image_T_pil = Image.open(image_T_path_sliding)
image_T_pil = image_T_pil.convert('RGB')
image_T_pil = np.array(image_T_pil)

qatm_model = qatm.qatm()
slires_model = slires.slires()


for name in imgs:
    tik = time.time()
    print(name)
    image_S_path = folder + '/' + name

    image_S = cv2.imread(image_S_path)
    image_S = cv2.cvtColor(image_S, cv2.COLOR_BGR2RGB)

    image_S_pil = Image.open(image_S_path)
    image_S_pil = image_S_pil.convert('RGB')
    image_S_pil = np.array(image_S_pil)
    t_qatm = time.time()
    qatm_boxes = qatm_model.get_qatm_boxes(image_T.copy(), image_S.copy(), ws_list=[64,128,256])
    print ("time qatm : ", time.time() - t_qatm)
    slires_boxes = slires_model.get_slires_boxes(image_T_pil.copy(), image_S_pil.copy(), ws_lst = [128,256],thresh_lst=[0.5,0.5])

    image_plot = image_S.copy()

    image_plot = draw_boxes(image_plot, qatm_boxes, (0,0,255))
    image_plot = draw_boxes(image_plot, slires_boxes, (0,255,0))

    final_rect = get_overlap_rect(qatm_boxes, slires_boxes)
    if final_rect is not None:
        draw_boxes(image_plot, [final_rect], (255,0,0))

    # cv2.imshow('Result', image_plot[..., ::-1])
    # cv2.waitKey(0) == ord('q')
    # exit()
    name = os.path.splitext(name)[0]
    save = "{}/{}.jpg".format(result_dir, name)
    image_plot = cv2.cvtColor(image_plot, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save, image_plot)
    tok = time.time()
    print('Time: %.2f'%(tok-tik))






