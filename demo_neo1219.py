import cv2
from scipy import misc
from easydict import EasyDict as edict
from tensorpack import *
import numpy as np
# list split utils
from operator import itemgetter
from itertools import *
import time
import functools
import os
from copy import deepcopy
import pdb
from tensorpack import *

# try:
#     # import models
#     from .classify_frames.train import Model as Model_classify_frames
#     from .detect_text_area.train import Model as Model_detect_text_area
#     from .segment_lines.train import Model as Model_segment_lines
#     from .recognize_sequences.train import Model as Model_recognize_sequences
#     # import configs
#     from .classify_frames.cfgs.config import cfg as cfg_classify_frames
#     from .detect_text_area.cfgs.config import cfg as cfg_detect_text_area
#     from .segment_lines.cfgs.config import cfg as cfg_segment_lines
#     from .recognize_sequences.cfgs.config import cfg as cfg_recognize_sequences
#     from .recognize_sequences.mapper import Mapper
# except Exception:
# import models
from classify_frames.train import Model as Model_classify_frames
from detect.train import Model as Model_detect_text_area
from segment_lines.train import Model as Model_segment_lines
from recognize_sequences.train import Model as Model_recognize_sequences
# import configs
from classify_frames.cfgs.config import cfg as cfg_classify_frames
from detect.cfgs.config import cfg as cfg_detect_text_area
from segment_lines.cfgs.config import cfg as cfg_segment_lines
from recognize_sequences.cfgs.config import cfg as cfg_recognize_sequences
from recognize_sequences.mapper import Mapper

time_record = {}
def timethis(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t_start = time.time()
        res = func(*args, **kw)
        t_end = time.time()
        time_record[func.__name__]= t_end - t_start
        return res
    return wrapper

def cap_video(video_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return [cap.read()[1] for _ in range(total_frame)]

def batch_data(data, batch_size):
    len_data = len(data)
    batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
    print('data will be splitted into {} batches'.format(batch_num))
    batched_data = np.array_split(data, batch_num)
    return batched_data

def classify_frames(inputs, pred_func):
    def preprocess(inputs):
        total_frame = len(inputs)
        assert len(inputs) >= len(cfg_classify_frames.frame_extract_pattern)
        # get resized gray-level frames
        resized_frames = [cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_BGR2GRAY) for i in inputs]
        # generate tensors in shape of (224, 224, c)
        tensors = []
      
        for frame_idx in range(total_frame):
            if frame_idx - margin < 0 or frame_idx + margin >= total_frame:
                continue
            selected_frames = resized_frames[frame_idx - margin:frame_idx + margin + 1]
            # select frames within margins, with shape(pattern_length, 224, 224)
            tensor = np.asarray(selected_frames)
            tensor = tensor.swapaxes(0,2)
            tensors.append(tensor)
        # generate tensors with shape (224, 224, pattern_length)

        return tensors

    def postprocess(preds):
        # 2-class probabilities to predictions
        label_pred = np.argmax(preds, axis = 1)
        # pad head and tail with `edge` mode
        label_pred = np.pad(label_pred, (margin, margin), mode='edge')
        # fill gaps smaller than `max_gap`
        label = [[idx, lb] for idx, lb in enumerate(label_pred)]
        label = [[f,list(g)] for f,g in groupby(label, lambda x:x[1])]
        label = [[i[0], len(i[1])] for i in label]

        chip_l = -1
        chip_r = -1
        inchip = False
        for i in range(len(label)):
            if inchip and label[i][1] >= 10:
                chip_r = i - 1
                inchip = False
                left, right = None, None
                if chip_l - 1 >= 0:
                    left = label[chip_l - 1][0]
                if chip_r + 1 < len(label):
                    right = label[chip_r + 1][0]
                if left is not None and right is not None:
                    if left==right==1:
                        for j in range(chip_l, chip_r + 1):
                            label[j][0] = 1
                    else:
                        for j in range(chip_l, chip_r + 1):
                            label[j][0] = 0
                elif left is not None and right is None:
                    for j in range(chip_l, chip_r + 1):
                        label[j][0] = left
                elif left is None and right is not None:
                    for j in range(chip_l, chip_r + 1):
                        label[j][0] = right
            elif not inchip and label[i][1] < 10:
                chip_l = i
                inchip = True
        new_label = []
        for i in label:
            new_label.extend([i[0]]*i[1])
        return np.array(new_label)
    
    margin = len(cfg_classify_frames.frame_extract_pattern) // 2
    batch_size = cfg_classify_frames.batch_size
    preprocessed = preprocess(inputs)
    batches = batch_data(preprocessed, batch_size)
    batched_preds = [pred_func([i])[0] for i in batches]
    preds = list(np.vstack(batched_preds))
    postprocessed = postprocess(preds)
    
    return postprocessed

def extract_frames(inputs, label):
    
    # extract valid frame indexes
    frame_idx = [idx for idx, lb in enumerate(label) if lb]
    # split into pieces
    frame_idxss = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frame_idx), lambda x: x[0]-x[1])]
    max_blurry_idxs = []
    # for each piece, find max-blurry index
    for frame_idxs in frame_idxss:
        max_blurry = 0
        max_blurry_idx = None
        for i in frame_idxs:
            blurry = cv2.Laplacian(inputs[i], cv2.CV_64F).var()
            if max_blurry < blurry:
                max_blurry = blurry
                max_blurry_idx = i
        max_blurry_idxs.append(max_blurry_idx)
    # collect max-blurry frames and their indexes
    outputs = [[inputs[i], {'frame_idx': i}] for i in max_blurry_idxs]

    return outputs

def detect_text_area(inputs, pred_func, enlarge_ratio=1.1):
    def preprocess(inputs):
        # resize images and convert BGR to RGB
        rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in inputs]
        resized_imgs = [cv2.resize(i, (img_w, img_h)) for i in rgb_imgs]
        spec_mask = [np.zeros((cfg_detect_text_area.n_boxes, img_w // 32, img_h // 32), dtype=float) == 0 for _ in rgb_imgs]
        return resized_imgs


    def postprocess(predictions, img, det_th=None):
        def non_maximum_suppression(boxes, overlapThresh):
            # if there are no boxes, return an empty list
            if len(boxes) == 0:
                return []
            boxes = np.asarray(boxes).astype("float")

            # initialize the list of picked indexes 
            pick = []

            # grab the coordinates of the bounding boxes
            conf = boxes[:,0]
            x1 = boxes[:,1]
            y1 = boxes[:,2]
            x2 = boxes[:,3]
            y2 = boxes[:,4]

            # compute the area of the bounding boxes and sort the bounding
            # boxes by the bottom-right y-coordinate of the bounding box
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(conf)

            # keep looping while some indexes still remain in the indexes
            # list
            while len(idxs) > 0:
                # grab the last index in the indexes list and add the
                # index value to the list of picked indexes
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                # compute the width and height of the bounding box
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                intersection = w * h
                union = area[idxs[:last]] + area[idxs[last]] - intersection

                # compute the ratio of overlap
                # overlap = (w * h) / area[idxs[:last]]
                overlap = intersection / union

                # delete all indexes from the index list that have
                idxs = np.delete(idxs, np.concatenate(([last],
                    np.where(overlap > overlapThresh)[0])))

            # return only the bounding boxes that were picked using the
            # integer data type
            return boxes[pick].astype("float")
        ori_height, ori_width = img.shape[:2]
        cfg = cfg_detect_text_area
        [pred_x, pred_y, pred_w, pred_h, pred_conf, pred_prob] = predictions

        _, box_n, klass_num, grid_h, grid_w = pred_prob.shape

        pred_conf_tile = np.tile(pred_conf, (1, 1, klass_num, 1, 1))
        klass_conf = pred_prob * pred_conf_tile

        width_rate = ori_width / float(cfg.img_w)
        height_rate = ori_height / float(cfg.img_h)

        boxes = {}
        for n in range(box_n):
            for gh in range(grid_h):
                for gw in range(grid_w):

                    k = np.argmax(klass_conf[0, n, :, gh, gw])
                    if klass_conf[0, n, k, gh, gw] < (det_th or cfg.det_th):
                        continue

                    anchor = cfg.anchors[n]
                    w = pred_w[0, n, 0, gh, gw]
                    h = pred_h[0, n, 0, gh, gw]
                    x = pred_x[0, n, 0, gh, gw]
                    y = pred_y[0, n, 0, gh, gw]

                    center_w_cell = gw + x
                    center_h_cell = gh + y
                    box_w_cell = np.exp(w) * anchor[0]
                    box_h_cell = np.exp(h) * anchor[1]

                    center_w_pixel = center_w_cell * 32
                    center_h_pixel = center_h_cell * 32
                    box_w_pixel = box_w_cell * 32
                    box_h_pixel = box_h_cell * 32

                    xmin = float(center_w_pixel - box_w_pixel // 2)
                    ymin = float(center_h_pixel - box_h_pixel // 2)
                    xmax = float(center_w_pixel + box_w_pixel // 2)
                    ymax = float(center_h_pixel + box_h_pixel // 2)
                    xmin = np.max([xmin, 0]) * width_rate
                    ymin = np.max([ymin, 0]) * height_rate
                    xmax = np.min([xmax, float(cfg.img_w)]) * width_rate
                    ymax = np.min([ymax, float(cfg.img_h)]) * height_rate

                    klass = cfg.classes_name[k]
                    if klass not in boxes.keys():
                        boxes[klass] = []

                    box = [klass_conf[0, n, k, gh, gw], xmin, ymin, xmax, ymax]

                    boxes[klass].append(box)

        # do non-maximum-suppresion
        nms_boxes = {}
        if cfg.nms == True:
            for klass, k_boxes in boxes.items():
                k_boxes = non_maximum_suppression(k_boxes, cfg.nms_th)
                nms_boxes[klass] = k_boxes
        else:
            nms_boxes = boxes

        output = []
        for klass, k_boxes in nms_boxes.items():
            for box_idx, each_box in enumerate(k_boxes):
                [conf, xmin, ymin, xmax, ymax] = each_box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2
                width = (xmax - xmin) * enlarge_ratio
                height = (ymax - ymin) * enlarge_ratio
                xmin = np.max([0, int(xcenter - width / 2)])
                ymin = np.max([0, int(ycenter - height / 2)])
                xmax = np.min([ori_width - 1, int(xcenter + width / 2)])
                ymax = np.min([ori_height - 1, int(ycenter + height / 2)])

                cropped_img = img[ymin:ymax, xmin:xmax]
                det_area = [xmin, ymin, xmax, ymax]
                output.extend([cropped_img, {'detect_area': det_area, 'type': klass, 'raw_img': img, 'conf': conf}])
        
        return output

    def _batch_data(data, batch_size):
        batched_data = batch_data(data, batch_size)
        spec_mask = [np.ones((i.shape[0], cfg_detect_text_area.n_boxes, img_h // 32, img_w // 32), dtype=bool) for i in batched_data]
        return list(zip(batched_data, spec_mask))

    img_h, img_w = cfg_detect_text_area.img_h, cfg_detect_text_area.img_w
    batch_size = cfg_detect_text_area.batch_size

    preprocessed = preprocess(inputs)
    batches = _batch_data(preprocessed, batch_size = batch_size)
    # output of model is in order of `['pred_x', 'pred_y', 'pred_w', 'pred_h', 'pred_conf', 'pred_prob']`
    # as feed `batch_size` minibatch into graph, each element will be shape of (batch_size, n_boxes, value, grid_h, grid_w)
    # for `pred_prob`, value is n_classes, for others, value is 1
    # to split (batch_size, n_boxes, value, grid_h, grid_w) tensor into [x] * batch_size list
    # ---WRONG WAY---
    # use `np.asarray()` to create (6, batch_size, n_boxes, value, grid_h, grid_w) tensor and then split on axis 1
    # ---RIGHT WAY---
    # as n_classes > 1, `np.array()` or `np.vstack()` can't concatenate arrays with different size
    # just split each output and then put them into one list
    batched_preds = [pred_func(i) for i in batches]
    # ---WRONG WAY---
    # batched_preds = [np.split(np.array(pred_func(i)), len(i[0]), axis = 0) for i in batches]
    # preds = list(np.vstack(batched_preds))

    # --RIGHT WAY---
    preds = []
    for each_batch in batched_preds:
        size = each_batch[0].shape[0]
        # `['pred_x', 'pred_y', 'pred_w', 'pred_h', 'pred_conf', 'pred_prob']`, and each element is a list of (1, batch_size, n_boxes, value, grid_h, grid_w) tensors
        out = [np.split(i, size) for i in each_batch]
        preds.extend([[out[j][i] for j in range(6)] for i in range(size)])
    postprocessed = [postprocess(preds[i], inputs[i]) for i in range(len(inputs))]
    return postprocessed

def segment_lines(inputs, pred_func):
    def preprocess(inputs):
        def split(img, img_idx):
            top, bottom, left, right = 0,1,2,3
            h_per, w_per = cfg_segment_lines.h, cfg_segment_lines.w
            overlap_top, overlap_bottom, overlap_left, overlap_right  = cfg_segment_lines.overlap
            h, w = img.shape[:2]
            res_imgs = []
            res_informations = []
            h_idx, w_idx = 0, 0
            # split
            for i in range(0, h, h_per):
                for j in range(0, w, w_per):
                    padding_shape = [0,0,0,0]
                    has_pad = False
                    h_idx_start, h_idx_end = i, i + h_per
                    # do padding
                    if h_idx_start - overlap_top < 0:
                        padding_shape[top] = overlap_top - h_idx_start
                        has_pad = True
                        h_idx_start = 0
                    else:
                        h_idx_start -= overlap_top
                    if h_idx_end + overlap_bottom > h:
                        padding_shape[bottom] = h_idx_end + overlap_bottom - h
                        has_pad = True
                        h_idx_end = h
                    else:
                        h_idx_end += overlap_bottom
                    w_idx_start, w_idx_end = j, j + w_per
                    if w_idx_start - overlap_left < 0:
                        padding_shape[left] = overlap_left - w_idx_start
                        has_pad = True
                        w_idx_start = 0
                    else:
                        w_idx_start -= overlap_left
                    if w_idx_end + overlap_right > w:
                        padding_shape[right] = w_idx_end + overlap_right - w
                        has_pad = True
                        w_idx_end = w
                    else:
                        w_idx_end += overlap_right
                    res_img = img[h_idx_start:h_idx_end,w_idx_start:w_idx_end]
                    if has_pad:
                        res_img = np.pad(res_img, ((padding_shape[top], padding_shape[bottom]), (padding_shape[left], padding_shape[right])), 'edge')
                    res_imgs.append(np.expand_dims(res_img, axis = -1))
                    res_informations.append(
                        {
                            'img_idx': img_idx,
                            'h_idx': h_idx,
                            'w_idx': w_idx,
                            'padding_shape': padding_shape
                        })
                    w_idx += 1
                h_idx += 1
                w_idx = 0
            return res_imgs, res_informations

        res_imgs = []
        res_informations = []
        for idx, img in enumerate(inputs):
            if len(img.shape) == 3:
                reshaped = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                reshaped = img
            res_img, res_information = split(reshaped, idx)
            res_imgs.extend(res_img)
            res_informations.extend(res_information)
        return res_imgs, res_informations

    def postprocess(preds, informations):
        def concat(inputs):
            res = []
            # group by h_idx
            grouped = [list(g) for k, g in groupby(inputs, lambda x: x[1]['h_idx'])]
            all_row = []
            for each_row in grouped:
                preds_per_row = [i[0] for i in each_row]
                all_row.append(np.concatenate(preds_per_row, axis = 1))
            preds_per_det_area = np.concatenate(all_row)
            return preds_per_det_area
    
        # cut off overlap part
        overlap_top, overlap_bottom, overlap_left, overlap_right = cfg_segment_lines.overlap
        cropped_preds = []
        for i in zip(preds, informations):
            h, w = i[0].shape[:2]
            padding_top, padding_bottom, padding_left, padding_right = i[1]['padding_shape']
            h_idx_start = max(overlap_top, padding_top)
            h_idx_end = h - max(overlap_bottom, padding_bottom)
            w_idx_start = max(overlap_left, padding_left)
            w_idx_end = w - max(overlap_right, padding_right)
            cropped_preds.append(i[0][h_idx_start:h_idx_end, w_idx_start:w_idx_end])

        # sort // maybe no need
        # and group by img_idx
        zipped = zip(cropped_preds, informations)
        grouped = [list(g) for k, g in groupby(zipped, lambda x:x[1]['img_idx'])]
        res = [np.argmax(concat(i), axis=2) for i in grouped]
        return res
    batch_size = cfg_segment_lines.batch_size

    preprocessed_tensors, preprocessed_informations = preprocess(inputs)
    batches = batch_data(preprocessed_tensors, batch_size)
    batched_preds = [pred_func([i])[0] for i in batches]
    preds = list(np.vstack(batched_preds))
    postprocessed = postprocess(preds, preprocessed_informations)

    return postprocessed


def extract_lines(inputs):
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        
        import numpy as np
        from math import factorial
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

    

    res = []
    for img_idx, each_input in enumerate(inputs):
        img, mask = each_input
        if len(img.shape) == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        H, W = img.shape
        # find all contours
        im2, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for each_contour in contours:
            # isolate each contour and extract
            isolated = np.zeros((H, W), np.uint8)
            cv2.fillPoly(isolated, pts = [each_contour], color = 255)
            # dilate
            kernel = np.zeros((15, 7), np.uint8)
            kernel[:11] = 1
            isolated = cv2.dilate(isolated, kernel)

            isolated_im2, isolated_contours, isolated_hierarchy = cv2.findContours(isolated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for each_isolated_contour in isolated_contours:
                x, y, w, h = cv2.boundingRect(each_isolated_contour)
                h_idx_start = max(y, 0)
                h_idx_end = min(y + h, H - 1)
                w_idx_start = max(x, 0)
                w_idx_end = min(x + w, W - 1)
                if h_idx_start >= h_idx_end or w_idx_start >= w_idx_end:
                    continue
                canvas = img[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                canvas_h, canvas_w = canvas.shape[:2]
                new_canvas = np.zeros((3*canvas_h,canvas_w)) + 255
                # collect all coords in contour
                centers = []
                ceilings = []
                floors = []
                # get contour's ceiling and floor
                for x in range(canvas_w):
                    max_coord = -1
                    min_coord = -1
                    for y in range(canvas_h):
                        if cv2.pointPolygonTest(each_isolated_contour,(x+w_idx_start, y+h_idx_start),False) >= 0:
                            min_coord = y
                            break
                    for y in range(canvas_h-1, -1, -1):
                        if cv2.pointPolygonTest(each_isolated_contour,(x+w_idx_start, y+h_idx_start),False) >= 0:
                            max_coord = y + 1
                            break
                    centers.append((max_coord+min_coord)//2)
                    floors.append(max_coord)
                    ceilings.append(min_coord)
                window_size = canvas_w//4
                if window_size % 2 == 0:
                    window_size -= 1
                window_size = max(window_size, 3)
                smoothed = list(np.round(savitzky_golay(np.array(centers), window_size, 1)))
                # 从新的中点为基准，放置每一列
                center = canvas_h + canvas_h // 2
                for i in range(canvas_w):
                    top = smoothed[i]-ceilings[i]
                    bottom = floors[i]-smoothed[i]
                    new_canvas[int(center-top):int(center+bottom), i] = canvas[int(ceilings[i]):int(floors[i]), i]
                # crop padding part
                for ceiling in range(canvas_h*3):
                    if not (new_canvas[ceiling] == 255).all():
                        break

                for floor in range(canvas_h*3-1, -1, -1):
                    if not (new_canvas[floor] == 255).all():
                        break
                if ceiling > floor:
                    continue
                data = new_canvas[ceiling:floor+1]
                information = {
                    'img_idx': img_idx,
                    'line_area': [w_idx_start, h_idx_start, w_idx_end, h_idx_end],
                    'before_align': canvas,
                    'binary': mask[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                }
                res.append([data, information])
    return res

def new_extract_lines(inputs):
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        # from: http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

   
    res = []
    error_idx = []
    for img_idx, each_input in enumerate(inputs):
        img, mask = each_input
        if len(img.shape) == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        H, W = img.shape
        # find all contours
        im2, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # current_contours_num = len(contours)
        # print("contours num ", current_contours_num)
        error_contours_num = 0
        for each_contour in contours:
            x, y, w, h = cv2.boundingRect(each_contour)
            # print('area:',w*h)

            if w * h <=250:
                error_contours_num += 1
                if error_contours_num == len(contours):
                    error_idx.append(img_idx)
                continue
            
            isolated = np.zeros((H, W), np.uint8)
            cv2.fillPoly(isolated, pts = [each_contour], color = 255)
            # dilate
            kernel = np.zeros((15, 7), np.uint8)
            kernel[:11] = 1
            isolated = cv2.dilate(isolated, kernel)
            isolated_im2, isolated_contours, isolated_hierarchy = cv2.findContours(isolated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for each_isolated_contour in isolated_contours:
                x, y, w, h = cv2.boundingRect(each_isolated_contour)
                # cv2.fillPoly(img, [each_isolated_contour], (255,0,0))
                h_idx_start = max(y, 0)
                h_idx_end = min(y + h, H - 1)
                w_idx_start = max(x, 0)
                w_idx_end = min(x + w, W - 1)
                if h_idx_start >= h_idx_end or w_idx_start >= w_idx_end:
                    continue

                # 从原图多取上下各20像素，不够的靠`mode`='edge'的pad
                if h_idx_start <= 20:
                    new_pad_top = 20 - h_idx_start
                    new_h_idx_start =  0
                else:
                    new_pad_top = 0
                    new_h_idx_start = h_idx_start - 20
                if h_idx_end >= H - 1 - 20:
                    new_pad_bottom = h_idx_end + 20 - H
                    new_h_idx_end = H - 1
                else:
                    new_pad_bottom = 0
                    new_h_idx_end = h_idx_end + 20
                raw_canvas = img[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                canvas = img[new_h_idx_start:new_h_idx_end, w_idx_start:w_idx_end]
                if w_idx_end == 485 and h_idx_end == 1214:
                    print(new_pad_top, new_pad_bottom, H-1-20)
                if new_pad_top > 0 or new_pad_bottom > 0:
                    canvas = np.pad(canvas, ((new_pad_top, new_pad_bottom),(0, 0)), mode = 'edge')
                canvas_h, canvas_w = canvas.shape[:2]
                # 创建大的空画布
                new_canvas = np.zeros((3*canvas_h,canvas_w))
                # collect all coords in contour
                centers = []
                ceilings = []
                floors = []
                # get contour's ceiling and floor
                for x in range(canvas_w):
                    max_coord = -1
                    min_coord = -1
                    for y in range(canvas_h):
                        if cv2.pointPolygonTest(each_isolated_contour,(x+w_idx_start, y+new_h_idx_start+new_pad_top),False) >= 0:
                            min_coord = y
                            break
                    for y in range(canvas_h-1, -1, -1):
                        if cv2.pointPolygonTest(each_isolated_contour,(x+w_idx_start, y+new_h_idx_start+new_pad_top),False) >= 0:
                            max_coord = y + 1
                            break
                    centers.append((max_coord+min_coord)//2)
                    floors.append(max_coord)
                    ceilings.append(min_coord)
                window_size = canvas_w//4
                if window_size % 2 == 0:
                    window_size -= 1
                window_size = max(window_size, 3)
                smoothed = list(np.round(savitzky_golay(np.array(centers), window_size, 1)))
                ceiling, floor = canvas_h, 0
                # 将整个旧的列移到空画布上
                for i in range(canvas_w):
                    new_canvas[int(canvas_h - smoothed[i]):int(2*canvas_h - smoothed[i]), i] = canvas[:, i]
                    ceiling = min(ceiling, int(canvas_h - smoothed[i] + ceilings[i]))
                    floor = max(floor, int(canvas_h - smoothed[i] + floors[i]))

                if ceiling >= floor:
                    continue
                
                res_canvas = np.copy(new_canvas[ceiling:floor])
                new_canvas[ceiling] = 0
                new_canvas[floor] = 0
                # 裁出需要的区域
                data = res_canvas
                # data = new_canvas
                information = {
                    'img_idx': img_idx,
                    'line_area': [w_idx_start, h_idx_start, w_idx_end, h_idx_end],
                    'before_align': new_canvas,
                    'binary': mask[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                }
                res.append([data, information])
    return res, error_idx

def recognize_sequences(inputs, pred_func):
    def preprocess(inputs):
        # resize inputs
        resized = []
        for each_img in inputs:
            if each_img.shape[0] != input_height:
                if input_width != None:
                    resized_img = cv2.resize(each_img, (input_width, input_height))
                else:
                    scale = input_height / each_img.shape[0]
                    resized_img = cv2.resize(each_img, None, fx=scale, fy=scale)
                resized.append(resized_img)
            else:
                resized.append(each_img)
        # pad 1 channel
        imgs = [np.expand_dims(i, -1) for i in resized]
        
        return imgs
    def _batch_data(data, batch_size):
        len_data = len(data)
        batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
        print('data will be splitted into {} batches'.format(batch_num))

        batched_data = []

        for i in range(batch_num):
            batch = []
            maxlen = max(j.shape[1] for j in data[i*batch_size:(i+1)*batch_size])
            seqlen = []
            for j in data[i*batch_size:(i+1)*batch_size]:
                seqlen.append(j.shape[1])
                if j.shape[1] == maxlen:
                    batch.append(j)
                else:
                    batch.append(np.pad(j, ((0,0), (0,maxlen - j.shape[1]), (0,0)), 'constant',constant_values=0))
            batch = np.array(batch)
            seqlen = np.array(seqlen)
            batched_data.append([batch, seqlen])
        
        return batched_data

    def postprocess(preds):
        mapper = Mapper()
        return [mapper.decode_output(i) for i in preds]

    input_height = cfg_recognize_sequences.input_height
    input_width = cfg_recognize_sequences.input_width
    batch_size = cfg_recognize_sequences.batch_size

    preprocessed = preprocess(inputs)
    batches = _batch_data(preprocessed, batch_size)
   
    batched_preds = [pred_func(i)[0] for i in batches]

    preds = [j for i in batched_preds for j in i]
    postprocessed = postprocess(preds)
    
    return postprocessed


class Extractor():
    def __init__(self):
        def _init_models():
            # Load weights
            self.video_path = ''
            weights_classify_frames = SaverRestore('models/classify_frames')
            weights_detect_text_area = SaverRestore('models/detect')
            weights_segment_lines = SaverRestore('models/segment_lines')
            weights_recognize_sequences = SaverRestore('models/recognize_sequences')
            
            # Build graphs
            model_classify_frames = Model_classify_frames()
            model_detect_text_area = Model_detect_text_area()
            model_segment_lines = Model_segment_lines()
            model_recognize_sequences = Model_recognize_sequences()

            # Build predict configs
            config_classify_frames = PredictConfig(session_init = weights_classify_frames, model = model_classify_frames, input_names = ['input'], output_names = ['output'])
            config_detect_text_area = PredictConfig(session_init = weights_detect_text_area, model = model_detect_text_area, input_names = ['input', 'spec_mask'], output_names = ['pred_x', 'pred_y', 'pred_w', 'pred_h', 'pred_conf', 'pred_prob'])
            config_segment_lines = PredictConfig(session_init = weights_segment_lines, model = model_segment_lines, input_names = ['input'], output_names = ['softmax_output'])
            config_recognize_sequences = PredictConfig(session_init = weights_recognize_sequences, model = model_recognize_sequences, input_names = ['feat', 'seqlen'], output_names = ['prediction'])

            # Build predictors
            self.predictor_classify_frames = OfflinePredictor(config_classify_frames)
            self.predictor_detect_text_area = OfflinePredictor(config_detect_text_area)
            self.predictor_segment_lines = OfflinePredictor(config_segment_lines)
            self.predictor_recognize_sequences = OfflinePredictor(config_recognize_sequences)
        _init_models()

    def _cap_video(self,video_path):
        def rotate(img):
            from PIL import Image
            im = Image.fromarray(img)
            im = im.transpose(Image.ROTATE_270)
            return np.array(im)
        self.video_path = video_path
        frames = cap_video(video_path)
        height, width = frames[0].shape[:2]
        if width > height:
            self.frames = [rotate(i) for i in frames]
            self.width = height
            self.height = width
        else:
            self.frames = frames
            self.width = width
            self.height = height
        self.total_frame = len(self.frames)

    def _classify_frames(self):
        print('classifing frames...')
        frames = self.frames
        pred_func = self.predictor_classify_frames
        self.output_classify_frames = classify_frames(frames, pred_func)

    def _extract_frames(self):
        print('extracting frames...')
        frames = self.frames
        label = self.output_classify_frames
        self.output_extract_frames = extract_frames(frames, label)

    def _detect_text_area(self):
        print('detecting text area...')
        inputs = []
        informations = []
        for i in self.output_extract_frames:
            inputs.append(i[0])
            informations.append(i[1])
        pred_func = self.predictor_detect_text_area
        pure_outputs = detect_text_area(inputs, pred_func)
        outputs = []
        for i in range(len(pure_outputs)):
            if len(pure_outputs[i]) > 0:
                data = pure_outputs[i][0]
                added_information = pure_outputs[i][1]
                information = deepcopy(informations[i])
                information.update(added_information)
                outputs.append([data, information])
        self.output_detect_text_area = outputs

    def _segment_lines(self):
        print('segmenting lines ...')
        inputs = []
        outputs = []
        informations = []
        
        for i in self.output_detect_text_area:
            if i[1]['type'] == 'text_area':
                inputs.append(i[0])
                informations.append(i[1])
  
        pred_func = self.predictor_segment_lines
        if len(inputs) > 0:
            pure_outputs = segment_lines(inputs, pred_func)
            for i in range(len(pure_outputs)):
                data = pure_outputs[i]
                information = deepcopy(informations[i])
                information['img'] = inputs[i]
                outputs.append([data, information])
        # pdb.set_trace()

        self.output_segment_lines = outputs
      
    
    def _extract_lines(self):
        print('extracting lines ...')
        inputs = []
        informations = []
       
        for i,j in zip(self.output_detect_text_area, self.output_segment_lines):
            inputs.append([i[0], j[0]])
            informations.append(j[1])
        
        pure_outputs, error_idx_lists = new_extract_lines(inputs)
        if len(error_idx_lists) > 0:
            for err_x in error_idx_lists:
                inputs = inputs.remove(inputs[err_x])
                informations = informations.remove(informations[err_x])



        grouped = [list(g) for k,g in groupby(pure_outputs, lambda x: x[1]['img_idx'])]
        outputs = []
        for i in range(len(grouped)):
            added_information = deepcopy(informations[i])
            for j in grouped[i]:
                data = j[0]
                information = j[1]
                information.pop('img_idx')
                information.update(added_information)
                outputs.append([data, information])
        self.output_extract_lines = outputs

    def _recognize_sequences(self):
        print('recognizing sequences...')
        data = []
        informations = []
        for i in self.output_extract_lines:
            data.append(i[0])
            informations.append(i[1])
        pred_func = self.predictor_recognize_sequences

        pure_outputs = recognize_sequences(data, pred_func)
        outputs = []
        for i in range(len(pure_outputs)):
            data = pure_outputs[i]

            information = informations[i]
            outputs.append([data, information])
        self.output_recognize_sequences = outputs


    def from_video(self, video_path):
        self._cap_video(video_path)
        self._classify_frames()
        self._extract_frames()
        self._detect_text_area()
        self._segment_lines()
        self._extract_lines()
        self._recognize_sequences()
        self.output_type = 'video'

    def from_image(self, img_paths):
        self.output_extract_frames = [[cv2.imread(img_path), idx] for idx, img_path in enumerate(img_paths)]
        self._detect_text_area()
        self._segment_lines()
        self._extract_lines()
        self._recognize_sequences()
        self.output_type = 'images'

    def from_txt(self, txt_path):
        img_paths = []
        lines = open(txt_path, 'r').readlines()
        for i in lines:
            img_paths.append(i[:-1])
        self.from_image(img_paths)
    
    def save(self):
        import os, shutil
        from datetime import datetime
        video_name  = self.video_path.split("/")[-1]
        self.filename = 'output-' + video_name + "-" + datetime.now().strftime('%Y%m%d%H%M%S')
        filename = self.filename
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        os.mkdir(filename)

        dirs = ['extract_frames', 'detect_text_area', 'segment_lines', 'extract_lines', 'recognize_sequences','gui_frames', 'gui_preds']
        if self.output_type == 'video':
            dirs = ['frames', 'classify_frames'] + dirs

        for i in dirs:
            os.mkdir('{}/{}'.format(self.filename, i))

        if self.output_type == 'video':
            # save frames
            for idx, data in enumerate(self.frames):
                cv2.imwrite('{}/frames/{}.png'.format(self.filename, idx), data)

            # save output of classify_frames
            f = open(self.filename + '/classify_frames/prediction.txt', 'w')
            for i in list(self.output_classify_frames):
                f.write(str(i))

        # save output of extract_frames
        for data in self.output_extract_frames:
            cv2.imwrite('{}/extract_frames/{}.png'.format(self.filename, data[1]['frame_idx']), data[0])

        # save output of detect_text_area
        grouped = [list(g) for f,g in groupby(self.output_detect_text_area, lambda x: x[1]['frame_idx'])]
        for group in grouped:
            img = group[0][1]['raw_img']
            for idx, data in enumerate(group):
                x, y, x_end, y_end = data[1]['detect_area']
                cv2.rectangle(img,
                        (x, y),
                        (x_end, y_end),
                        (0, 0, 255),
                        3)

                cv2.imwrite('{}/detect_text_area/{}-{}.png'.format(self.filename,data[1]['frame_idx'], idx), data[0])
            cv2.imwrite('{}/detect_text_area/{}.png'.format(self.filename,data[1]['frame_idx']), img)

        # save output of segment_lines
        for i,j in zip(self.output_detect_text_area, self.output_segment_lines):
            img = i[0]
            print(img.shape)
            img = j[1]['img']
            mask = j[0]
            print(img.shape)
            # mask = np.expand_dims(j[0], -1)
            # mask = np.concatenate((mask, np.zeros_like(mask), np.zeros_like(mask)), axis=2)
            # mask = mask
            boolean = mask == 255
            information = j[1]
            mask = mask == 1
            mask_img = np.zeros(img.shape)
            mask_img[:,:,2][mask] = 255
            canvas = img * 0.7 + mask_img * 0.3

            detect_area_coor = [j[1]['detect_area'][1], j[1]['detect_area'][0], j[1]['detect_area'][3], j[1]['detect_area'][2]]

            # cv2.imwrite('{}/segment_lines/{}-({},{})({},{}).png'.format(self.filename,j[1]['frame_idx'], *(j[1]['detect_area'])), canvas)
            # misc.imsave('{}/segment_lines/{}-({},{})({},{})-mask.png'.format(self.filename,j[1]['frame_idx'], *(j[1]['detect_area'])), mask)
            cv2.imwrite('{}/segment_lines/{}-({},{})({},{}).png'.format(self.filename,j[1]['frame_idx'], *(detect_area_coor)), canvas)
            misc.imsave('{}/segment_lines/{}-({},{})({},{})-mask.png'.format(self.filename,j[1]['frame_idx'], *(detect_area_coor)), mask)

        # save output of extract_lines
        for data in self.output_extract_lines:
            # line_img = cv2.cvtColor(data[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)
            # cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}).png'.format(self.filename, data[1]['frame_idx'], *data[1]['detect_area'], *data[1]['line_area']), data[0])
            # cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}) - before_align.png'.format(self.filename, data[1]['frame_idx'], *data[1]['detect_area'], *data[1]['line_area']), data[1]['before_align'])
            detect_area_coor = [data[1]['detect_area'][1], data[1]['detect_area'][0], data[1]['detect_area'][3], data[1]['detect_area'][2]]
            line_area_coor = [data[1]['line_area'][1], data[1]['line_area'][0], data[1]['line_area'][3], data[1]['line_area'][2]]

            cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}).png'.format(self.filename, data[1]['frame_idx'],\
            detect_area_coor[0], detect_area_coor[1], detect_area_coor[2], detect_area_coor[3], line_area_coor[0], line_area_coor[1], line_area_coor[2], line_area_coor[3]), data[0])
            cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}) - before_align.png'.format(self.filename, data[1]['frame_idx'],\
            detect_area_coor[0], detect_area_coor[1], detect_area_coor[2], detect_area_coor[3], line_area_coor[0], line_area_coor[1], line_area_coor[2], line_area_coor[3]), data[1]['before_align'])
        
        # save output of recognize_sequences
        for data in self.output_recognize_sequences:
            f = open('{}/recognize_sequences/{}-({},{})({},{})-({},{})({},{}).txt'.format(self.filename, data[1]['frame_idx'], *data[1]['detect_area'], *data[1]['line_area']), 'w')
            f.write(data[0])

    def gui(self):
        def sort_areas(l):
            # TODO
            # 文本区域/图/表， 应返回文本区域的，排序后结果
            if len(l) == 1:
                return l
            
        def sort_lines(l):
            # l = sorted(l, key = lambda x: x[3][1])
            l = sorted(l, key = lambda x: (x[1]['line_area'][1], x[1]['line_area'][0]))
            return l
        self.gui_frames = []
        # pdb.set_trace()
        for idx in range(len(self.output_segment_lines)):
            img = np.copy(self.output_extract_frames[idx][0])
            origin_h, origin_w = img.shape[:2]
            mask = np.copy(self.output_segment_lines[idx][0])
            x, y, x_end, y_end = self.output_segment_lines[idx][1]['detect_area']
            filled = mask.astype(np.uint8).copy()
            for line in self.output_recognize_sequences:
                # same frame and same area
                if line[1] == self.output_extract_frames[idx][1] and line[2] == [x, y, x_end, y_end]:
                    txt = ''.join(line[0].split())
                    if len(txt) == 0:
                        cv2.fillPoly(filled, pts =[line[4]], color = 0)
            mask = np.pad(filled, ((y, origin_h - y_end), (x, origin_w - x_end)), 'constant', constant_values = 0)
            mask = mask == 1
            mask_img = np.zeros(img.shape)
            mask_img[:,:,2][mask] = 255
            img = img * 0.7 + mask_img * 0.3
            cv2.rectangle(img, (x,y), (x_end, y_end), (0, 0, 255))
            self.gui_frames.append(img)

        self.gui_preds = []
        # sort by frame_idx, det_area, and line_area
        # pdb.set_trace()
        predictions = self.output_recognize_sequences
        predictions = sorted(predictions, key = lambda pre_ : (pre_[1]['frame_idx'], pre_[1]['detect_area'], pre_[1]['line_area']))

        # predictions = sorted(predictions, key = itemgetter(1,2,3))
        # split by frame_idx
        # predictions_per_frame = [list(g) for k, g in groupby(predictions, lambda x: x[1])]
        predictions_per_frame = [list(g) for k, g in groupby(predictions, lambda x: x[1]['frame_idx'])]
        for i in predictions_per_frame:
            for j in i:
                if j[1]['frame_idx'] != i[0][1]['frame_idx']:
                    print(i)
                    print('---')
                    print(j)
        # 同帧不同的det_area排序，先上后下，先左后右
        # 上下：比较y，左右：比较x 
        # 每帧分别排序，每个det_area分别排序，根据x, y, x_end, y_end来排序
        output_per_frame = []
        for idx, pred_each_frame in enumerate(predictions_per_frame):
            # predictions_per_det_area = [list(g) for k, g in groupby(pred_each_frame, lambda x: x[2])]
            predictions_per_det_area = [list(g) for k, g in groupby(pred_each_frame, lambda x: x[1]['detect_area'])]
            # 按先上后下，先左后右排序
            sorted_areas = sort_areas(predictions_per_det_area)
            # 对每一个det_area
            output_lines = []
            for det_area_idx, pred_each_det_area in enumerate(sorted_areas):
                # 先上后下，先左后右（合并）
                sorted_lines = sort_lines(pred_each_det_area)
                output_lines.append('\n'.join(i[0] for i in sorted_lines if i[0] != ''))
            output_per_frame.append('\n'.join(output_lines))
            self.gui_preds = output_per_frame

        for idx, img in enumerate(self.gui_frames):
            cv2.imwrite('{}/gui_frames/{}.png'.format(self.filename, idx), img)
        for idx, pred in enumerate(self.gui_preds):
            open('{}/gui_preds/{}.txt'.format(self.filename, idx), 'w').write(pred)

if __name__ == '__main__':
    # ==================================================================
    # TEST functions
    # ==================================================================
    # initialize models
    # ext = Extractor()

    # test classify_frames OK！
    # frames = cap_video('test_inputs/classify_frames.mp4')
    # outputs = classify_frames(frames, ext.predictor_classify_frames)
    # print(outputs)

    # test extract_frames OK!
    # frames = list(np.load('test_inputs/Classification/frames.npy'))
    # label = np.load('test_inputs/Classification/label.npy')
    # outputs = extract_frames(frames, label)
    # print(outputs)

    # test detect_text_area OK！test_datasets_outputs
    # inputs = [cv2.imread('test_inputs/Detection/input.png')]
    # outputs = detect_text_area(inputs, ext.predictor_detect_text_area)[0]
    # x, y, x_end, y_end = outputs[0]['detect_area']
    # output = cv2.rectangle(inputs[0], (x, y), (x_end, y_end), (0, 0, 255), 3)
    # cv2.imwrite('test_inputs/Detection/output.png', output)
    # print(outputs)

    # test segment_lines OK!
    # inputs = [misc.imread('test_inputs/Segmentation/input.png', mode = 'L')]
    # outputs = segment_lines(inputs, ext.predictor_segment_lines)[0]
    # misc.imsave('test_inputs/Segmentation/output.png', outputs)
    # print(outputs)

    # test extract_lines OK!
    # img = misc.imread('test_inputs/Extraction/img.png', mode = 'L')
    # mask = misc.imread('test_inputs/Extraction/mask.png', mode = 'L')
    # outputs = new_extract_lines([[img, mask]])
    # print(len(outputs))
    # for idx, output in enumerate(outputs):
    #     misc.imsave('test_inputs/Extraction/output/{}.png'.format(idx), output[0])
    #     # misc.imsave('test_inputs/Extraction/output/{}_raw.png'.format(idx), output[1]['before_align'])

    # test recognize_sequences OK!test_datasets_outputs
    # inputs = [cv2.imread('test_inputs/SeqRecog/img.png')]
    # outputs = recognize_sequences(inputs, ext.predictor_recognize_sequences)[0]
    # print(outputs)
    # ==================================================================
    # END of TEST functions
    # ==================================================================

    # ==================================================================
    # START of Generating YOLOv2 output
    # ==================================================================
    # import shutil
    # video_path_collections = [os.path.join(dirpath,filename) for dirpath, dirnames, filenames in os.walk('test_dataset') for filename in filenames]
    # ext = Extractor()

    # for i in video_path_collections:
    #     filename = i.split('/')[-1].split('.')[0]
    #     print('='*5, filename, '='*5)
    #     target = os.path.join('detection_output',filename)
    #     if os.path.exists(target):
    #         shutil.rmtree(target)
    #     os.mkdir(target)
    #     ext.from_video(i)
    #     cnt = 0
    #     for i in ext.output_detect_text_area:
            # img = i[0]
            # # x, y, x_end, y_end = outputs[0]['detect_area']
            # # output = cv2.rectangle(inputs[0], (x, y), (x_end, y_end), (0, 0, 255), 3)
            # # cv2.imwrite('test_inputs/Detection/output.png', output)
            # # print(outputs)
            # misc.imsave(os.path.join(target, '{}.png'.format(cnt)), img_after_align)
            # # misc.imsave(os.path.join(target, '{}-before_align.png'.format(cnt)), img_before_align)
            # # misc.imsave(os.path.join(target, '{}-img_binary.png'.format(cnt)), img_binary)
            # f = open(os.path.join(target, '{}.txt'.format(cnt)), 'w')
            # f.write(txt)
            # cnt += 1
    # ==================================================================
    # END of Generating YOLOv2 output
    # ==================================================================
    # ==================================================================
    # START of Generating test_dataset
    # ==================================================================
    # import shutil
    # video_path_collections = [os.path.join(dirpath,filename) for dirpath, dirnames, filenames in os.walk('test_dataset') for filename in filenames]
    # ext = Extractor()

    # for i in video_path_collections:
    #     filename = i.split('/')[-1].split('.')[0]
    #     print('='*5, filename, '='*5)
    #     target = os.path.join('detection_output',filename)
    #     if os.path.exists(target):
    #         shutil.rmtree(target)
    #     os.mkdir(target)
    #     ext.from_video(i)
    #     cnt = 0
    #     for i,j in zip(ext.output_extract_lines, ext.output_recognize_sequences):
    #         img_after_align = i[0]
    #         # img_before_align = i[1]['before_align']
    #         # img_binary = i[1]['binary']
    #         txt = j[0]
    #         print(img_after_align.shape, txt)
    #         misc.imsave(os.path.join(target, '{}.png'.format(cnt)), img_after_align)
    #         # misc.imsave(os.path.join(target, '{}-before_align.png'.format(cnt)), img_before_align)
    #         # misc.imsave(os.path.join(target, '{}-img_binary.png'.format(cnt)), img_binary)
    #         f = open(os.path.join(target, '{}.txt'.format(cnt)), 'w')
    #         f.write(txt)
    #         cnt += 1
    # ==================================================================
    # END of Generating test_dataset
    # ==================================================================

    # ==================================================================
    # START of augmenting SeqRecog data
    # ==================================================================
    # import shutil
    # ext = Extractor()
    # raw_data_dir = 'output'
    # target_data_dir = 'data_new'
    # dataset_name = 'data_new'

    # # clear
    # if os.path.exists(target_data_dir):
    #     shutil.rmtree(target_data_dir)
    # os.mkdir(target_data_dir)

    # # collect raw data
    # raw_data_path_collections = []
    # files = os.listdir(raw_data_dir)
    # for file in files:
    #     if file.endswith('txt'):
    #         shutil.copy(os.path.join(raw_data_dir, file), os.path.join(target_data_dir, file))
    #         continue
    #     name = file
    #     raw_data_path = os.path.join(raw_data_dir, name)
    #     raw_data_path_collections.append(raw_data_path)
    # num_raw_data = len(raw_data_path_collections)
    # print('Number of raw data:', num_raw_data)
    # target_data_path_collections = []
    # for idx, each_raw_data_path in enumerate(raw_data_path_collections):
    #     img = misc.imread(each_raw_data_path, mode = 'L')
    #     mask = segment_lines([img], ext.predictor_segment_lines)[0]
    #     after = extract_lines([[img, mask]])
    #     if len(after) == 0:
    #         continue
    #     for each_out in after:
    #         if each_out[0].shape[0] <= 10:
    #             continue
    #         output = each_out[0]
    #     target_path = os.path.join(target_data_dir, each_raw_data_path.split('/')[-1])
    #     misc.imsave(target_path, output)
    #     target_data_path_collections.append(target_path+'\n')
    #     print(idx, num_raw_data, sep='/')

    # dataset_name = 'data'
    # target_data_path_collections = [os.path.join('data_new', i) + '\n' for i in os.listdir('data_new')]
    # total_num = len(target_data_path_collections)
    # # split into training set and test set
    # test_ratio = 0.1
    # # total_num = len(target_data_path_collections)
    # test_num = int(test_ratio * total_num)
    # train_num = total_num - test_num
    # train_records = target_data_path_collections[0:train_num]
    # test_records = target_data_path_collections[train_num:]

    # # save to text file
    # all_out_file = open(dataset_name + '_all.txt', 'w')
    # for record in target_data_path_collections:
    #     all_out_file.write(record)
    # all_out_file.close()

    # train_out_file = open(dataset_name + '_train.txt', 'w')
    # for record in train_records:
    #     train_out_file.write(record)
    # train_out_file.close()

    # test_out_file = open(dataset_name + '_test.txt', 'w')
    # for record in test_records:
    #     test_out_file.write(record)
    # test_out_file.close()
    # ==================================================================
    # END of augmenting SeqRecog data
    # ==================================================================

    ext = Extractor()
    # ext.from_video('test_dataset/第一批_20170911/VID_20170911_131636.mp4')
    ext.from_video('/home/user/VideoText/classify_frames/label_tool/raw_videos/data_20171127/VID_20171127_094148.mp4')
    ext.save()
