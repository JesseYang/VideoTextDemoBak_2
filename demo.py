import cv2
from scipy import misc
from easydict import EasyDict as edict
from tensorpack import *
import numpy as np

# list split utils
from operator import itemgetter
from itertools import *

import time

# import models
from classify_frames.train import Model as Model_classify_frames
from detect_text_area.train import Model as Model_detect_text_area
from segment_lines.train import Model as Model_segment_lines
from recognize_sequences.train import Model as Model_recognize_sequences

# import configs
from classify_frames.cfgs.config import cfg as cfg_classify_frames
from detect_text_area.cfgs.config import cfg as cfg_detect_text_area
from segment_lines.cfgs.config import cfg as cfg_segment_lines
from recognize_sequences.cfgs.config import cfg as cfg_recognize_sequences

# import utils
from recognize_sequences.mapper import Mapper

import pdb

import functools
import os
time_record = {}

def timming(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t_start = time.time()
        res = func(*args, **kw)
        t_end = time.time()
        time_record[func.__name__]= t_end - t_start
        return res
    return wrapper

class Extractor():
    @timming
    def __init__(self):
        # Load weights
        weights_classify_frames = SaverRestore('models/classify_frames')
        weights_detect_text_area = SaverRestore('models/detect_text_area')
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
        print('Building classification predictor...')
        self._classify_frames = OfflinePredictor(config_classify_frames)
        print('Building detection predictor...')
        self._detect_text_area = OfflinePredictor(config_detect_text_area)
        print('Building segmentation predictor...')
        self._segment_lines = OfflinePredictor(config_segment_lines)
        print('Building sequences recognization predictor...')
        self._recognize_sequences = OfflinePredictor(config_recognize_sequences)

        self.kernel = np.zeros((15, 7), np.uint8)
        self.kernel[:11] = 1


    @timming
    def __cap_video(self,video_path):
        cap = cv2.VideoCapture(video_path)
        self.width, self.height = int(cap.get(3)), int(cap.get(4))
        self.total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = [cap.read()[1] for _ in range(self.total_frame)]
        if self.width > self.height:
            self.frames = [np.rot90(e, k=-1).copy() for e in self.frames]

    @timming
    def __classify_frames(self):
        def postprocess(preds):
            label = [[idx, lb] for idx, lb in enumerate(preds)]
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
        print('classifing frames...')
        """将`self.frames`输出`self.output_classify_frames`
        """
        # get resized gray-level frames
        resized_frames = [cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_BGR2GRAY) for i in self.frames]
        # generate tensors in shape of (224, 224, c)
        tensors = []
        margin = len(cfg_classify_frames.frame_extract_pattern) // 2
        for frame_idx in range(self.total_frame):
            if frame_idx - margin < 0 or frame_idx + margin >= self.total_frame:
                continue
            selected_frames = resized_frames[frame_idx - margin:frame_idx + margin + 1]
           
            tensor = np.asarray(selected_frames)
            tensor = tensor.swapaxes(0,2)
            tensors.append(tensor)
        # batch data
        batched_data = self.__batch_data(tensors, cfg_classify_frames.batch_size)

        output_classify_frames = []
        for each_batch in batched_data:
            output_classify_frames.append(self._classify_frames([each_batch])[0])
        output_classify_frames_stacked = np.vstack(output_classify_frames)
        # 二分类probability转prediction
        label_pred = np.argmax(output_classify_frames_stacked, axis = 1)

        # 对于视频序列两侧没有预测的帧补满
        label_pred = np.pad(label_pred, (margin, margin), mode='edge')

        # 填补小于`max_gap`的空隙
        

        self.output_classify_frames = postprocess(label_pred)
        
        
    @timming
    def __extract_frames(self):
        """

        # Input
            [img, ...]
            label

        # Output
            [[img, frame_idx], ...]

        """
        print('extracting frames...')
        label = list(self.output_classify_frames)
        frame_idx = [idx for idx, lb in enumerate(label) if lb]
        frame_idxss = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frame_idx), lambda x: x[0]-x[1])]
        max_blurry_idxs = []
        for frame_idxs in frame_idxss:
            max_blurry = 0
            max_blurry_idx = None
            for i in frame_idxs:
                blurry = cv2.Laplacian(self.frames[i], cv2.CV_64F).var()
                if max_blurry < blurry:
                    max_blurry = blurry
                    max_blurry_idx = i
            max_blurry_idxs.append(max_blurry_idx)

        self.output_extract_frames = [[self.frames[i], i] for i in max_blurry_idxs]
        print('extract {} frames'.format(len(max_blurry_idxs)))


    @timming
    def __detect_text_area(self):
        """

        # Input


        # Output
            [ [img, frame_idx, det_area], ...]
                - det_area: [x, y, x_end, y_end]

        """
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

        def postprocess(predictions, img, det_th=None):
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

            
            return nms_boxes

        def draw_result(image, boxes):
            colors = [(255,0,0), (0,255,0), (0,0,255),
                    (255,255,0), (255,0,255), (0,255,255),
                    (122,0,0), (0,122,0), (0,0,122),
                    (122,122,0), (122,0,122), (0,122,122)]

            text_colors = [(0,255,255), (255,0,255), (255,255,0),
                        (0,0,255), (0,255,0), (255,0,0),
                        (0,122,122), (122,0,122), (122,122,0),
                        (0,0,122), (0,122,0), (122,0,0)]

            image_result = np.copy(image)
            k_idx = 0
            for klass, k_boxes in boxes.items():
                for k_box in k_boxes:

                    [conf, xmin, ymin, xmax, ymax] = k_box

                    label_height = 14
                    label_width = len(klass) * 10
        
                    cv2.rectangle(image_result,
                                (int(xmin), int(ymin)),
                                (int(xmax), int(ymax)),
                                colors[k_idx % len(colors)],
                                3)
                    cv2.rectangle(image_result,
                                (int(xmin) - 2, int(ymin) - label_height),
                                (int(xmin) + label_width, int(ymin)),
                                colors[k_idx % len(colors)],
                                -1)
                    cv2.putText(image_result,
                                klass,
                                (int(xmin), int(ymin) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                text_colors[k_idx % len(text_colors)])
                k_idx += 1

            return image_result

        print('detecting text area...')

        # copy
        input_raw = np.copy(self.output_extract_frames)

        # resize images and convert BGR to RGB, remaining original images in the meantime
        # shape: [ [resized_img, original_img, frame_idx], ... ]
        data = []
        for each_input in input_raw:
            cvted_img = cv2.cvtColor(each_input[0], cv2.COLOR_BGR2RGB)
            img = cv2.resize(cvted_img, (cfg_detect_text_area.img_h, cfg_detect_text_area.img_w))
            original_img = each_input[0]
            frame_idx = each_input[1]
            data.append([img, original_img, frame_idx])

        # batch data
        # shape: [ [ [img, frame_idx, h_idx, w_idx, padding], ... ], ... ]
        batched_data = self.__batch_data(data, cfg_detect_text_area.batch_size)

        # extract pure images and feed graph
        # shape: [ [img, ... ], ... ]
        pure_preds = []
        for each_batch in batched_data:
            tensor = np.array([i[0] for i in each_batch])
            # generate spec_mask
            spec_mask = np.zeros((len(tensor), cfg_detect_text_area.n_boxes, cfg_detect_text_area.img_w // 32, cfg_detect_text_area.img_h // 32), dtype=float) == 0
            preds = self._detect_text_area([tensor, spec_mask])
            pure_preds.extend(np.split(np.array(preds), len(tensor), axis=1))
    
        # group predictions and their informations
        # shape: [ [pred, original_img, frame_idx], ... ]
        for data_idx in range(len(pure_preds)):
            data[data_idx][0] = pure_preds[data_idx]

        # postprocess
        output = []
        self.bounding_boxes = []
        for each_data in data:
            boxes = postprocess(each_data[0], img = each_data[1], det_th=0.25)
            many_boxes = []
            for klass, k_boxes in boxes.items():
                for each_box in k_boxes:
                    [conf, xmin, ymin, xmax, ymax] = each_box
                    x, y, x_end, y_end = int(xmin), int(ymin), int(xmax), int(ymax)
                    many_boxes.append([x, y, x_end, y_end])
                    cropped_img = each_data[1][y:y_end, x:x_end]
                    frame_idx = each_data[2]
                    det_area = [x, y, x_end, y_end]
                    output.append([cropped_img, frame_idx, det_area])
            self.bounding_boxes.append([many_boxes, each_data[2]])
        
        # shape: [ [img, frame_idx, det_area], ...]
        #   det_area: [x, y, x_end, y_end]
        self.output_detect_text_area = output


    @timming
    def __segment_lines(self):
        """

        # Input
            [ [img, frame_idx, det_area], ...]
                - det_area: [x, y, x_end, y_end]

        # Output
            [ [mask, frame_idx, det_area], ...]
                - det_area: [x, y, x_end, y_end]

        """
        def split(img_frameidx):
            top, bottom, left, right = 0,1,2,3
            img, frame_idx, det_area = img_frameidx
            h_per, w_per = cfg_segment_lines.h, cfg_segment_lines.w
            overlap_top, overlap_bottom, overlap_left, overlap_right  = cfg_segment_lines.overlap
            h, w = img.shape[:2]
            res = []
            h_idx, w_idx = 0, 0
            for i in range(0, h, h_per):
                for j in range(0, w, w_per):
                    padding_shape = [0,0,0,0]
                    has_pad = False
                    h_idx_start, h_idx_end = i, i + h_per
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
                    # print('[{}:{},{}:{}] with padding (({}, {}), ({}, {}))'.format(h_idx_start, h_idx_end, w_idx_start, w_idx_end, padding_shape[top], padding_shape[bottom], padding_shape[left], padding_shape[right]))
                    res_img = img[h_idx_start:h_idx_end,w_idx_start:w_idx_end]
                    if has_pad:
                        res_img = np.pad(res_img, ((padding_shape[top], padding_shape[bottom]), (padding_shape[left], padding_shape[right])), 'edge')
                    res.append([res_img, frame_idx, det_area, h_idx, w_idx, padding_shape])
                    w_idx += 1
                h_idx += 1
                w_idx = 0
            return res

        def concat(preds):
            res = []
            # grouped by frame_idx
            frame_group = [list(g) for k, g in groupby(preds, lambda x: x[1])]
            for each_frame in frame_group:
                frame_idx = each_frame[0][1]
                # grouped by det_area
                det_area_group = [list(g) for k, g in groupby(each_frame, lambda x: x[2])]
                for each_det_area in det_area_group:
                    det_area_borders = each_det_area[0][2]
                    # grouped by h_idx
                    h_group = [list(g) for k, g in groupby(each_det_area, lambda x: x[3])]
                    all_row = []
                    for each_row in h_group:
                        preds_per_row =[j[0] for j in each_row]
                        all_row.append(np.concatenate(preds_per_row, axis = 1))

                    preds_per_det_area = np.concatenate(all_row)
                    res.append([preds_per_det_area, frame_idx, det_area_borders])
 
            return res

        print('segmenting lines...')
        # copy
        input = np.copy(self.output_detect_text_area)

        # convert data from 3 channels to 1 channels(gray level)
        # shape: [ [img, frame_idx, det_area], ... ]
        data = [[cv2.cvtColor(i[0], cv2.COLOR_BGR2GRAY), i[1], i[2]] for i in input]

        # blur data
        # data = [[cv2.blur(i[0], (14,5)), i[1], i[2]] for i in data]

        # crop each frame evenly, with overlaps
        # shape: [ [img, frame_idx, h_idx, w_idx, padding], ... ]
        splitted_imgs = []
        for i in data:
            splitted_imgs.extend(split(i))

        # batch data
        # shape: [ [ [img, frame_idx, h_idx, w_idx, padding], ... ], ... ]
        batched_data = self.__batch_data(splitted_imgs, cfg_segment_lines.batch_size)

        # extract pure images and feed graph
        # shape: [ [img, ... ], ... ]
        pure_preds = []
        for each_batch in batched_data:
            tensor = [i[0] for i in each_batch]
            tensor = np.expand_dims(tensor, -1)
            pure_preds.extend(self._segment_lines([tensor])[0])

        # group predictions and their informations
        # shape: [ [pred, frame_idx, h_idx, w_idx, padding], ... ]
        splitted_preds = []
        for i in range(len(pure_preds)):
            frame_idx, det_area, h_idx, w_idx, padding = splitted_imgs[i][1:]
            splitted_preds.append([pure_preds[i], frame_idx, det_area, h_idx, w_idx, padding])

        # cut off overlap part
        # shape: [ [pred, frame_idx, h_idx, w_idx], ... ]
        cropped_preds = []
        overlap_top, overlap_bottom, overlap_left, overlap_right  = cfg_segment_lines.overlap
        for i in splitted_preds:
            pred, frame_idx, det_area, h_idx, w_idx, padding = i
            h, w = pred.shape[:2]
            padding_top, padding_bottom, padding_left, padding_right = padding
            h_idx_start = max(overlap_top, padding_top)
            h_idx_end = h - max(overlap_bottom, padding_bottom)
            w_idx_start = max(overlap_left, padding_left)
            w_idx_end = w - max(overlap_right, padding_right)
            cropped_preds.append([pred[h_idx_start:h_idx_end, w_idx_start:w_idx_end], frame_idx, det_area, h_idx, w_idx])

        # concatenate parts of each frame
        # shape: [ [pred, frame_idx, det_area], ... ]
        preds = concat(cropped_preds)

        # CRF and get output images
        # shape: [ [output_img, frame_idx, det_area], ... ]
        crf_result = []
        self.output_segment_lines = []
        if cfg_segment_lines.crf:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
            for i in range(len(preds)):
                d = dcrf.DenseCRF2D(data[i][0].shape[1], data[i][0].shape[0], 2)
                # set unary potential
                predictions = np.transpose(preds[i][0], (2, 0, 1))
                U = unary_from_softmax(predictions)
                d.setUnaryEnergy(U)

                # set pairwise potential
                # This creates the color-independent features and then add them to the CRF
                d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NORMALIZE_SYMMETRIC)
                # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
                # output_detect_text_area is sliced from original image, so in memory it does not saved as C-contiguous fashion, which is needed here
                c_contiguous = np.array(self.output_detect_text_area[i][0])
                d.addPairwiseBilateral(sxy=(8, 8), srgb=(13, 13, 13), rgbim=c_contiguous,
                                    compat=10,
                                    kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NORMALIZE_SYMMETRIC)

                iter_num = 5
                result = np.argmax(d.inference(iter_num), axis=0)
                result = np.reshape(result, (data[i][0].shape[0], data[i][0].shape[1]))
                self.output_segment_lines.append([result, preds[i][1], preds[i][2]])
        else:
            self.output_segment_lines = [(np.argmax(i[0], axis=2), i[1], i[2]) for i in preds]


    @timming
    def __extract_lines(self):
        """

        # Input
            [[img, frame_idx, det_area], ...]
                - det_area: [x, y, x_end, y_end]
            [[mask, frame_idx, det_area], ...]
                - det_area: [x, y, x_end, y_end]

        # Output
            [[img, frame_idx, det_area, line_area], ...]
                - det_area: [x, y, x_end, y_end]
                - line_area: [x, y, x_end, y_end]

        """
        def sort(imgs):
            """sort images

            # Arguments
                imgs: `[img, x, y, w, h]`

            # Returns
                res_imgs: sorted images
            """
            imgs.sort(key=lambda x:x[2])
            for i in range(len(imgs)-1):
                # find overlaps, sort by x
                j = 0
                while i+j < len(imgs)-1 and imgs[i+j][2] + imgs[i+j][4] > imgs[i+j+1][2]:
                    j += 1
                if j > 0:
                    imgs[i:i+j+1] = sorted(imgs[i:i+j+1], key=lambda x:x[1])
            res_imgs = [i[0] for i in imgs]

            return res_imgs

        print('extracting lines...')
        imgs = np.copy(self.output_detect_text_area)
        # dirty way to deep copy
        masks = np.copy(self.output_segment_lines)
        all_pieces = []
        self.min_area_rects = []
        for idx in range(len(imgs)):
            img = imgs[idx][0]
            mask, frame_idx, det_area = masks[idx]
            mask = mask.astype(np.uint8)
    
            # find all lists of contours
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            H, W = img.shape[:2]
            boxes = []
            for jdx, each_contour in enumerate(contours):
                # isolate
                blank = np.zeros((H, W), np.uint8)
                cv2.fillPoly(blank, pts =[each_contour], color = 255)
                # TODO: 还没测试过
                # 按照行高来变化kernel size
                x,y,w,h = cv2.boundingRect(each_contour)
                # dilate
                kernel = np.vstack((np.zeros((h//2,1), np.uint8), np.ones((h//2, 1), np.uint8)))
                kernel = self.kernel
                blank = blank.astype(np.uint8)
                blank = cv2.dilate(blank, kernel)
                # get contour again
                child_im2, child_contours, child_hierarchy = cv2.findContours(blank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                for each_child_contour in child_contours:
                # ======================================================================
                # This is for drawing rectangles on video frames

                # rect = cv2.minAreaRect(each_contour)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # offset = np.concatenate([[[det_area[0], det_area[1]]]*4], axis=1)
                # box += offset
                # boxes.append(box)
                # ======================================================================
                    x,y,w,h = cv2.boundingRect(each_child_contour)
                    h_idx_start = max(y, 0)
                    h_idx_end = min(y + h, H - 1)
                    w_idx_start = max(x, 0)
                    w_idx_end = min(x + w, W - 1)
                    if h_idx_start >= h_idx_end or w_idx_start >= w_idx_end:
                        continue

                    canvas = img[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                    # generate a matrix with same shape of image, but every element is a coordinate
                    # coords = np.indices((H,W)).transpose((1,2,0))
                    # boolean = cv2.pointPolygonTest(each_contour,coords,False) >= 0
                    all_points = []
                    maxLength = 0
                    for ix in range(w_idx_end - w_idx_start):
                        # get all points of a column
                        line = []
                        for iy in range(h_idx_end - h_idx_start):
                            if cv2.pointPolygonTest(each_child_contour,(ix+w_idx_start,iy+h_idx_start),False) >= 0:
                                line.append(canvas[iy][ix])
                        line = np.array(line)
                        # length of pixels
                        maxLength = max(maxLength, line.shape[0])
                        all_points.append(line)
                    
                    all_points = np.asarray(all_points)
                    # pad every column to fixed size
                    res = []
                    if len(img.shape) == 3:
                        for j in all_points:
                            if j.shape[0] == 0:
                                continue
                            pad_prev = (maxLength - j.shape[0]) // 2
                            pad_post = maxLength - pad_prev - j.shape[0]
                            pad_prev = np.reshape(np.array([255]*pad_prev*3),(pad_prev,3))
                            pad_post = np.reshape(np.array([255]*pad_post*3),(pad_post,3))
                            line = np.vstack((pad_prev, j, pad_post))
                            res.append(line)
                        all_points = np.array(res)
                        all_points = np.transpose(all_points,(1,0,2))
                    else:
                        for j in all_points:
                            pad_prev = (maxLength - j.shape[0]) // 2
                            pad_post = maxLength - pad_prev - j.shape[0]
                            pad_prev = np.array([255]*pad_prev)
                            pad_post = np.array([255]*pad_post)
                            line = np.concatenate((pad_prev, j, pad_post), axis=0)
                            res.append(line)
                        all_points = np.array(res)
                        all_points = np.transpose(all_points)
                    if all_points.shape[0] < all_points.shape[1]:
                        line_area = [w_idx_start,h_idx_start,w_idx_end,h_idx_end]
                        all_pieces.append([all_points, frame_idx, det_area, line_area, each_child_contour])
            # self.min_area_rects.append([boxes, frame_idx])

        # shape: [ [img, frame_idx, x, y, x_end, y_end], ... ]
        self.output_extract_lines = all_pieces
        print('extract {} lines'.format(len(self.output_extract_lines)))


    @timming
    def __recognize_sequences(self):
        def postprocess(txt):
            return ' '.join(txt.split())
        print('recognizing sequences...')
        mapper = Mapper()
        input_height = cfg_recognize_sequences.input_height
        input_width = cfg_recognize_sequences.input_width
        # convert data from 3 channels to 1 channels(gray level)
        # shape: [ [img, frame_idx, det_area, line_area], ... ]
        imgs = [[cv2.cvtColor(i[0].astype(np.uint8), cv2.COLOR_BGR2GRAY), i[1], i[2], i[3], i[4]] for i in self.output_extract_lines]
        # resize images to same height
        for each_img in imgs:
            if each_img[0].shape[0] != input_height:
                if input_width:
                    each_img[0] = cv2.resize(each_img[0], (input_width, input_height))
                else:
                    scale = input_height / each_img[0].shape[0]
                    each_img[0] = cv2.resize(each_img[0], None, fx=scale, fy=scale)
        
        # batch data
        batched_imgs = self.__batch_data(imgs, cfg_recognize_sequences.batch_size)

        # extract pure images and feed graph
        # shape: [ [img, ... ], ... ]
        pure_preds = []
        for each_batch in batched_imgs:
            tensor = [i[0] for i in each_batch]
            maxlen = max([i.shape[1] for i in tensor])
            tensor = [np.pad(i, ((0,0),(0,maxlen - i.shape[1])), 'constant',constant_values=255) for i in tensor]
            tensor = np.expand_dims(tensor, -1)
            seqlen = np.array([maxlen] * tensor.shape[0])
            pred = self._recognize_sequences([tensor, seqlen])[0]
            pure_preds.extend([mapper.decode_output(i) for i in pred])

        # group predictions and their informations
        # shape: [ [pred, frame_idx, x, y, x_end, y_end], ... ]
        preds = []
        for i in range(len(pure_preds)):
            frame_idx, det_area, line_area, each_child_contour = imgs[i][1:]
            after_postprocess = postprocess(pure_preds[i])
            preds.append([after_postprocess, frame_idx, det_area, line_area, each_child_contour])

        self.output_recognize_sequences = preds

    def __batch_data(self, data, batch_size):
        batch_num = len(data) // batch_size + 1 if len(data) % batch_size else len(data) // batch_size
        print('data will be splitted into {} batches'.format(batch_num))
        batched_data = np.array_split(data, batch_num)

        return batched_data

    def from_video(self, video_path):
        self.__cap_video(video_path)
        self.__classify_frames()
        self.__extract_frames()
        self.__detect_text_area()
        self.__segment_lines()
        self.__extract_lines()
        self.__recognize_sequences()
        self.output_type = 'video'

    def from_image(self, img_paths):
        self.output_extract_frames = [[cv2.imread(img_path), idx] for idx, img_path in enumerate(img_paths)]
        self.__detect_text_area()
        self.__segment_lines()
        self.__extract_lines()
        self.__recognize_sequences()
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
        self.filename = 'output' + datetime.now().strftime('%Y%m%d-%H%M%S')
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
            cv2.imwrite('{}/extract_frames/{}.png'.format(self.filename, data[1]), data[0])

        # save output of detect_text_area
        for each_frame in self.output_extract_frames:
            frame_idx = each_frame[1]
            canvas = np.copy(each_frame[0])
            for each_det_area in self.output_detect_text_area:
                _, frame_idx_2, det_area = each_det_area
                if frame_idx_2 == frame_idx:
                    x, y, x_end, y_end = det_area
                    cv2.rectangle(canvas,
                                (x, y),
                                (x_end, y_end),
                                (0, 0, 255),
                                3)
            cv2.imwrite('{}/detect_text_area/{}.png'.format(self.filename,frame_idx), canvas)

        # save output of segment_lines
        for data in self.output_segment_lines:
            misc.imsave('{}/segment_lines/{}-({},{})({},{}).png'.format(self.filename,data[1], *data[2]), data[0])

        # save output of extract_lines
        for data in self.output_extract_lines:
            line_img = cv2.cvtColor(data[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)
            cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}).png'.format(self.filename, data[1], *data[2], *data[3]), line_img)

        # save output of recognize_sequences
        for data in self.output_recognize_sequences:
            f = open('{}/recognize_sequences/{}-({},{})({},{})-({},{})({},{}).txt'.format(self.filename,data[1], *data[2], *data[3]), 'w')
            f.write(data[0])

    def generate_video(self):
        print('generating output video...')
        # collect all frames, remember to padding to 2x width size
        frames = np.copy(self.frames)

        # draw detection boxes on detected frames
        idxs = []
        for each_rects in self.bounding_boxes:
            boxes, frame_idx = each_rects
            idxs.append(frame_idx)
            for box in boxes:
                x, y, x_end, y_end = box
                cv2.rectangle(frames[frame_idx], (x, y), (x_end, y_end), (255, 0, 0), 3)

        # draw min area rectangles on detected frames
        for each_rects in self.min_area_rects:
            boxes, frame_idx = each_rects
            cv2.drawContours(frames[frame_idx], boxes, -1, (0,0,255),3)

        # put output texts on the right

        # generate output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30, (self.width, self.height))
        for idx, frame in enumerate(frames):
            out.write(frame)
            if idx in idxs:
                for _ in range(29):
                    out.write(frame)
            print('{}/{}'.format(idx + 1, self.total_frame))
    
    def gui(self):
        def sort_areas(l):
            if len(l) == 1:
                return l
            
        def sort_lines(l):
            l = sorted(l, key = lambda x: x[3][1])
            return l
        self.gui_frames = []
        for idx in range(len(self.output_segment_lines)):
            img = np.copy(self.output_extract_frames[idx][0])
            origin_h, origin_w = img.shape[:2]
            mask = np.copy(self.output_segment_lines[idx][0])
            x, y, x_end, y_end = self.output_segment_lines[idx][2]
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
        predictions = self.output_recognize_sequences
        predictions = sorted(predictions, key = itemgetter(1,2,3))
        # split by frame_idx
        predictions_per_frame = [list(g) for k, g in groupby(predictions, lambda x: x[1])]

        # 同帧不同的det_area排序，先上后下，先左后右
        # 上下：比较y，左右：比较x 
        # 每帧分别排序，每个det_area分别排序，根据x, y, x_end, y_end来排序
        output_per_frame = []
        for idx, pred_each_frame in enumerate(predictions_per_frame):
            predictions_per_det_area = [list(g) for k, g in groupby(pred_each_frame, lambda x: x[2])]
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
    ext = Extractor()
    ext.from_video('test_dataset/第一批_20170911/VID_20170911_134231.mp4')
    ext.save()
    ext.gui()
    # ext.generate_video()
    # ext.from_image('test.jpg')
    # ext.save()
    for func, time in time_record.items():
        print(func, time)
