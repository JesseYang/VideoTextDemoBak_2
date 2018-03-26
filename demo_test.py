# import cv2
# from scipy import misc
# from easydict import EasyDict as edict
# from tensorpack import *
import numpy as np

# list split utils
# from operator import itemgetter
# from itertools import *

# import time

# import pdb


if __name__ == '__main__':
    # ext = Extractor()
    # ext.from_video('test_dataset/第一批_20170911/VID_20170911_134231.mp4')
    # ext.save()
    # ext.gui()
    # # ext.generate_video()
    # # ext.from_image('test.jpg')
    # # ext.save()
    # for func, time in time_record.items():
    #     print(func, time)

    # a=[0,1,2,3,4,5,6,7,8,9]
    # print(a)
    # c=[0,1,2,4,5]
    # for i_idx, i in enumerate(c):
       
    #     if i_idx >=1:
    #         a.pop(i-i_idx)
    #     else:
    #         a.pop(i)
       
    # print(a)

    a=[0,1,3,5,6]
    b=[0,1,2,3,4,5,6]
    print(list(set([i for i in range(8)]).difference(set(a))))
    # print(list(set(b).union(set(a))))
    # print(list(set(a).union(set(b))))