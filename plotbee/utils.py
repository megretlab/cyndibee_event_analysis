import json
import cv2
import numpy as np
import random
import os
from skimage import io
from tqdm import tqdm
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import bisect
from skimage import io
import pandas as pd
from concurrent import futures


YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)


COLOR_BY_CONNECTION = {
    (1, 3) : BLUE, 
    (3, 2) : RED,
    (2, 4) : YELLOW,
    (2, 5) : YELLOW,
    (1, 2) : MAGENTA
}

COLOR_BY_PART = {
    '0' : BLUE,    #TAIL
    '1' : RED,     #HEAD
    '2' : MAGENTA, #ABDOMEN
    '3' : YELLOW,  #ANTENA
    '4' : YELLOW   #ANTENA
}

def read_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def hash1(i):
    return ((88*i+78) % 2029) % 256

def hash2(i):
    return ((90*i+9) % 2683) % 256

def hash3(i):
    return ((99*i+100) % 2719) % 256


def id2color(i):
    if i == -1:
        return (0,0,0)
    #return hash1(i), hash2(i), hash3(i)
    return hash1(i)//2+128, hash2(i)//2+128, hash3(i)//2+128


def angleBetweenPoints(p1, p2):
    myradians = math.atan2(p1[0]-p2[0],p1[1]-p2[1])
    mydegrees = math.degrees(myradians)
    return (mydegrees)%360


def rotate_around_point(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return int(qx), int(qy)


def bound_box_points(p):
    W = 100
    HU = 200
    HL = 200
    x, y, *_ = p
    *_, angle = p
    if angle == -1:
        angle = 0
    angle = math.radians(angle - 90)
    x, y = int(x), int(y)
    p1 = x - W, y - HU
    p2 = x + W, y - HU
    p3 = x + W, y + HL
    p4 = x - W, y + HL
    
    p1 = rotate_around_point(p1, angle, (x,y))
    p2 = rotate_around_point(p2, angle, (x,y))
    p3 = rotate_around_point(p3, angle, (x,y))
    p4 = rotate_around_point(p4, angle, (x,y))

    return p1, p2, p3, p4


def rotatedBoundBoxPoints(p, angle, width, height):
    W = width//2
    H = height//2

    x, y = p
    if angle == -1:
        angle = 0
    angle = math.radians(angle)
    x, y = int(x), int(y)
    p1 = x - W, y - H
    p2 = x + W, y - H
    p3 = x + W, y + H
    p4 = x - W, y + H
    
    p1 = rotate_around_point(p1, angle, (x,y))
    p2 = rotate_around_point(p2, angle, (x,y))
    p3 = rotate_around_point(p3, angle, (x,y))
    p4 = rotate_around_point(p4, angle, (x,y))

    return p1, p2, p3, p4

def pointInRotatedBbox(p, center, angle, width, height):

    angle = math.radians(angle)
    rotated_p = rotate_around_point(p, -angle, center)

    W = width//2
    H = height//2
    

    x, y = center
    rx, ry = rotated_p

    if rx >= x - W and rx <= x + W and ry >= y - H and ry <= y + H:
        return True
    return False 



def plot_bounding_box(frame, p, color):
    *_, angle = p
    if angle == -1:
        #color = (0,0,0)
        return frame
    p1, p2, p3, p4 = bound_box_points(p)
    frame = cv2.line(frame, p1, p2, color=color, thickness=7)
    frame = cv2.line(frame, p2, p3, color=color, thickness=7)
    frame = cv2.line(frame, p3, p4, color=color, thickness=7)
    frame = cv2.line(frame, p4, p1, color=color, thickness=7)
    
    return frame

def getRotationMatrix(image_size,x,y,angle, w,h, cX=None, cY=None, scale=1.0):
    # grab the dimensions of the image and then determine the
    # center
    (h0, w0) = image_size
    (pX, pY) = (x, y) # Rect center in input
    
    if cX is None:
        cX = w / 2     # Rect center in output
        
    if cY is None:
        cY = h / 2     # Rect center in output
    
    
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale) # angle in degrees
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
  # adjust the rotation matrix to take into account translation
    M[0, 2] += pX - cX
    M[1, 2] += pY - cY

    return M


def rotate_bound2(image,x,y,angle, w,h, cX=None, cY=None, scale=1.0):
    image_size = image.shape[:2]
    M = getRotationMatrix(image_size,x,y,angle, w,h, cX, cY, scale)
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)


def get_fname(path):
    return path.split('/')[-1]


def find_connections(body, detsByLimbs):
    updated = True 
    while updated:

        updated = False
        
        for part, points in body.get_parts():
            paths = get_connections_with(part)

            
            for limb, target in paths:
                for p in points:
                    if p in detsByLimbs[limb][part]:
                        indices = [i for i, x in enumerate(detsByLimbs[limb][part]) if x == p]
                        
                        for indx in indices:
                            target_point = detsByLimbs[limb][target][indx]
                            if (target, target_point) not in body:
                                body.update(target, target_point)
                                updated = True
                    else:
                        continue


def trackevent2color(track):
    if track.pollen and track.event == 'entering':
        return YELLOW
    elif track.event == 'entering':
        return GREEN
    elif track.event == 'leaving':
        return RED
    else:
        return None

def rescale_image(image, rescale_factor=4):
    image_height, image_width, _ = image.shape
    
    dim = (image_width//rescale_factor, image_height//rescale_factor)
    
    image = cv2.resize(image, dim)
    return image
    
    
# def divide_video(video, fname, N):
#     frames = len(video)
#     batch = frames//N
    
#     fpath, ext = os.path.splitext(fname)
    
#     filenames = list()
    
#     for i in range(N):
#         start = i * batch
#         end = (i + 1) * batch
#         if end > frames:
#             end = frames
            
#         v = video[start:end]
        
#         path = fpath + "_" + str(i) + ext
#         v.save(path)
        
#         filenames.append(path)
#     return filenames


# def merge_videos(video_names):
    
#     v = Video.load(video_names[0])
    
#     folder, file = os.path.split(video_names[0])
    
#     pfname, ext = os.path.splitext(file)
    
#     pfname = "_".join(pfname.split("_")[:-1]) + ext
    
#     for pname in pollen_names[1:]:
#         vi = Video.load(pname)
#         v.append(vi)

#     out_filename = os.path.join(folder, pfname)
#     v.save(out_filename)
#     return out_filename 
    
