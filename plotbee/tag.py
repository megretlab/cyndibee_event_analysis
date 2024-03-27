import cv2
from tqdm import tqdm
import numpy as np

try:
    import apriltag.apriltagdetect as atd
    from apriltag.apriltagdetect import detectionsToObj
    import apriltag.apriltag as apriltag
except ImportError:
    _has_apriltag = False
else:
    _has_apriltag = True


def requires_apriltag(func):
    def wrapper():
        if not _has_apriltag:
            raise ImportError("apriltag is required to do this. plotbee with [tags] should be installed.\n\n pip install plotbee[tags]\n")
        func()
    return wrapper


def dist(a, b):
    npa = np.array([a])
    npb = np.array([b])

    return np.sqrt(np.sum((npa - npb)**2))

@requires_apriltag
def tagDetector():

    options = atd.presets('tag25h5inv')
    #options.quad_contours = False
    options.nthreads=40
    options.debug=0
    options.multiframefile_version = 3

    det = apriltag.Detector(options)

    det.tag_detector.contents.qcp.contour_margin = 50
    det.tag_detector.contents.qcp.min_side_length = 15

    return det


def filter_corners(detections):
    filtered = [det for det in detections if det.tag_id != 15 and det.tag_id != 16]
    return filtered


def detect_tags(det, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detections = det.detect(gray)
    filtered = filter_corners(detections)
    return filtered


# This function is used for tag detection
def match_tag2body(frame, tag, min_dist=50):
    closest_body = None
    min_distance = float("Inf")

    for body in frame:
        if body.tag is not None:
            continue
        if dist(tag.center, body.center) < min_distance:
            closest_body = body
    
    closest_body.tag = tag

    return

# This functions is used for  merge tag file into skeleton file
# def match_tags(frame, tag_list, th_dist=50):
    
#     for tag in tag_list:
#         min_dist = th_dist
#         closest_body = None
#         for body in frame:
#             if body.tag is not None:
#                 continue
#             d = dist(body.center, tag['c'])
#             if d < min_dist:
#                 min_dist = d
#                 closest_body = body
#         if closest_body is not None:
#             closest_body.tag = tag
#         else:
#             # Add new body with the tag as thorax
#             x, y = tag['c']
#             body = Body({3: [(x,y)]}, center=3,
#                         connections=[],angle_conn=[3,3],
#                         frame=frame,tag=tag,body_id=-1)
#             frame.append(body)


def detect_tags_on_frame(det, frame):
    fimage = frame.image
    tags = detect_tags(det, fimage)
    for tag in tags:
        match_tag2body(frame, tag)
    return

def detect_tags_on_video_frame_level(video, max_workers=5):
    det = tagDetector()
    for frame in tqdm(video):
        detect_tags_on_frame(det, frame)



def detect_tags_on_video(video, max_workers=5):
    det = tagDetector()

    # with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     future_frames = list()
    for frame in tqdm(video):
        bodies, images = frame.bodies_images()
        for body, bimage in zip(bodies, images):
            tags = detect_tags(det, bimage)
            tags = detectionsToObj(tags)
            if len(tags) == 1:
                body.tag = tags[0]

@requires_apriltag
def get_tag_image(body):
    tag_position = body.tag['p']
    image = body._frame.image
    tag_image = atd.extract_tag_image(tag_position, image, pixsize=3)
    return tag_image

