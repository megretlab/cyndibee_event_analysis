import json
import cv2
import numpy as np
import random
import numbers
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
from plotbee.utils import read_json, get_fname

from multiprocessing import Process, Queue, Lock

from plotbee.frame import Frame
from plotbee.body import Body
from plotbee.body import parse_parts
from plotbee.track import Track
from plotbee.utils import save_json, rescale_image
from plotbee.tracking import hungarian_tracking, sort_tracking, non_max_supression_video, hungarian_tracking_with_prediction
from plotbee.events import track_classification
from plotbee.tag import detect_tags_on_video, detect_tags_on_video_frame_level
from plotbee.video_wrapper import VideoCaptureWrapper
# from plotbee.tag import match_tags
from plotbee.videoplotter import extract_body
# from plotbee.pollen import process_pollen 

# from plotbee.utils import divide_video, merge_videos
from tensorflow.keras.models import model_from_json, load_model
import tensorflow as tf
import multiprocessing as mp


SIZE=(375, 450)

    
def divide_video(video, fname, N):
    frames = len(video)
    batch = frames//N
    
    fpath, ext = os.path.splitext(fname)
    
    filenames = list()
    
    for i in range(N):
        start = i * batch
        end = (i + 1) * batch
        if end > frames:
            end = frames
            
        v = video[start:end]
        
        path = fpath + "_" + str(i) + ext
        v.save(path)
        
        filenames.append(path)
    return filenames


def merge_videos(video_names):
    
    v = Video.load(video_names[0])
    
    folder, file = os.path.split(video_names[0])
    
    pfname, ext = os.path.splitext(file)
    
    pfname = "_".join(pfname.split("_")[:-1]) + ext
    
    for pname in video_names[1:]:
        vi = Video.load(pname)
        v.append(vi)

    out_filename = os.path.join(folder, pfname)
    v.save(out_filename)
    return out_filename 
    


def load_json_model(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return model_from_json(data)

def load_pollen_model(model_path):
    path, ext = os.path.splitext(model_path)

    if ext.lower() == "json":
        model = load_json_model(model_path)
    else:
        model = load_model(model_path, compile=False)
    return model


def preprocess_input(image, rescale_factor=1):

    image_height, image_width, _ = image.shape
    dim = (image_width//rescale_factor, image_height//rescale_factor)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dim)
    image = cv2.normalize(image,dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

def tfv2_pollen_classifier(video_filename, model_path, weigths_path, gpu, gpu_fraction, model_size=2048, scale=1.0):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
        
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices, gpu)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=model_size)])
    
    folder = '/'.join(video_filename.split('/')[:-1])
    
    model = load_pollen_model(model_path)
    if weigths_path != None:
        model.load_weights(weigths_path)

    video_data = Video.load(video_filename)
    # start = video_data[0].id
#     print(start)
    video = video_data.get_video_stream()
    data = list()

    Body.scale = scale
    
    Body.width=360
    Body.height=360

    Body.out_width = 90
    Body.out_height = 90
    
    for i, frame in enumerate(tqdm(video_data, desc=video_filename)):
        ret, im = video.read()
        if not ret:
            print("Something wrong with the video.")
        bodies, images = Frame._extract_bodies_images(im, frame)
        
        images = np.array(images)/255.

        try:
            score=model.predict_on_batch(images)
        except:
#             print(images.shape)
#             print(frame)
            continue
        for body, pscore in zip(bodies, score):
            body.pollen_score = float(pscore[0])

    video.release()
    
    video_data.save(video_filename)
    
    return    

def process_pollen(video, model_path, model_weights, workers=4, gpus=["1", "0"], model_size=2048, scale=1.0):
    
    tmp_folder = "pollen_temp"
    os.makedirs(tmp_folder, exist_ok=True)
    pollen_path = os.path.join(tmp_folder, "pollen_temp.json")
    
    frames = len(video)
    
    processes = dict()
    
    # Divide current video into N temp_files
    filenames = divide_video(video, pollen_path, workers)
    
    # Process each file with pollen classification
    for i, file in enumerate(filenames):
        gpu = gpus[i % len(gpus)]
        processes[file] = mp.Process(target=tfv2_pollen_classifier,args= (file, model_path, model_weights, gpu, (1*len(gpus))/workers, model_size, scale))
        processes[file].start()

    for k in processes:
        processes[k].join()
    
    # Merge files
    fname = merge_videos(filenames)
    
    return Video.load(fname)



def find_connections(point, part, mappings):
    
    skleton = list(mappings.keys())
    
    points = defaultdict(list)
    buffer = [(point, part)]
        
    while len(buffer) != 0:
        
        p, pt = buffer.pop()
        
        for limb in skleton:
            if pt in limb:
                target_part = limb[0] if limb[0] != pt else limb[1]
                indices = [i for i, x in enumerate(mappings[limb][pt]) if x == p]
                
                for indx in indices:
                    target_point = mappings[limb][target_part][indx]    
                    
                    # check if not in points
                    if target_point not in points[target_part]:
                        buffer.append((target_point, target_part))
                        points[target_part].append(target_point)
    return points


def get_mappings_by_limb(maps):
    detsByLimbs = defaultdict(lambda: defaultdict(list))
    for m in maps:
        detsByLimbs[tuple(m[5])][m[5][0]].append(tuple(m[0]))
        detsByLimbs[tuple(m[5])][m[5][1]].append(tuple(m[1]))
    return detsByLimbs

def point_in_frame(track_id, id_tracks, frame_id):
    track_info = id_tracks[str(int(track_id))]

    init_frame = track_info["init_frame"]
    track_points = track_info["positions"]
    track_position = frame_id - init_frame

    x = track_points[track_position][0]
    y = track_points[track_position][1]

    return (x, y)


def find_bodyid(body_point, tracks, id_tracks, frame_id):
    for track_id in tracks:
        if track_id == 0:
            continue
        track_point = point_in_frame(track_id, id_tracks, frame_id)
        if track_point == body_point:
            return track_id
    return -1


def create_bodies_from_mapping(tracking_limb, tracking_part, mapping, tracks, id_tracks, frame):

    tracking_points = mapping[tracking_limb][tracking_part]
    skeleton = list(mapping.keys())
    bodies = list()
    
    for point in tracking_points:
        
        body_id = -1
        if tracks is not None:
            body_id = find_bodyid(point, tracks, id_tracks, frame.id)
        

        body_parts = find_connections(point, tracking_part, mapping)
        bodies.append(Body(body_parts, tracking_part, 
                           tracking_limb, skeleton, 
                           frame, body_id=body_id))
    
    return bodies


def frames_from_detections(detections, tracks, id_tracks, tracking_limb=(1, 3), tracking_part=3, video_path=None, load_image=False):
    """
    Create Video from Ivan's detections and tracks JSON objects
    """
    
    frame_list = list()
    prev_track = defaultdict(lambda: None)
    track_dict = dict()

    dets_size = len(detections.keys())
    image = None

    if load_image:
        vid  = cv2.VideoCapture(video_path)

    for frame_id in tqdm(range(dets_size)):
        
        str_id = str(frame_id)

        if str_id not in detections:
            continue
        
        data = detections[str_id]

        frametracks = None
        if tracks is not None:
            frametracks = tracks[frame_id]
        
        mappings = get_mappings_by_limb(data["mapping"])
        parts = data["parts"]

        if load_image:
            res, image = vid.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        frameobj = Frame([], frame_id, image, mappings, parts)
        
        bodies = create_bodies_from_mapping(tracking_limb, tracking_part, mappings, frametracks, id_tracks, frameobj)

        if tracks is not None:
            for b in bodies:

                if b.id not in track_dict:
                    track_dict[b.id] = Track(b)
                
                b.prev = prev_track[b.id]

                if prev_track[b.id] is not None:
                    prev_track[b.id].next = b
                
                prev_track[b.id] = b


        frameobj.update(bodies)
        
        frame_list.append(frameobj)

    if load_image:
        vid.release()
        
    return frame_list, track_dict


def image_from_video(video_path, frameid):

    video = cv2.VideoCapture(video_path)

    video.set(cv2.CAP_PROP_POS_FRAMES, frameid)

    res, im = video.read()
        
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    video.release()

    return im


def get_video(video_path, start=0, end=np.inf, step=1):

    video = cv2.VideoCapture(video_path)

    video.set(cv2.CAP_PROP_POS_FRAMES, start)

    return video




def process_video(frames, video_path, start, end, img_folder, file_format, lock, pbar):
    with lock:
        print('Starting Consumer => {}'.format(os.getpid()))
    
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame in frames:
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame.id)
        ret, image = vid.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process_frame(image, frame, img_folder, file_format)
        with lock:
            pbar.update(1)





def process_frame(image, frame, img_folder, file_format):
    body_info = list()
    for body in frame:
        if not body.valid:
            continue
        # info = body.info()
        filename = file_format.format(body.frameid, body.id)
        body_filename = os.path.join(img_folder, filename)

        if body.tag is not None:
            pts = np.array(body.tag["p"]).astype(np.int32)
            pts = pts.reshape((-1,1,2))
            image = cv2.fillPoly(image,[pts], (0,0,0))
            image = cv2.polylines(image,[pts],True,(0,0,0),35)
        # info["filename"] = body_filename
        # body.save(body_filename, width=width, height=height, cX=cX, cY=cY)
        im = extract_body(image, body, width=Body.width, height=Body.height,
                    cX=Body.cX, cY=Body.cY)
        io.imsave(body_filename, im)

def process_frame_consumer(video_path, q, lock, pbar):
    with lock:
        print('Starting Consumer => {}'.format(os.getpid()))
    video = cv2.VideoCapture(video_path)
    while True:
        params = q.get()
        if params == "Done!":
            break
        image = video.set(cv2.CAP_PROP_POS_FRAMES, params[0].id)
        extract_body(image, *params)
        with lock:
            pbar.update(1)
    video.release()
    with lock:
        print(' Exit Consumer => {}'.format(os.getpid()))
    return


def dist(a, b):
    npa = np.array([a])
    npb = np.array([b])

    return np.sqrt(np.sum((npa - npb)**2))


def match_tags(frame, tag_list, virtual=False, th_dist=50):
    
    for tag in tag_list:
        min_dist = th_dist
        closest_body = None
        for body in frame:
            if body.tag is not None:
                continue
            d = dist(body.center, tag['c'])
            if d < min_dist:
                min_dist = d
                closest_body = body
        if closest_body is not None:
            closest_body.tag = tag
        elif virtual:
            # Add new body with the tag as thorax
            x, y = tag['c']
            body = Body({3: [(x,y)]}, center=3,
                        connections=[],angle_conn=[3,3],
                        frame=frame,tag=tag,body_id=-1)
            frame.update([body])




class Video():
    
    @classmethod
    def from_config(cls, config, load_image=False):
        """ Video Constructor from config dict (this is used to import beepose format file)

        :param config: A dictionary containing the beepose files path
            - `DETECTIONS_PATH` path to detection file
            - `VIDEO_PATH` path to video
            - `TRACK_PATH` path to tracks optional
            - `ID_TRACK_PATH` path to the ids of tracks optional
        :type config: dict
        :param load_image: if `True` will load images from the video, defaults to False
        :type load_image: bool, optional
        :return: Video object
        :rtype: Video

        .. warning:: `load_image = True` is not recommended for large videos.
            This option will result in slower performance and high memory usage.

        .. warning:: This is old, generally `Video.load` is used.
        """
        try:
            detection_path = config['DETECTIONS_PATH']
            video_path = config['VIDEO_PATH']
        except KeyError as e:
            raise Exception('You should provide an {} with the config.'.format(e))
        else:

            tracks_json =  None
            id_tracks_json = None

            if 'TRACK_PATH' in config:
                track_path = config['TRACK_PATH']

                try:
                    id_track_path = config['ID_TRACK_PATH']
                except KeyError as e:
                    raise Exception('You should provide an {} with the config.'.format(e))

                tracks_json = read_json(track_path)
                id_tracks_json = read_json(id_track_path)

            dets = read_json(detection_path)
            
            frames, tracks = frames_from_detections(dets, tracks_json, id_tracks_json,
                                                    video_path=video_path, load_image=load_image)

        return cls(frames, tracks, config)


    @classmethod
    def load(cls, json_path):
        """JSON Constructor

        :param json_path: plotbee file path. Normally, with suffix "_skeleton.json" 
        :type json_path: str
        :return: Video Object
        :rtype: Video
        """
        data = read_json(json_path)

        config = data["config"]

        start = 0
        stop = None
        step = 1
        # to support old versions
        if "start" in data:
            start = data["start"]
        if "stop" in data:
            stop = data["stop"]
        if "step" in data:
            step = data["step"]

        frames = list()
        track_dict = dict()
        prev_track = defaultdict(lambda: None)

        for frame in tqdm(data["frames"]):
            bodies = list()
            frameobj = Frame([], frame["id"], parts=frame["parts"])

            for body in frame["bodies"]:
                bodies.append(Body.load_body(body, frameobj))
                # Compatible with older versions
                # if "tag" not in body:
                #     body["tag"] = None
                # parsed_parts = parse_parts(body["parts"]) 
                # bodies.append(Body(parsed_parts, body["center_part"],
                #                   tuple(body["angle_conn"]), body["connections"],
                #                   frameobj, body["id"], body["suppressed"], body["pollen_score"], body["tag"]))

            for b in bodies:
                if b.id == -1:
                    continue

                if b.id not in track_dict:
                    track_dict[b.id] = Track(b)
                
                b.prev = prev_track[b.id]

                if prev_track[b.id] is not None:
                    prev_track[b.id].next = b
                    track_dict[b.id].endframe = b.frameid
                    track_dict[b.id]._data[b.frameid] = b


                
                prev_track[b.id] = b

            frameobj.update(bodies)
            frames.append(frameobj)
            
        # for i, tr in track_dict.items():
        #     tr.init()

        video = cls(frames, track_dict, config, start=start, stop=stop, step=step)
        video.track_clasification()
        return video

    @classmethod
    def from_detections(cls, detections, video_path=None, load_images=False):
        """Beepose dictionary constructor

        :param detections: Beepose dictionary with detections
        :type detections: dict
        :param video_path: video path, defaults to None
        :type video_path: str, optional
        :param load_images: if `True` images will load at loading time, defaults to False
        :type load_images: bool, optional
        :return: Video Object
        :rtype: Video

        .. note:: This function is mostly internal. This function runs on beepose implementation to generate plotbee output.
        """
        frames, tracks = frames_from_detections(detections, None, None,
                                                video_path=video_path, load_image=load_images)
        return cls(frames, tracks, dict())

                 
    
    def __init__(self, frames, tracks, config, start=0, stop=None, step=1):
        """Default Constructor

        :param frames: Frame object that belong to the video
        :type frames: list[Frame]
        :param tracks: Track of the video
        :type tracks: dict[Track]
        :param config: config dictionary with path to the video
        :type config: dict
        :param start: starting frame id, defaults to 0
        :type start: int, optional
        :param stop: end frame id, defaults to None
        :type stop: int, optional
        :param step: step for the video, defaults to 1
        :type step: int, optional
        """
        self._frames = frames
        self._tracks = tracks
        self._start = start
        self._stop = stop
        if not stop: 
            self._stop = len(self._frames)
        self._step = step

        for frame in self._frames:
            frame.set_video(self)

        self._config = config
        
        
    @property
    def config(self):
        """Get video config file

        :return: config file
        :rtype: dict
        """
        return self._config
    
    @property
    def frames(self):
        """Get a list of frames

        :return: list of video frames
        :rtype: list[Frame]
        """
        return self._frames

    @property
    def tracks(self):
        """Get tracks

        :return: video tracks
        :rtype: dict
        """
        return self._tracks

    @property
    def video_path(self):
        """Get video path

        :return: video path
        :rtype: str
        """
        return self.config['VIDEO_PATH']

    @property
    def video_name(self):
        """Get video name

        :return: video name
        :rtype: str
        """
        folder, video_name = os.path.split(self.config['VIDEO_PATH'])
        return video_name

    def frame_image(self, index):
        """Get frame image at `index` position

        :param index: index of the request frame image
        :type index: int
        :return: return image of the requested frame
        :rtype: numpy.ndarray
        """
        return image_from_video(self.video_path, index)

    
    def _get_frame(self, frame_id):
        """Get frame by frame id

        :param frame_id: frame id
        :type frame_id: int
        :return: Frame object
        :rtype: Frame
        """
        return self._frames[frame_id]
    
    
    def get_video_stream(self, start=None, stop=None, step=None):
        """Get video stream

        :param start: frame to start, defaults to None
        :type start: int, optional
        :param stop: frame to stop, defaults to None
        :type stop: int, optional
        :param step: frame step, defaults to None
        :type step: int, optional
        :return: An "efficient" VideoCapture Wrapper
        :rtype: VideoCaptureWrapper
        """

        if step is None:
            step = self._step

        if start is None:
            start =  self._start

        if stop is None:
            stop = self._stop

        return VideoCaptureWrapper(self.video_path, start=start, stop=stop, step=step)


    def __repr__(self):
        """Print video info

        :return: Show video info such as video filename, length and their detection
        :rtype: str
        """
        detection_fname = None


        video_fname = get_fname(self._config['VIDEO_PATH'])
        if 'DETECTIONS_PATH' in self._config:
            detection_fname = get_fname(self._config['DETECTIONS_PATH'])
        return "Video(name={}, detections={}, len={})".format(video_fname,
                                                              detection_fname, len(self))

    
    def __len__(self):
        """Get the number of frames

        :return: number of frames in the video
        :rtype: int
        """
        return len(self._frames)


    def append(self, video):
        """Concatenate a video at the end

        :param video: Video to concatenate
        :type video: Video
        """
        if self.config == video.config:
            self._frames += video._frames


    def __getitem__(self, index):
        """Get frame by index

        :param index: index of the requested frame
        :type index: int
        :return: the requested frame
        :rtype: Frame
        """
        cls = type(self)
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            if index.step is None:
                step = self._step
            if index.start is None:
                start = self._start
            if index.stop is None:
                stop = self._stop
            
            return cls(self._get_frame(index), self._tracks, self.config, start=start, stop=stop, step=step)
        elif isinstance(index, numbers.Integral):
            return self._get_frame(index)
        else:
            msg = '{.__name__} indices must be integers'
            raise TypeError(msg.format(cls))

    
    def export(self, folder, skeleton=True, bbox=True, tracks=True, events=False, min_parts=-1, max_workers=5, idtext=False, fontScale=2.5, fontThickness=8):
        """Export video images with plotting options on folder.

        :param folder: folder to save the frame images
        :type folder: str
        :param skeleton: if `True` images will show the predicted skeleton, defaults to True
        :type skeleton: bool, optional
        :param bbox: if `True` images will show the bounding boxes of the predicted bodies, defaults to True
        :type bbox: bool, optional
        :param tracks: if `True` images will show the tracks of the predicted bodies, defaults to True
        :type tracks: bool, optional
        :param events: if `True` draws frame's detections events on the frame_image, defaults to False
        :type events: bool, optional
        :param min_parts: this functions only draws bodies that `len(body) >= min_parts`, defaults to -1
        :type min_parts: int, optional
        :param idtext: if `True` shows track id on the image, defaults to False
        :type idtext: bool, optional
        :param fontScale: idtext font scale, defaults to 1.5
        :type fontScale: float, optional
        :param fontThickness: idtext fontsize, defaults to 3
        :type fontThickness: int, optional
        :param thickness: bounding box line thickness, defaults to 7
        :type thickness: int, optional:param events: , defaults to False
        :param max_workers: workers to process this task in parallel, defaults to 5
        :type max_workers: int, optional
        """
        os.makedirs(folder, exist_ok=True)
        # Parallel Implementaion
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_frames = list()
            for frame in self.frames:
                future = executor.submit(frame.save, folder, skeleton, bbox,
                                        tracks, events, min_parts,  
                                        idtext=idtext, fontScale=fontScale,
                                        fontThickness=fontThickness)
                future_frames.append(future)
                
            
            done_iter = futures.as_completed(future_frames)
            done_iter = tqdm(done_iter, total=len(self.frames))
            
            for future in done_iter:
                _ = 1+1
        return

    def clear_ids(self):
        """Clear track ids of all bodies in the video.
        """
        for frame in self:
            for body in frame:
                body.set_id(-1)
                body.prev = None
                body.next = None

    def non_max_supression(self, nms_overlap_fraction=0.6):
        """Non maximum supression will supress all the bodies detection
        that hava an overlap greater or equal to `nms_overlap_fraction` to remove potential detection duplicates.

        :param nms_overlap_fraction: amount of overlaping fraction to supress a detection, defaults to 0.6
        :type nms_overlap_fraction: float, optional
        """
        non_max_supression_video(self, nms_overlap_fraction)
    
    def hungarian_tracking(self, predictive=True, **kwargs):
        """Hungarian Tracking

        :param predictive: if `True` in case that a body was misdetected at frame `t+1` 
            then this will be filled with a virtual detection in order to continue with the track, defaults to True
        :type predictive: bool, optional
        """
        if predictive:
            hungarian_tracking_with_prediction(self, **kwargs)
        else:
            hungarian_tracking(self, **kwargs)

    def sort_tracking(self, bbox=200, nms_overlap_fraction=0.6):
        """SORT tracking

        :param bbox: the size of the tracking box, defaults to 200
        :type bbox: int, optional
        :param nms_overlap_fraction: amount of overlaping fraction to supress a detection, defaults to 0.6
        :type nms_overlap_fraction: float, optional
        """
        sort_tracking(self, bbox, nms_overlap_fraction)
        
    def track_clasification(self, **kwargs):
        """Track Classification
            This will classify all the tracks in the video.
        """
        track_classification(self, **kwargs)

    def tag_detection(self, max_workers=5, frame_level=False):
        """Tag detection of the video

        :param max_workers: amount of worker to perform the task in parallel, defaults to 5
        :type max_workers: int, optional
        :param frame_level: if `True` the tag will be detected at frame level and then match the close body to the detected tag.
            if `False` the tag detection will be perform to each body image on the video, defaults to False
        :type frame_level: bool, optional
        """
        if frame_level:
            detect_tags_on_video_frame_level(self, max_workers=max_workers)
        else:    
            detect_tags_on_video(self, max_workers=max_workers)

    def events_counter(self):
        """Count Events

        :return: frequecy per event
        :rtype: dict
        """
        event_counter = defaultdict(int)
        for track in self.tracks.values():
            if track.event is not None:
                event_counter["event." + track.event] += 1
            if track.track_shape is not None:
                event_counter["trackshape." + track.track_shape] += 1
            if track.pollen:
                event_counter["pollen"] += 1
        event_counter["tracks"] = len(self.tracks)
        return event_counter


    # def export_bodies(self, folder, width=None, height=None, cX=None, cY=None, workers=5):
    #     """Parallel Implementation of bodies Export"""
        
    #     file_format = "{:09d}_{:09d}.jpg"
        
    #     os.makedirs(folder, exist_ok=True)
    #     img_folder = os.path.join(folder, "images")
    #     cvs_filename = os.path.join(folder, "dataset.csv")
    #     json_filename = os.path.join(folder, "dataset.json")
    #     os.makedirs(img_folder, exist_ok=True)

    #     body_info = []
        
    #     with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         future_frames = list()
    #         for frame in self.frames:
    #             future = executor.submit(process_frame, image, frame, img_folder, file_format,
    #                     width, height, cX, cY)
    #             future_frames.append(future)
                
            
    #         done_iter = futures.as_completed(future_frames)
    #         done_iter = tqdm(done_iter, total=len(self.frames))
            

    #         for future in done_iter:
    #             body_info += future.result()
            
    #     df = pd.DataFrame(body_info)
    #     df.to_csv(cvs_filename, index=False)
    #     self.save(json_filename)
    #     return

    def export_bodies(self, folder, width=None, height=None, cX=None, cY=None, workers=5):
        """Export bodies and body images into a JSON format and image folder

        :param folder: folder to save the exported json file and images
        :type folder: str
        :param width: width of the patch rectagle,, defaults to 160
        :type width: int, optional
        :param height: height of the patch rectagle, defaults to 320
        :type height: int, optional
        :param cX: `x` of the center point in the output rectangle, defaults to None
        :type cX: int, optional
        :param cY: `y` of the center point in the output rectangle, defaults to None
        :type cY: int, optional
        :param workers: amount of workers to process this task in parallel, defaults to 5
        :type workers: int, optional
        """
        
        file_format = "{:09d}_{:09d}.jpg"
        video_name, ext = os.path.splitext(self.video_name)
        
        os.makedirs(folder, exist_ok=True)
        img_folder = os.path.join(folder, video_name)
        # cvs_filename = os.path.join(folder, "dataset.csv")
        json_filename = os.path.join(folder, video_name + ".json")
        os.makedirs(img_folder, exist_ok=True)

        body_info = []

        video_size = len(self)
        chunksize = video_size//workers
        lock = Lock()
        
        ws = list()

        pbar = list()


        for w in range(workers - 1):
            start = w * chunksize
            end = (w + 1) * chunksize
            pbar.append(tqdm(total=chunksize))

            p = Process(target=process_video, args=(self[start:end], self.video_path, start, end, img_folder, file_format, lock, pbar[w]))
            p.deamon = True
            ws.append(p)
            p.start()

        start = (workers - 1)* chunksize
        end = len(self)
        pbar.append(tqdm(total=end-start))
        p = Process(target=process_video, args=(self[start:end], self.video_path, start, end, img_folder, file_format, lock, pbar[workers -1]))
        p.deamon = True
        ws.append(p)
        p.start()

        for w in ws:
            w.join()
        self.save(json_filename)
        return
    
    def json(self):
        """Export Video to JSON format

        :return: Video as JSON format
        :rtype: dict
        """
        video_json = dict()
        video_json["config"] = self._config
        video_json["start"] = self._start
        video_json["stop"] = self._stop
        video_json["step"] = self._step

        video_json["frames"] = list()

        for frame in tqdm(self._frames):
            
            frame_info = dict()
            frame_info["id"] = frame.id
            frame_info["parts"] = frame._parts

            frame_bodies = list()

            for body in frame:
                frame_bodies.append(body.params())

            frame_info["bodies"] = frame_bodies

            video_json["frames"].append(frame_info)
        return video_json
        

    def save(self, path):
        """Save video as JSON

        :param path: path to save the video data
        :type path: str
        """
        video_json = self.json()
        save_json(path, video_json)

    def clear_tags(self):
        """Clear tags for all body in the video.
        """
        for frame in self:
            for body in frame:
                body.tag = None

    def load_tags(self, tags_file, virtual=False):
        """Load tags from Tag multi-file format

        :param tags_file: path to the tag file to load
        :type tags_file: str
        :param virtual: if `True` will attach tags to virtual body, defaults to False
        :type virtual: bool, optional
        """
        tags = read_json(tags_file)
        self.clear_tags()

        for frame in self:
            sid = str(frame.id)
            if sid in tags["data"]:
                tagged_bees = tags["data"][sid]['tags']

                match_tags(frame, tagged_bees, virtual=virtual, th_dist=50)
        return

    def load_video(self, video_file):
        """Set the video path to load images

        :param video_file: path to the video file
        :type video_file: str

        .. note:: `load_video` is critical for all plotting. 
            To load the frame image it is important that Video knows the location to load the frame image.
        """
        if os.path.exists(video_file):
            self.config['VIDEO_PATH'] = video_file
        else:
            raise ValueError

    def tagged(self):
        """Get all tagged body.

        :return: list of body with tags detected
        :rtype: list[Body]
        """
        tagged = list()
        for frame in self:
            for body in frame:
                if body.tag is not None:
                    tagged.append(body)
        return tagged

    def get_frame_with_untracked_body(self):
        """Get the first frame with an untracked body

        :return: First frame that at least has one detection without a tracking id
        :rtype: Frame

        .. note:: This was implemented to help to annotate bodies for a track evaluation.
        """
        for frame in self:
            for body in frame:
                if body.id == -1:
                    return frame


    def export_tagged(self, output_folder, save_images=True):
        """Export all tagged bodies in a JSON file and optionally all the body images.

        :param output_folder: folder to save the images and json file
        :type output_folder: str
        :param save_images: if `True` will export tagged bodies images, defaults to True
        :type save_images: bool, optional

        .. tip:: This function will serve to create a dataset of tagged bees.
        """
        tag_bodies = self.tagged()
        _, video_name = os.path.split(self.video_path)
        video_name, ext = os.path.splitext(video_name)
        out_path = os.path.join(output_folder, video_name)
        os.makedirs(out_path, exist_ok=True)
        
        json_path = os.path.join(output_folder, "{}.json".format(video_name))
        tagged_json= list()

        for body in tqdm(tag_bodies):
            tagged_json.append(body.params())

            if save_images:
                fname = "TID{:05}_F{:08}.jpg".format(body.tag_id, body.frameid)
                fpath = os.path.join(out_path, fname)
                body.save(fpath)
                # print("body saved.")
            

        save_json(json_path, tagged_json)
        
        return

    def export_pollen(self, output_folder, limit=None, sorted_scores=False):
        """Export pollen detection dataset

        :param output_folder: folder to save the dataset
        :type output_folder: str
        :param limit: set a limit of detection to export, defaults to None
        :type limit: int, optional
        :param sorted_scores: if `True` the pollen score will be sorted and take the the top `limit//2` and the bottom `limit//2` to export, defaults to False
        :type sorted_scores: bool, optional

        .. note:: This function was design to provide aditional training data
            for pollen classification in active learning fashion.
        """
        bodies = [body for frame in self for body in frame]
        bodies = sorted(bodies, key=(lambda b: b.pollen_score))
        

        def valid(body):
            x, y = body.center
            if (x > 500 and  x < 2100) and (y > 500 and  y < 800):
                return True
            return False

        bodies = [body for body in bodies if valid(body)]

        if limit and sorted_scores:
            bodies = bodies[:limit//2] + bodies[-limit//2:]
        elif limit:
            pollen_scores = [body.pollen_score for body in bodies]
            threshold_index = bisect.bisect(pollen_scores, 0.5)
            no_pollen_bodies = bodies[:threshold_index]
            pollen_bodies = bodies[threshold_index:]

            random.shuffle(no_pollen_bodies)
            random.shuffle(pollen_bodies)

            bodies = no_pollen_bodies[:limit//2] + pollen_bodies[:limit//2]

        _, video_name = os.path.split(self.video_path)
        video_name, ext = os.path.splitext(video_name)

        print(video_name)
        
        out_path = os.path.join(output_folder, video_name)
        os.makedirs(out_path, exist_ok=True)
        
        pollen_path = os.path.join(out_path, "P")
        os.makedirs(pollen_path, exist_ok=True)
        
        nopollen_path = os.path.join(out_path, "NP")
        os.makedirs(nopollen_path, exist_ok=True)

        pollen_csv = os.path.join(out_path, "pollen.csv")

        fnames = list()
        pollen_scores = list()
        xs = list()
        ys = list()

        for body in tqdm(bodies):
            x, y = body.center
            pollen_score = body.pollen_score
            fname = "{:09}_X{:04}_Y{:04}.jpg".format(body.frameid, x, y)
            if pollen_score < 0.5:
                fpath = os.path.join(nopollen_path, fname)
            else:
                fpath = os.path.join(pollen_path, fname)
            
            fnames.append(fpath)
            pollen_scores.append(pollen_score)
            xs.append(x)
            ys.append(y)
            body.save(fpath)
            
        df_dict = {
            "filename":fnames,
            "pollen": pollen_scores,
            "x":xs,
            "y":ys
        }
        df = pd.DataFrame(df_dict)
        df.to_csv(pollen_csv, index=False)
        
        return
    
    def process_pollen(self,  model_path, weights=None, workers=4, gpus=["1", "0"], model_size=2048, scale=1.0):
        """Process polle

        :param model_path: path to the pollen detection model
        :type model_path: str
        :param weights: path to the weights of the model, defaults to None
        :type weights: str, optional
        :param workers: amount of workers to process this task in parallel, defaults to 4
        :type workers: int, optional
        :param gpus: which GPU this task will use, defaults to ["1", "0"]
        :type gpus: list, optional
        :param model_size: size of the model, defaults to 2048
        :type model_size: int, optional
        :param scale: upsampling (resize) scale of the body image (scale > 1.0 => zoom), defaults to 1.0
        :type scale: float, optional
        """
        pollen_video = process_pollen(self, model_path, weights, workers=workers, gpus=gpus, model_size=model_size, scale=scale)
        
        self._frames = pollen_video._frames
        self._tracks = pollen_video._tracks
        for frame in self._frames:
            frame.set_video(self)
            
        self._config = pollen_video._config
        return






