from plotbee.videoplotter import bodies_bbox_drawer, bodies_skeleton_drawer, bodies_track_drawer
from plotbee.videoplotter import parts_drawer, bodies_event_track_drawer, bbox_drawer
from plotbee.videoplotter import extract_body, skeleton_drawer, event_track_drawer, track_drawer
from plotbee.utils import rotate_bound2
from plotbee.utils import pointInRotatedBbox
import os
from skimage import io
from functools import lru_cache
from plotbee.body import Body
import numpy as np
import warnings
import cv2


class Frame():
    """Frame object


    :param bodies: list of beepose detection
    :type bodies: list[Body]
    :param frame_id: frame id of the video
    :type frame_id: int
    :param image: frame image., defaults to None
    :type image: np.array, optional
    :param mapping: Beepose connection or limbs predictions, defaults to None
    :type mapping: dict, optional
    :param parts: Beepose keypoint prediction, defaults to None
    :type parts: dict, optional
    :return: return a Frame object
    :rtype: Frame
    """
    
    
    
    TRACK_DIRECTION = "forward" 
    """Set a default TRACK_DIRECTION for all Frame's instances, default to "forward"

    .. note:: This is a class attribute.
    """

    def __init__(self, bodies, frame_id, image=None, mapping=None, parts=None):
        """constructor method

        :param bodies: list of beepose detection
        :type bodies: list[Body]
        :param frame_id: frame id of the video
        :type frame_id: int
        :param image: frame image., defaults to None
        :type image: np.array, optional
        :param mapping: Beepose connection or limbs predictions, defaults to None
        :type mapping: dict, optional
        :param parts: Beepose keypoint prediction, defaults to None
        :type parts: dict, optional
        """
        self._id = int(frame_id)
        # self._frame = frame
        # self._tracks = tracks
        self._video = None
        # self._parts = parts
        self._bodies = bodies

        self._cached_image = image
        # self._mappings = get_mappings_by_limb(mappings)
        # self._bodies =  mapping_to_body(self, self._mappings,
        #                                 self._tracks, id_tracks, self._id)

        self._mapping = mapping
        self._parts = parts
        
    def set_video(self, video):
        """set Frame's parent Video object 

        :param video: Video object in which this frame object belong
        :type video: Video

        .. note:: This was implemented to have a two-way link from video to frame.
        """
        self._video = video

    def get_track(self, body):
        """Get track from a body in the same video.

        :param body: The body to find the tracks.
        :type body: Body
        :return: returns the body's track.
        :rtype: Track
        """
        bid = body.id
        track = self._video.tracks[bid]
        return track

    @property
    def id(self):
        """Get frame id

        :return: frame id
        :rtype: int
        """
        return self._id


#     @property
#     def parts_image(self):
#         for body in self._bodies:
#             for part, points in body._parts.items():
#                 color = self.COLOR_BY_PART[part]
#                 for point in points:
#                     p = tuple(point[:2])
#                     frame = cv2.circle(frame, p, radius, color, thickness)
#         return frame

    # @property
    # def height(self):
    #     return self._frame.shape[0]
    
    # @property
    # def width(self):
    #     return self._frame.shape[1]

    # @property
    # def shape(self):
    #     return self._frame.shape

    # @property
    # def frame(self):
    #     return self._frame

    @property
    def bodies(self):
        """Get all bodies in the frame.

        :return: all bodies in the frame
        :rtype: list[Body]
        """
        return self._bodies

    @property
    def valid_bodies(self):
        """Get valid bodies

        :return: Valid bodies in the frame.
        :rtype: list[Body]

        .. note:: A valid body is a body that is not supressed (:func:`~plotbee.body.Body.suppressed`). Generally, this is used internally for tracking.
        """
        valid = []
        for body in self._bodies:
            if not body.suppressed:
                valid.append(body)
        return valid
    
    def delete_virtual_bodies(self):
        """Remove all virtual bodies in the frame
        """
        self._bodies = [b for b in self._bodies if (not b.virtual)]

    def update(self, bodies):
        """Add new bodies to the frame 

        :param bodies: new bodies to add
        :type bodies: list[Body]
        """
        self._bodies += bodies


    # @property
    # def parts(self):
    #     return self._parts

    @property
    def video_name(self):
        """get video filename

        :return: video_name from the parent video
        :rtype: str
        """
        return self._video.video_name

    def _image(self, **kwargs):
        """Get Frame's image with :func:`draw_frame_image` options

        :return: Frame image with options ()
        :rtype: numpy.ndarray
        """
        frame_image = self.image.copy()
        frame_image = self.draw_frame_image(frame_image, **kwargs)
        return frame_image

    def draw_frame_image(self, frame_image, skeleton=False, bbox=False, tracks=False, events=False, min_parts=-1, track_direction="forward", idtext=False, fontScale=1.5, fontThickness=3, thickness=7):
        """draws options in frame image

        :param frame_image: Frame image
        :type frame_image: numpy.ndarray
        :param skeleton: if `True` draws frame's skeletons on the frame_image, defaults to False
        :type skeleton: bool, optional
        :param bbox: if `True` draws frame's detections bounding boxes on the frame_image, defaults to False
        :type bbox: bool, optional
        :param tracks: if `True` draws frame's detections tracks on the frame_image, defaults to False
        :type tracks: bool, optional
        :param events: if `True` draws frame's detections events on the frame_image, defaults to False (see note below)
        :type events: bool, optional
        :param min_parts: this functions only draws bodies that `len(body) >= min_parts`, defaults to -1
        :type min_parts: int, optional
        :param track_direction: track direction can be "full" show the whole path of the track,
            "backward" only show tracks from the current and the previous frames or
            "forward" only show tracks from the current and the next frames, defaults to "forward"
        :type track_direction: str, optional
        :param idtext: if `True` shows track id on the image, defaults to False
        :type idtext: bool, optional
        :param fontScale: idtext font scale, defaults to 1.5
        :type fontScale: float, optional
        :param fontThickness: idtext fontsize, defaults to 3
        :type fontThickness: int, optional
        :param thickness: bounding box line thickness, defaults to 7
        :type thickness: int, optional
        :return: image with requested options
        :rtype: numpy.ndarray

        .. note:: Events are tracks classified as entrance, exit or pollen (this shows a tracks with color)
        """
        filtered_bodies = [body for body in  self.bodies if len(body) >= min_parts]

        if bbox:
            frame_image = bodies_bbox_drawer(frame_image, filtered_bodies, idtext=idtext, fontScale=fontScale, fontThickness=fontThickness, linethick=thickness)

        if skeleton:
            frame_image = bodies_skeleton_drawer(frame_image, filtered_bodies, thickness=thickness)

        if tracks:
            frame_image = bodies_track_drawer(frame_image, filtered_bodies, direction=track_direction)

        if events:
            tracked_bodies = [body for body in filtered_bodies if body.id != -1]
            frame_image = bodies_event_track_drawer(frame_image, tracked_bodies)

        return frame_image

    def bbox_image(self, idtext=False, suppression=False):
        """Get frame image with bounding boxes

        :param idtext: if `True` shows the id on the image, default to False
        :type idtext: bool
        :param suppression: if `True` not show suppressed bodies, defaults to False
        :type suppression: bool, optional
        :return: image with bounding boxes
        :rtype: numpy.ndarray
        """

        frame = self.image.copy()

        for body in self.bodies:
            if suppression and body.suppressed:
                continue
            frame = bbox_drawer(frame, body, idtext=idtext)

        return frame

    @property
    def skeleton_image(self):
        """Get frame image with skeletons

        :return: image with skeletons
        :rtype: numpy.ndarray
        """

        frame = self.image.copy()

        for body in self.bodies:
            frame = skeleton_drawer(frame, body)
        
        return frame


    def track_image(self, direction=None):
        """Get frame image with tracks

        :param direction: track direction can be "full" show the whole path of the track,
            "backward" only show tracks from the current and the previous frames or
            "forward" only show tracks from the current and the next frames, defaults to "forward", defaults to None
        :type direction: str, optional
        :return: frame image with tracks
        :rtype: numpy.ndarray
        """

        if direction is None:
            direction = Frame.TRACK_DIRECTION

        frame = self.image.copy()

        for body in self.bodies:
            frame = track_drawer(frame, body, direction=direction)
        
        return frame

    @property
    def parts_image(self):
        """Get frame image with keypoints (parts)

        :return: frame image with keypoints
        :rtype: numpy.ndarray
        """

        frame = self.image.copy()
        for body in self._bodies:
            frame = parts_drawer(frame, body._parts)

        return frame

    @property
    def event_image(self):
        """Get frame image with events

        :return: frame image with events
        :rtype: numpy.ndarray

        .. note:: Events are tracks classified as entrance, exit or pollen (this shows a tracks with color)
        """
        frame = self.image.copy()

        for body in self.bodies:
            if body.id == -1:
                continue
            btrack = self.get_track(body)
            frame = event_track_drawer(frame, body, btrack)
        return frame



    @property
    # @lru_cache(maxsize=1000)
    def image(self):
        """Get frame image

        :return: frame image
        :rtype: numpy.ndarray
        """
        # if self._cached_image is None:
        #     self._cached_image = self._video.frame_image(self.id)
        # return self._cached_image.copy()
        return self._video.frame_image(self.id)


    def extract_patch(self, x, y, angle=0, width=160, height=320, cX=None, cY=None):
        """Extract a rotated path from frame

        :param x: `x` of the center point in the input rectangle
        :type x: int
        :param y: `y` of the center point in the input rectangle
        :type y: int
        :param angle: angle in degrees, defaults to 0
        :type angle: float, optional
        :param width: width of the patch rectagle,, defaults to 160
        :type width: int, optional
        :param height: height of the patch rectagle, defaults to 320
        :type height: int, optional
        :param cX: `x` of the center point in the output rectangle, defaults to None
        :type cX: int, optional
        :param cY: `y` of the center point in the output rectangle, defaults to None
        :type cY: int, optional
        :return: image path
        :rtype: numpy.ndarray
        """
        return rotate_bound2(self.image, x, y, angle, width, height, cX, cY)
    
    @staticmethod
    def _extract_bodies_images(image, frame_data, width=None, height=None, cX=None, cY=None, scale=None, suppression=False, min_parts=-1):
        """Extract bodies images from frame image

        :param image: image (ideally should be frame image)
        :type image: numpy.ndarray
        :param frame_data: frame object
        :type frame_data: Frame
        :param width: body image width, defaults to None
        :type width: int, optional
        :param height: body image height, defaults to None
        :type height: int, optional
        :param cX: `x` of the center point in the output rectangle of the body image, defaults to None
        :type cX: int, optional
        :param cY: `y` of the center point in the output rectangle of the body image, defaults to None
        :type cY: int, optional
        :param scale: scale of the body image, defaults to None
        :type scale: float, optional
        :param suppression: if `True` will not return body images from suppressed bodies, defaults to False
        :type suppression: bool, optional
        :param min_parts: this functions only returns bodies that `len(body) >= min_parts`, defaults to -1
        :type min_parts: int, optional
        :return: list of bodies images
        :rtype: list[numpy.ndarray]
        """

        if width is None:
            width = Body.width
        if height is None:
            height = Body.height
        if cX is None:
            cX = Body.cX
        if cY is None:
            cY = Body.cY

        if scale is None:
            scale = Body.scale
        
        images =list()
        bodies = list()


        for body in frame_data:
            if suppression and not body.valid:
                continue

            if len(body) < min_parts:
                continue
            cbodyimg = extract_body(image, body, width=width, 
                                    height=height, cX=cX, cY=cY, scale=scale)

            if Body.out_width is not None and Body.out_height is not None:
                cbodyimg = cv2.resize(cbodyimg, (Body.out_height, Body.out_width))
            images.append(cbodyimg)
            bodies.append(body)
        return bodies, np.array(images)
        


    def bodies_images(self):
        """Return bodies images from frame image
        
        :return: list of images of bodies
        :rtype: list[numpy.ndarray]
        """
        return self._extract_bodies_images(self.image, self)
    
    
    def __repr__(self):
        """Print frame

        :return: shows frame id and bodies in the frame
        :rtype: str
        """
        frepr = "Frame: {}".format(self.id)
        brepr = [repr(b) for b in self._bodies]
        repr_list = [frepr] + brepr
        return "\n".join(repr_list)
    
    def __len__(self):
        """Return the number of detection in the frame

        :return: number of detection in the frame
        :rtype: int
        """
        return len(self.bodies)

    def __getitem__(self, index):
        """Get body by their 'detection id'

        :param index: index of the body in the bodies list
        :type index: int
        :return: body with detection id in the frame
        :rtype: Body

        .. note:: Note that up to now there is not a detection id explicitly coded.
            For now we are using the index in the bodies list.
        """
        return self.bodies[index]

    def bodies_at_point(self, p, silent=False):
        """ Returns bodies that intersect the point `p`

        :param p: point to intersec
        :type p: tuple(int, int)
        :param silent: if `True` will not show warnings, defaults to False
        :type silent: bool, optional
        :return: bodies that intersect the point `p`
        :rtype: list[Body]
        """
        bodies = list()

        for body in self:
            if pointInRotatedBbox(p, body.center, body.angle, body.width, body.height):
                bodies.append(body)

        if not silent:
            if len(bodies) > 1:
                warnings.warn("More than one body in {} point.".format(p))


        return bodies  

        
    def save(self, folder, skeleton=True, bbox=True, tracks=False, events=True, min_parts=-1,
             idtext=False, fontScale=2.5, fontThickness=8):
        """Save image with requested option on `<folder>/{frame_id:09d}.jpg`

        :param folder: folder to save the image
        :type folder: str
        :param skeleton: if `True` draws frame's skeletons on the frame_image, defaults to False
        :type skeleton: bool, optional
        :param bbox: if `True` draws frame's detections bounding boxes on the frame_image, defaults to False
        :type bbox: bool, optional
        :param tracks: if `True` draws frame's detections tracks on the frame_image, defaults to False
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
        :type thickness: int, optional

        .. note:: This function was mean to run in parrallel or concurrent with other frames.
            This approach didn't result on the best approach therefore, this is not used anymore. 
        """
        file_format = "{:09d}.jpg"
        os.makedirs(folder, exist_ok=True)
        im = self._image(skeleton=skeleton, bbox=bbox, tracks=tracks, events=events, min_parts=min_parts, idtext=idtext, fontScale=fontScale, fontThickness=fontThickness)
        
        fname = file_format.format(self.id)
        im_path = os.path.join(folder, fname)
        io.imsave(im_path, im)
        
