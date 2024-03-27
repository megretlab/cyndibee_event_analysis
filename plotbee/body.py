from plotbee.utils import angleBetweenPoints
from plotbee.utils import rotatedBoundBoxPoints, getRotationMatrix
from plotbee.videoplotter import extract_body, skeleton_drawer
from plotbee.tag import get_tag_image
from skimage import io
import numpy as np
import cv2
# from plotbee.video import parse_parts


def parse_parts(parts):
    # json only stores strings as keys
    # openCV needs points as tuple
    return {int(k):[tuple(v[0])] for k, v in parts.items()}



def valid_fn(body):
    """Default Valid function
    Filters suppressed bodies and bodies with `id = -1`.

    :param body: body check
    :type body: Body
    :return: `True` if body is valid else `False`
    :rtype: bool
    """
    if body.suppressed:
        return False
    if body._id == -1:
        return False
    return True



class Body():
    y_offset = 0
    """Class Attribute for y_offset for :func:`Body.center`
        
        
    .. note:: This was used to centralize the body center when there is not a center keypoint available.
        For example, in my case I used to move the center to the thorax using the neck keypoint as `Body.center`.
    """
    width=200 #: Class Attribute for width for plotting 
    height=400 #: Class Attribute for height for plotting 
    scale = 1.0 #: Class Attribute for scale for plotting 
    out_width=None #: Class Attribute for out_width for plotting 
    out_height=None #: Class Attribute for out_height for plotting
    cX=None #: Class Attribute for cX for plotting 
    cY=None #: Class Attribute for cX for plotting 
    ignore_angle = False #: Class Attribute for ignore_angle for plotting (this is used to extract images without angle normalization)

    pollen_threshold = 0.5 #: Class Attribute for pollen_threshold this is used as self.pollen_score > Body.pollen_threshold
    valid_function = valid_fn #: Class Attribute (function) for to decided what is a valid body

    @classmethod
    def load_body(cls, body_dict, frameobj):
        """Body Constructor from dictionary and Frame object

        :param body_dict: recieves a dictionary with the following:
        :type body_dict: dict
        :param frameobj: parent frame (or future parent frame) 
        :type frameobj: Frame
        :return: Body object
        :rtype: Body

        .. note:: body_dict have the following keys:
        *  parts
        *  center_part
        *  angel_conn
        *  connections
        *  id
        *  suppressed
        *  pollen_score
        *  tag (optional)
        *  features (optional)
        *  virtual  (optional)
        *  annotations (optional)
        *  metadata (optional)
        see Body constructor for more information about these keys.

        .. note:: Frame object is used to have a two-way link between Frame and Body.
            Generally, `frameobj` is a new and empty frame that will be populated with the data from `body_dict`.
        """
        if "tag" not in body_dict:
            body_dict["tag"] = None

        if "features" not in body_dict:
            body_dict["features"] = np.array(None)
        else:
            body_dict["features"] = np.array(body_dict["features"])

        if "metadata" not in body_dict:
            body_dict["metadata"] = dict()

        if "annotations" not in body_dict:
            body_dict["annotations"] = dict()

        if "virtual" not in body_dict:
            body_dict["virtual"] = False

        parsed_parts = parse_parts(body_dict["parts"]) 

        body = Body(parsed_parts, body_dict["center_part"],
                    tuple(body_dict["angle_conn"]), body_dict["connections"],
                    frameobj, body_dict["id"], body_dict["suppressed"],
                    body_dict["pollen_score"], body_dict["tag"], body_dict["features"],
                    body_dict["virtual"],
                    body_dict["annotations"], body_dict["metadata"])
        return body


    def __init__(self, parts, center, angle_conn, connections, frame, body_id=-1, suppressed=False,
                pollen_score=0.0, tag=None, features=None, virtual=False, annotations=dict(), metadata=dict()):
        """Default Constructor

        :param parts: detected parts
        :type parts: dict[list[tuple(x:int, y:int)]]
        :param center: part used for keypoint tracking
        :type center: int
        :param angle_conn: connection used to compute angle
        :type angle_conn: tuple(int, int)
        :param connections: predicted connection configuration, note this can be different for each body
        :type connections: list[list[x:int, y:int]]
        :param frame: Frame object which this body belongs
        :type frame: Frame
        :param body_id: track_if if `-1` the body do not have an id assigned, defaults to -1
        :type body_id: int, optional
        :param suppressed: `True` if this body is suppressed, defaults to False
        :type suppressed: bool, optional
        :param pollen_score: pollen score predicted from a pollen detection model, defaults to 0.0
        :type pollen_score: float, optional
        :param tag: tag dictionary computed by apriltag, defaults to None
        :type tag: dict, optional
        :param features: identity features computed by a identity model, defaults to None
        :type features: np.ndarray, optional
        :param virtual: `True`, if this body wasn not detected by beepose but detected by tracking, defaults to False
        :type virtual: bool, optional
        :param annotations: annotations (key-value), defaults to dict()
        :type annotations: dict, optional
        :param metadata: metadata (key-value), defaults to dict()
        :type metadata: dict, optional

        .. note:: The parts is a list of list of keypoints,
            if there is a case that a body has two detected keypoint of on part because a duplicate detection due to the skeleton association.
            Ideally, this list should contain the keypoints with more confidence first.
            I believe I never found a case for this, but the part attribute was left as this just in case.
        """
        self._parts = parts
        self._center_part = center
        self._connections = connections
        self._frame = frame
        self._id = int(body_id)
        self._angle_conn = angle_conn
        self._prev = None
        self._next = None
        self.suppressed = suppressed #: Instance attribute to suppress body. Generally, use during tracking to remove double detections.
        self.pollen_score = pollen_score
        self.tag = tag
        self.features = features
        self.virtual = virtual
        self._annotations = annotations
        self._metadata = metadata

    def annotate(self, key, value):
        """Make a body annotation

        :param key: name of the annotation field.
        :type key: Any hashable (ideal str)
        :param value: Value of the annotation.
        :type value: Any

        .. note:: Note that annotation can overwrite previous annotations.
        """
        self._annotations[key] = value

    def set_metadata(self, key, value):
        """Set metadata

        :param key: name of the metadata field.
        :type key: Any hashable (ideal str)
        :param value: Value of metadata.
        :type value: Any

        .. note:: Note that `set_metadata` can overwrite previous metadata.
        """
        self._metadata[key] = value

    @property
    def annotations(self):
        """get annotation dictionary

        :return: annotations
        :rtype: dict
        """
        return self._annotations

    @property
    def metadata(self):
        """get metadata dictionary

        :return: metdatda
        :rtype: dict
        """
        return self._metadata
    
    @property
    def valid(self):
        """Return if body is a valid

        :return: check if body is a valid body with the function defined on :func:`Body.valid_function`.
        :rtype: bool

        .. tip:: This is customizable, and be useful to filter bodies in multiple scenarios such as plotting or Frame.bodies
        """
        return self.valid_function()

    @property
    def frameid(self):
        """Get frame id

        :return: frame id
        :rtype: int
        """
        return self._frame.id

    @property
    def tag_id(self):
        """Tag id

        :return: `None` if is the body is not tagged
        :rtype: float
        """
        if self.tag is None:
            return None
        else:
            return self.tag["id"]

    @property
    def video_name(self):
        """Get video name

        :return: video name
        :rtype: str
        """
        return self._frame.video_name


    @property
    def connections(self):
        """Get parts connections (skeleton)

        :return: list of parts connections
        :rtype: list[list[x:int, y:int]]
        """
        return self._connections


    @property
    def id(self):
        """Get body id

        :return: get track id
        :rtype: int
        """
        return self._id


    @property
    def parts(self):
        """Get detected parts

        :return: detected parts
        :rtype: dict[list[tuple(x:int, y:int)]]
        """
        return self._parts
    

    @property
    def center(self):
        """get body center

        :return: get center point
        :rtype: tuple(x, y)

        .. note:: `y = Body.center.y - Body.y_offset` 
        """
        x, y = self._parts[self._center_part][0]
        return x, y - Body.y_offset


    @property
    def angle(self):
        """Return angle

        :return: Return angle in degrees from the angle connection
        :rtype: float
        """
        p1 = self.parts[self._angle_conn[0]][0]
        p2 = self.parts[self._angle_conn[1]][0]

        return angleBetweenPoints(p1, p2)

    @property
    def pollen(self):
        """Return if the body has pollen

        :return: return if is body with pollen
        :rtype: bool

        .. note:: Pollen decision is based on `self.pollen_score > Body.pollen_threshold`
        """
        if self.pollen_score > Body.pollen_threshold:
            return True
        else:
            return False

    def _image(self, width=None, height=None, cX=None, cY=None, scale=None, ignore_angle=None, erase_tag=False):
        """ Get a body image

        :param width: image width, defaults to None
        :type width: int, optional
        :param height: image height, defaults to None
        :type height: int, optional
        :param cX: `x` of the center point in the output rectangle of the body image, defaults to None
        :type cX: int, optional
        :param cY: `y` of the center point in the output rectangle of the body image, defaults to None
        :type cY: int, optional
        :param scale: scale of the body image, defaults to None
        :type scale: float, optional
        :param ignore_angle: ignore angle normalization, defaults to None
        :type ignore_angle: bool, optional
        :param erase_tag: if body is a tagged bee then the tag will be erase from the image, defaults to False
        :type erase_tag: bool, optional
        :return: body image
        :rtype: numpy.ndarray

        .. note:: If the following params width, height, cX, cY, scale and ignore_angle are set to None,
            default values will be used. These default values are the class attributes Body.`param` (e.g. Body.width).

        .. tip:: For a batch image extraction or plotting use class attributes to change image extarction at global (Body) level.
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
        if ignore_angle is None:
            ignore_angle = Body.ignore_angle

        if erase_tag and self.tag is not None:
            pts = np.array(self.tag["p"]).astype(np.int32)
            pts = pts.reshape((-1,1,2))
            frame = cv2.fillPoly(self._frame.image,[pts], (0,0,0))
            frame = cv2.polylines(frame,[pts],True,(0,0,0),35)
        else:
            frame = self._frame.image
        
        body_image = extract_body(frame, self, width=width, 
                            height=height, cX=cX, cY=cY, scale=scale, ignore_angle=ignore_angle)
        if Body.out_width is None and Body.out_height is None:
            return body_image
        return cv2.resize(body_image, (Body.out_height, Body.out_width))

    @property
    def image(self):
        """Get Body image 

        :return: body image
        :rtype: numpy.ndarray
        """
        return self._image()

    @property
    def skeleton_image(self):
        """Get body image with skeleton overlay

        :return: body image with skeleton on top
        :rtype: numpy.ndarray
        """
        frame = self._frame.image    
        frame = skeleton_drawer(frame, self)
        
        body_image = extract_body(frame, self, width=Body.width, 
                            height=Body.height, cX=Body.cX, cY=Body.cY, scale=Body.scale)
        if Body.out_width is None and Body.out_height is None:
            return body_image
        return cv2.resize(body_image, (Body.out_height, Body.out_width))



    @property
    def prev(self):
        """Get the previous body in the track. This is only work with tracks.

        .. warning:: Not sure id new tracking support this linked-list track.
        """
        return self._prev

    @prev.setter
    def prev(self, p):
        """ Set previous body in the track. This is mostly used during tracking or loading a plotbee video from file. 

        :param p: previous body in the track
        :type p: Body
        """
        self._prev = p

    @property
    def next(self):
        """Get the next body in the track. This is only work with tracks.

        .. warning:: Not sure id new tracking support this linked-list track.
        """
        return self._next

    @next.setter
    def next(self, n):
        """ Set next body in the track. This is mostly used during tracking or loading a plotbee video from file. 

        :param p: next body in the track
        :type p: Body
        """
        self._next = n
    
    
    def set_id(self, i):
        """Set track id (body id)

        :param i: body id
        :type i: int
        """
        self._id = i
    
    
    def cbox(self, w_size=100, h_size=200):
        """Get keypoint of a rectangle in the center of the body

        :param w_size: rectangle width, defaults to 100
        :type w_size: int, optional
        :param h_size: rectangle height, defaults to 200
        :type h_size: int, optional
        :return: rectangle points (top-left x, top-left y, bottom-right x, bottom-right y)
        :rtype: tuple(x1, y1, x2, y2)
        """
        x, y = self.center
        return (x - w_size , y - h_size, x + w_size, y + h_size)

    def boundingBox(self):
        """Get keypoint of a *rotated* bounding box of the body

        :return: 4 x-y points of the rotated bounding box.
        :rtype: tuple(tuple(x, y))

        .. note:: this bounding box uses the following as parameters `self.center`, `self.angle`, `Body.width` and `Body.height`.
        """
        center = self.center
        angle = self.angle
        return rotatedBoundBoxPoints(center, angle, Body.width, Body.height)

    @property
    def skeleton(self):
        """Get skeleton

        :return: Skeleton connections (list of lines)
        :rtype: list[tuple(tuple(x:int, y:int))]
        """
        points = list()
        for part1, part2 in self._connections:
            if part1 not in self._parts:
                continue
            if part2 not in self._parts:
                continue
            points.append((self._parts[part1][0], self._parts[part2][0]))
        return points

    
    def __len__(self):
        """Return the number of detected parts(not duplicates) 

        :return: number of detected parts
        :rtype: int
        """
        return len(self._parts.keys())


    def __repr__(self):
        """Print body info

        :return: Show body id, parts points and if virtual
        :rtype: str
        """
        coords = repr(self.parts)
        coords = coords[coords.find('{'):-1]
        if (self.virtual):
            return "Body(id={}, parts={}, virtual=True)".format(self.id, coords)
        else:
            return "Body(id={}, parts={})".format(self.id, coords)


    def save(self, path, width=None, height=None, cX=None, cY=None, erase_tag=False):
        """Save body image

        :param path: path to save the image
        :type path: str
        :param width: image width, defaults to None
        :type width: int, optional
        :param height: image height, defaults to None
        :type height: int, optional
        :param cX: `x` of the center point in the output rectangle of the body image, defaults to None
        :type cX: int, optional
        :param cY: `y` of the center point in the output rectangle of the body image, defaults to None
        :type cY: int, optional
        :param scale: scale of the body image, defaults to None
        :type scale: float, optional
        :param ignore_angle: ignore angle normalization, defaults to None
        :type ignore_angle: bool, optional
        :param erase_tag: if body is a tagged bee then the tag will be erase from the image, defaults to False
        :type erase_tag: bool, optional
        """
        im = self._image(width=width, height=height, cX=cX, cY=cY, erase_tag=erase_tag)
        io.imsave(path, im)


    def info(self):
        """Get body info as dictionary

        :return: body info
        :rtype: dict

        .. warning:: This is old. Better use `self.params`.
        """
        x, y = self.center
        
        info = {
            "track_id": self.id,
            "frame": self.frameid,
            "angle": self.angle,
            "x": x,
            "y": y,
            "parts_num": len(self),
            "tag_id": self.tag_id,
            "virtual":self.virtual,
            "pollen_score": self.pollen_score
        }

        if self.tag_id is None:
            info["tagx"] = None
            info["tagy"] = None
            info["taghamming"] = None
            info["tagdm"] = None
        else:
            tagx, tagy = self.tag["c"]
            info["tagx"] = tagx
            info["tagy"] = tagy
            info["taghamming"] = self.tag["hamming"]
            info["tagdm"] = self.tag["dm"]

        
        return info

    def tag_erased_image(self):
        """Get tag erased image

        :return: return body image. If body is tagged then erase the tag.
        :rtype: numpy.ndarray
        """
        if self.tag is None:
            return self.image
        else:
            return self._image(erase_tag=True)

    # def get_abdomen_image(self):
    #     frame = self._frame.image
    #     image_size = frame.shape[:2]
    #     width=Body.width
    #     height=Body.height
    #     x, y = self.center
    #     angle = self.angle
    #     return getRotationMatrix(image_size,x,y,angle, width, height)


    def tag_image(self):
        """Return image

        :return: If body is tagged then return image of the Apriltag else return `np.array([])`
        :rtype: np.ndarray
        """
        if self.tag is None:
            return np.array([])
        else:
            return get_tag_image(self)

    def params(self):
        """Get body dictionary
        
        :return: return a dictionary of the body.
        :rtype: dict

        .. warning:: For developers. If body has new attributes,
            these attributes should be manually code into this function to be able save and load bodies.
        """
        body_dict = dict()
        body_dict["parts"] = self.parts
        body_dict["center_part"] = self._center_part
        body_dict["connections"] = self.connections
        body_dict["frameid"] = self.frameid
        body_dict["id"] = self.id
        body_dict["angle_conn"] = self._angle_conn
        body_dict["suppressed"] = self.suppressed
        body_dict["pollen_score"] = self.pollen_score
        body_dict["tag"] = self.tag
        body_dict["features"] = np.array(self.features).tolist()
        body_dict["virtual"] = self.virtual
        body_dict["metadata"] = self._metadata
        body_dict["annotations"] = self._annotations

        return body_dict
