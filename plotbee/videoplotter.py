import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from plotbee.utils import id2color, rotate_bound2, trackevent2color, rescale_image
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm

YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)


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

RADIUS = 10
THICKNESS = -1


def imshow(frame, **kwargs):
    plt.imshow(frame._image(**kwargs))

def bbox(frame, idtext=False, ax=None, suppression=False):
    if ax:
        ax.imshow(frame.bbox_image(idtext=idtext, suppression=suppression))
    else:
        plt.imshow(frame.bbox_image(idtext=idtext, suppression=suppression))

def skeleton(frame):
    plt.imshow(frame.skeleton_image)

def tracks(frame, direction="forward"):
    plt.imshow(frame.track_image(direction=direction))

def parts(frame):
    plt.imshow(frame.parts_image)

def plot(frame, skeleton=False, bbox=False, tracks=False, min_parts=5):
    plt.imshow(frame._image(skeleton=skeleton,
                bbox=bbox, tracks=tracks,
                min_parts=min_parts))

def events(frame):
    plt.imshow(frame.event_image)


def bbox_drawer(frame_image, body, idtext=False, idpos='avg', fontScale=1.5, fontThickness=3, linethick=7):
    color = id2color(body.id)
    font = cv2.FONT_HERSHEY_DUPLEX
    #fontScale = 1.5
    p1, p2, p3, p4 = body.boundingBox()
        
    thick = 3 if body.virtual else linethick  # Thin line if virtual
    frame = cv2.line(frame_image, p1, p2, color=color, thickness=thick)
    frame = cv2.line(frame_image, p2, p3, color=color, thickness=thick)
    frame = cv2.line(frame_image, p3, p4, color=color, thickness=thick)
    frame = cv2.line(frame_image, p4, p1, color=color, thickness=thick)

    if idtext:
        if (idpos=='p1'):
            text = "id={}".format(body.id)
            P = p1
            cv2.putText(frame_image, text, P, font, fontScale, color=color, thickness=3)
        else:
            thick = (fontThickness+1)//2 if body.virtual else fontThickness   # Thin line if virtual
            text = "{}".format(body.id)
            P = tuple( [int(x) for x in np.mean( np.array((p1,p2,p3,p4)), axis=0)] )
            textsize = cv2.getTextSize(text, font, fontScale, thickness=thick)
            #print(P,textsize[0])
            pos = (2*P[0] - textsize[0][0]) // 2, (2*P[1] + textsize[0][1]) // 2
            cv2.putText(frame_image, text, pos, font, fontScale, color=color, thickness=thick)

    return frame_image

def bodies_bbox_drawer(frame_image, bodies, **kwargs):
    for body in bodies:
        frame_image = bbox_drawer(frame_image, body, **kwargs)
    return frame_image


def skeleton_drawer(frame_image, body, idtext=False, thickness=7):
    color = id2color(body.id)
    for p1, p2 in body.skeleton:
        frame_image = cv2.line(frame_image, p1, p2, color=color, thickness=thickness)
    return frame_image

def bodies_skeleton_drawer(frame_image, bodies, **kwargs):
    for body in bodies:
        frame_image = skeleton_drawer(frame_image, body, **kwargs)
    return frame_image


def parts_drawer(frame_image, parts_dict):
    for part, points in parts_dict.items():
        color = COLOR_BY_PART[str(part - 1)]
        for point in points:
            p = tuple(point[:2])
            frame_image = cv2.circle(frame_image, p, RADIUS, color, THICKNESS)
    return frame_image


def extract_body(frame, body, width=200, height=400, cX=None, cY=None, scale=1.0, ignore_angle=False):
    x, y = body.center
    
    if ignore_angle:
        angle = 0
    else:
        angle = body.angle

    return rotate_bound2(frame,x,y,angle, width, height, cX, cY, scale)


def track_drawer(frame, body, thickness=3, direction="forward"):
    points = list()
    color = id2color(body.id)
    x = body

    if direction.lower() == "backward":
        
        while x.prev is not None:
            p = x.center
            points.append(np.int32(p))
            x = x.prev
    else:
        if direction.lower() == "full":
            while x.prev is not None:
                x = x.prev

        while x.next is not None:
            p = x.center
            points.append(np.int32(p))
            x = x.next

    points = np.array([points], dtype=np.int32)

    return cv2.polylines(frame, [points], False, color, thickness)

def bodies_track_drawer(frame_image, bodies, **kwargs):
    for body in bodies:
        frame_image = track_drawer(frame_image, body, **kwargs)
    return frame_image



def event_track_drawer(frame, body, track, thickness=3):
    points = list()
    color = trackevent2color(track)
    if color is None:
        return frame
    x = body

    while x.next is not None:
        p = x.center
        points.append(np.int32(p))
        x = x.next

    points = np.array([points], dtype=np.int32)

    return cv2.polylines(frame, [points], False, color, thickness)

def bodies_event_track_drawer(frame_image, bodies, **kwargs):
    for body in bodies:
        track = body._frame.get_track(body)
        frame_image = event_track_drawer(frame_image, body, track, **kwargs)
    return frame_image


def track_images(track, figsize=(10, 20)):
    num_images = len(track)
    rows = (num_images // 10) + 1


    fig, ax = plt.subplots(nrows=rows, ncols=10, figsize=figsize)
    ax = ax.ravel()

    for i, body in enumerate(track):
        ax[i].imshow(body.image)


def tag_images(video, save_folder=None, black_listed_ids=[15, 16, 13]):
    tagged_bees = list()
    
    
    for frame in video:
        for body in frame:
            if body.tag_id is not None:
                if body.tag_id not in black_listed_ids:
                    tagged_bees.append(body)
                
    num_images = len(tagged_bees)
    rows = (num_images // 10) + 1
    figure_height = int(2.28 * rows) + 1

    fig, ax = plt.subplots(nrows=rows, ncols=10, figsize=(20, figure_height))

    ax = ax.ravel()

    for i, body in enumerate(tagged_bees):
        ax[i].imshow(body.image)
        ax[i].set_title(str(body.tag_id))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    if save_folder is not None:
        _ , video_name = os.path.split(video.video_path)
        video_name, ext = os.path.splitext(video_name) 
        path = os.path.join(save_folder, video_name + ".pdf")
        plt.savefig(path)

def tagged_contact_sheet(bee_list, save_path=None, cols=10, bodies=True):

    tag_image_folder = "/home/jchan/tags_dataset/tag25h5inv/png/"
    num_images = len(bee_list)
    rows = (num_images // cols)
    figure_height = int(2.5 * rows) + 1


    if bodies:
        rows *= 2
        figure_height *= 2.0
        fig, ax = plt.subplots(nrows=rows, ncols=cols + 1, figsize=(15, figure_height))

        tag_ax = ax[::2, ...].ravel()
        body_ax = ax[1::2, ...].ravel()
        
        j = 0
        for i, (tax, bax) in enumerate(zip(tag_ax, body_ax)):
            
            if j == len(bee_list):
                break
            body = bee_list[j]

            if i % (cols + 1) == 0:
                tag_path = os.path.join(tag_image_folder, "keyed{:04}.png".format(body.tag_id))
                tag_image = cv2.imread(tag_path)
                tax.imshow(tag_image)
                tax.set_ylabel(str(body.tag_id) + "       ", rotation='horizontal')
                bax.set_visible(False)
            else:
                tax.imshow(body.tag_image())
                tax.set_xlabel(str(body.tag["hamming"]))
                tax.set_title("{0:.2f}".format(body.tag["dm"]))

                bax.imshow(body.image)
                bax.set_xlabel(str(body.frameid))
                vname = body.video_name
                vname, ext = os.path.splitext(vname)
                bax.set_ylabel(vname)
                j += 1
            tax.set_xticks([])
            tax.set_yticks([])
            bax.set_xticks([])
            bax.set_yticks([])
    else:

        fig, ax = plt.subplots(nrows=rows, ncols=cols + 1, figsize=(15, figure_height))

        axes = ax.ravel()
        j = 0
        for i, ax in enumerate(axes):
            
            if j == len(bee_list):
                break
            body = bee_list[j]

            if i % (cols + 1) == 0:
                tag_path = os.path.join(tag_image_folder, "keyed{:04}.png".format(body.tag_id))
                tag_image = cv2.imread(tag_path)
                ax.imshow(tag_image)
                ax.set_ylabel(str(body.tag_id) + "       ", rotation='horizontal')
            else:
                ax.imshow(body.tag_image())
                ax.set_xlabel(str(body.tag["hamming"]))
                ax.set_title("{0:.2f}".format(body.tag["dm"]))
                j += 1
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(hspace=0.4)
        

    if save_path is not None:
        path = os.path.join(save_path)
        plt.savefig(path, bbox_inches='tight')

def contact_sheet(bee_list, save_path=None, cols=10):

    num_images = len(bee_list)
    rows = (num_images // cols)
    if num_images % cols:
        rows += 1
    figure_height = int(2.5 * rows) + 1

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, figure_height))

    axes = ax.ravel()
    [ax.set_xticks([]) for ax in axes]
    [ax.set_yticks([]) for ax in axes]

    for i, ax in enumerate(axes):
        
        if i == len(bee_list):
            break
        body = bee_list[i]

        ax.imshow(body.image)
        ax.set_xlabel(str(body.frameid))
        ax.set_title("Pollen: {0:.2f}".format(body.pollen_score))

    plt.subplots_adjust(hspace=0.4)
        

    if save_path is not None:
        path = os.path.join(save_path)
        plt.savefig(path, bbox_inches='tight')




    

    
class VideoAnimation():
    
    def __init__(self, video, skeleton=False, bbox=True, tracks=False, events=False, min_parts=-1, track_direction="forward", idtext=False, fontScale=2.5, fontThickness=8, rescale_factor=4):
        
        self.video = video
        
        self.plot_params = {
            "skeleton":skeleton,
            "bbox":bbox,
            "tracks":tracks,
            "events":events,
            "min_parts":min_parts,
            "idtext":idtext,
            "fontScale":fontScale,
            "fontThickness":fontThickness,
            "track_direction":track_direction
        }
        
        self.rescale_factor = rescale_factor
        
        
        self.fig = plt.figure(figsize=(20, 12))
        self.ax = plt.axes()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        frame_image = self.video[0]._image(**self.plot_params)
        if self.rescale_factor > 1:
            frame_image = rescale_image(frame_image, self.rescale_factor)

        self.empty_frame = np.zeros(frame_image.shape)
        self.image = self.ax.imshow(self.empty_frame, interpolation="nearest")
        self.fig.tight_layout()
        plt.close()
    
    def get_init(self):
        
        def init():
            frame_id = self.video[0].id
            self.ax.set_title("Frame :{}".format(frame_id))
            return [self.image]
        
        return init
    
    def get_animate(self, video_stream, pbar):
        def animate(i):
            fid, frame_image = video_stream.read()
            frame_id = self.video[i].id
            if fid:
                frame_image = self.video[i].draw_frame_image(frame_image, **self.plot_params)
                if self.rescale_factor > 1:
                    frame_image = rescale_image(frame_image, self.rescale_factor)
            else:
                frame_image = self.empty_frame
                
            self.image.set_array(frame_image)
            self.ax.set_title("Frame: {}".format(frame_id))
            pbar.update(1)
            return [self.image]
        return animate
    
    def show(self):
        
        video_stream = self.video.get_video_stream()
        pbar = tqdm(total=len(self.video), position=0, leave=True)
        
        animate_func = self.get_animate(video_stream, pbar)
        
        
        anim = FuncAnimation(self.fig, animate_func, init_func=self.get_init(),
                               frames=len(self.video), interval=50, blit=True)
        
        out = HTML(anim.to_html5_video())
            
        video_stream.release()
        pbar.close()
        return out
        
    def save(self, filename, fps=20):
        video_stream = self.video.get_video_stream()
        pbar = tqdm(total=len(self.video), position=0, leave=True)
        
        animate_func = self.get_animate(video_stream, pbar)
        
        
        anim = FuncAnimation(self.fig, animate_func, init_func=self.get_init(),
                               frames=len(self.video), interval=50, blit=True)
        
        anim.save(filename, fps=fps)
            
        video_stream.release()
        pbar.close()
        return