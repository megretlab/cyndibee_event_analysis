import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from decord import VideoReader
from decord import cpu, gpu

import gc

def load_frame(video, frame, videoreader=None):
    # Load one frame from a video
    # videoreader: decord.VideoReader
    if (videoreader is None):
        vr = VideoReader(video, ctx=cpu(0))
    else:
        vr = videoreader
    frameimg = vr[frame].asnumpy()
    
    return frameimg

def plot_track(row, frame=None, videoreader=None, ax=None):
    if (frame is None):
        frame = row["track_startframe"]
    frameimg = load_frame(row["video_filename"], frame, videoreader)
    start_x = row["track_startx"] 
    start_y = row["track_starty"]
    
    end_x = row["track_endx"]
    end_y = row["track_endy"]
    
    p1 = (start_x, start_y)
    p2 = (end_x, end_y)

    # Draw in the image directly vs plot on top
    #frameimg = cv2.line(frameimg, p1, p2, [255, 0, 0], 7)
    if (ax is None):
        ax = plt.gca()
    ax.imshow(frameimg)
    ax.plot( (start_x,end_x), (start_y,end_y), 'r.-' )
    ax.plot( (start_x,), (start_y,), 'ro' )
    ax.plot( (end_x,), (end_y,), 'rx' )

## GUI for frame by frame navigation in a video

# Need 
# %matplotlib widget

# Example of faster GUI to plot images
from ipywidgets import widgets, interact
#from skimage.transform import resize, rescale
import PIL

class GUI_PlotVideo:
    def __init__(self, video_filename, frame, vr=None):
        #self.event = event

        # Open video
        if (vr is None):
            gc.collect()
            self.vr = VideoReader(video_filename, ctx=cpu(0))
        else:
            self.vr = vr

        # Create sliders
        self.nframes = len(self.vr)
        self.frame_slider = widgets.IntSlider(min=0, max=self.nframes-1, step=1, value=frame, continuous_update=False)
        self.scale_slider = widgets.IntSlider(min=0, max=6, step=1, value=0, continuous_update=False)

        # Create one figure for all display
        plt.rcParams['figure.figsize'] = [10, 10]
        fig = plt.figure(figsize=(8,4))
        fig.set_tight_layout(True)
        fig.canvas.header_visible = False
        ax = fig.add_subplot(111)
        self.fig = fig
        self.ax = ax

        # Initialize plot to set the image extent
        #frame = self.event['track_startframe']
        #frame=0
        frameimg = self.vr[frame].asnumpy()
        ax.clear()
        ax.imshow(self.imscale(frameimg, 1.0))
        ax.set_title(f"{frame}")

        # Start the GUI
        interact(self.show_frame, frame=self.frame_slider, resolution_level=self.scale_slider)

    def set_video(self, video_filename, vr=None):
        # Open video
        if (vr is None):
            self.vr = VideoReader(video_filename, ctx=cpu(0))
        else:
            self.vr = vr

        self.frame_slider.value = 0
        self.frame_slider.max = self.nframes-1
        self.show_frame(0, self.scale_slider.value)

    def imscale(self, img, s):
        h,w,_ = img.shape
        return np.asarray(PIL.Image.fromarray(img).resize((int(w * s), int(h * s))))

    def show_frame(self, frame, resolution_level):
        # Load image
        frameimg = self.vr[frame].asnumpy()
        scale = 2**resolution_level
        frameimg = self.imscale(frameimg, 1.0/scale)
        
        fig = self.fig
        ax = self.ax
        # Alternate method: reset the whole axes
        #ax.clear()
        #ax.imshow(frameimg)
        # Replace image data (keep same axis extent)
        ax.images[0].set_data( frameimg )
        ax.set_title(f"Frame {frame}, Scale 1/{scale}")
        #fig.canvas.draw()



# GUI for choosing an event and displaying it

from ipywidgets import widgets
from ipywidgets import Layout, VBox, HBox
from collections import namedtuple
import time
from functools import lru_cache
#from skimage.transform import resize, rescale
import PIL
#from dataclasses import dataclass
from traitlets.utils.bunch import Bunch

from .labelbee_link import labelbee_url


class GUI_BrowseEvents:
    def __init__(self, vdf, tids=None, hide_noise=True):

        self.vdf = vdf

        if (tids is None):
            tids = vdf['track_tagid'].unique()
        self.tids = tids

        self.hide_noise = hide_noise

        self.vr = None

        # Create one figure
        plt.ioff()
        fig = plt.figure(figsize=(8,4))
        plt.ion()
        fig.canvas.header_visible = False
        fig.clear()
        ax = fig.add_subplot(111)
        ax.clear()
        ax.imshow(np.array([[0,1]]))
        ax.set_title(f"Initialized")
        fig.set_tight_layout(True)

        self.fig = fig
        self.ax = ax

        # Widgets
        self.tids_widget = widgets.Dropdown(options = self.tids)
        options = self.get_event_options(self.tids_widget.value)
        self.events_widget = widgets.Dropdown(options=options)
        self.frame_slider = widgets.IntSlider(min=0, max=10000, step=1, value=0, continuous_update=False)
        self.scale_widget = widgets.Dropdown(options=[0,1,2,3,4,5,6])

        self.debug_view = widgets.Output(layout=Layout(height='200px', overflow_y='auto'))
        self.event_view = widgets.Output(layout=Layout(height='400px', max_width='600px', overflow_x='auto', overflow_y='auto'))

        self.tids_widget.observe(self.tid_changed, names="value")
        self.events_widget.observe(self.event_changed, names="value")
        self.frame_slider.observe(self.frame_changed, names="value")
        self.scale_widget.observe(self.redraw, names="value")

        self.scene = VBox([ 
                        HBox([self.tids_widget, self.events_widget]),
                        HBox([VBox([HBox([self.frame_slider, self.scale_widget]), self.fig.canvas], layout=Layout(border='1px solid black', width='100%',overflow_x='auto',overflow_y='auto')), 
                            VBox([self.event_view], layout=Layout(border='1px solid black', width='100%',overflow_x='auto',overflow_y='auto')) ]),
                        HBox([self.debug_view], layout=Layout(border='1px solid black'))
                    ], layout=Layout(width='100%',overflow_x='auto'))
        display(self.scene)

        # Send event to refresh first display
        self.tid_changed(Bunch(new=self.tids[0]))
        self.event_changed(Bunch(new=self.events_widget.options[0][1]))

    def event_name(self, row):
        noise = False
        event_name = ""
        if row.pollen:            event_name += "pollen"
        if row.entering:          event_name += "entering"
        if row.leaving:           event_name += "leaving"
        if row.walking:           event_name += "walking"
        if row.entering_leaving:  event_name += "entering_leaving"
        if event_name == "":
            event_name += "noise"
            noise = True
            
        event_name += " - {}".format(row.datetime)
        return noise, event_name

    def get_event_options(self, tid):
        options = list()
        vdf = self.vdf
        for i, row in vdf[vdf.track_tagid == tid].iterrows():
            noise, name = self.event_name(row)
            if (self.hide_noise and noise): continue;
            D = dict(row)
            D['loc']=i
            options.append((name, D))
        return options

    @lru_cache(maxsize=20)
    def get_frame(self, frame):
        img = self.vr[frame].asnumpy()
        self.vr.seek(0)
        return img

    def imscale(self, img, s):
        h,w,_ = img.shape
        return np.asarray(PIL.Image.fromarray(img).resize((int(w * s), int(h * s))))
        
    # # Decorator
    # def debug(clear_output=False):
    #     def wrapped_fun(fn):
    #         def wraps(*args, **kwargs):
    #             with debug_view:
    #                 if (clear_output):
    #                     debug_view.clear_output()
    #                 fn(*args, **kwargs)
    #         return wraps
    #     return wrapped_fun

    def tid_changed(self, change):
        with self.debug_view:
            self.debug_view.clear_output()

            ctid = change.new
            print('tid_changed',ctid)
            options = self.get_event_options(ctid)
            self.events_widget.options = options

    def event_changed(self, change):
        with self.debug_view:
            event = change.new
            print(f"event_changed loc={event['loc']}: {event}")
            
            eventrow = self.vdf.loc[event['loc']]
            
        with self.event_view:
            self.event_view.clear_output()
            with pd.option_context('display.max_colwidth', 200):
                display(pd.DataFrame(eventrow))
                print(eventrow['video_filename'])

        with self.debug_view:
            print('LOADING VIDEO...')
            tic = time.perf_counter()
            if (self.vr is not None):
                # Need to kill the previous VideoReader before opening a new one
                # else performance is much slower (up to 30s instead of 2s)
                del self.vr
            gc.collect()
            self.vr = VideoReader(event['video_filename'], ctx=cpu(0))
            toc = time.perf_counter()
            print(f'LOADED. ({toc-tic:.2f}s)')
            self.vr.seek(0)
            self.get_frame.cache_clear()
            
            self.ax.clear()

            #frameimg = glob.vr[0].asnumpy()
            #ax.imshow(imscale(frameimg*0, 1.0)) # Dummy
            
            frame = event["track_startframe"]
            self.frame_slider.min = -10000
            self.frame_slider.max = frame+100
            self.frame_slider.min = frame-100
            self.frame_slider.value = frame

            print(f'Labelbee: {labelbee_url(eventrow)}')
            
            self.redraw()

    def frame_changed(self, change):
        frame = change.new
        #with self.debug_view:
        #    print('frame_changed',frame)
        self.redraw()

    def redraw(self, change=None):
        with self.debug_view:
            event, frame = self.events_widget.value, self.frame_slider.value
            #print(event, frame)
            resolution_level = self.scale_widget.value
            scale = 2**resolution_level
            
            #print('REDRAW', frame, resolution_level)
            loc = event['loc']
            
            #plot_track(event, frame, vr, ax)
            tic = time.perf_counter()
            frameimg = self.get_frame(frame)
            toc = time.perf_counter()
            #print(f'TOOK {toc-tic:.2f}s')

            ax = self.ax
            if len(self.ax.images)==0:
                ax.clear()
                ax.imshow(self.imscale(frameimg, 1.0/scale), extent=[0,frameimg.shape[1],frameimg.shape[0],0])
                #ax.plot([0,3],[6,4],'r-')

                # OVERLAY TRACK BEGIN/END
                start_x = event["track_startx"] 
                start_y = event["track_starty"]
                end_x = event["track_endx"]
                end_y = event["track_endy"]

                #ax.plot( (start_x,end_x), (start_y,end_y), 'r.-' )
                #ax.plot( (start_x,), (start_y,), 'ro' )
                #ax.plot( (end_x,), (end_y,), 'rx' )
                #ax.text( start_x,start_y, 'S', color='red', fontsize=24.0, horizontal_alignemnt='center' )
                ax.arrow( start_x, start_y, end_x-start_x, end_y-start_y, color='red', length_includes_head=True, head_width=40.0 )
            else:
                ax.images[0].set_data( self.imscale(frameimg, 1.0/scale) )
            ax.set_title(f"Event {loc}, Frame {frame}")
    
    #fig.canvas.draw()
    
#interact(f, tid=tids_widget, event=events_widget, frame=frame_slider);



