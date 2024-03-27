# import pigeonXT as pixt
import os
from plotbee.body import Body
from plotbee.frame import Frame
from plotbee.video import Video
from plotbee.track import Track

import plotbee.videoplotter as vplt
from collections import defaultdict
import cv2
from PIL import Image
from collections import defaultdict
import functools
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import random

from IPython.display import clear_output, display
from ipywidgets import (Button,
                        Dropdown,
                        FloatSlider,
                        HBox,
                        HTML,
                        IntSlider,
                        Output,
                        Text,
                        Label,
                        Layout)
# from traitlets.traitlets import default

DEFAULTS = {'tag':'no',
            'pollen':'no_pollen', 
            'view':'top',
            'blurry':'no',
            'occluded':'no',
            'crowded':'no',
            'lighting':'sunny',
            'track': "-1",
            'tag_id': "-1"}

OPTIONS = {'tag':['yes','no','blurred','?'], 
             'pollen':['no_pollen','yellow','black','white','red','?'],
             'view':['top','bottom','side','back','front','?'],
             'blurry':['yes','a bit','no','?'],
             'occluded':['yes','no','?'],
             'crowded':['yes','no','?'],
             'lighting':['sunny','dark','?'],
             'track':'', 
             'tag_id':''}

def isBodyAnnotated(body):
    return bool(body.annotations)

def isBodyCompletedAnnotated(options, body):
    annotation = body._annotations
    if isAnnotationComplete(annotation, options):
        return True
    return False   

def isAnnotationComplete(annotations, options):
    for key in options:
        if key not in annotations:
            return False
    return True

def record_annotation(body, annotation):
    for key, value in annotation.items():
        body.annotate(key, value)
        
def count_annotated_bodies(bodies, options):
    annotated_bodies_amount = 0
    for body in bodies:
        if isBodyCompletedAnnotated(options, body):
            annotated_bodies_amount += 1
    return annotated_bodies_amount

def body_info_str(body):
    return str(body.info())

def show_body(body):
    image = load_image(body)
    display(Image.fromarray(image))

@lru_cache(maxsize=128)
def load_image(body):
    return body._image()

def preload_bodies(bodies):
    """
    preload images in the lru_cache
    """
    for body in bodies:
        load_image(body)


class MultiLabelButton(object):
    def __init__(self, description, task_name):
        self.description = description
        self.task_name = task_name
        self.button = Button(description=description)


        
class BodyAnnotator():

    ACTIVE = 'lightgreen'
    DEACTIVE = None


    def __init__(self, video, options=OPTIONS, defaults=DEFAULTS, shuffle=True, display_fn=show_body, output_file=None):

        self.video = video
        self.options = options
        self.defaults = defaults
        self.shuffle = shuffle
        self.output_file = output_file
        self.current_index = 0
        self.display_fn = display_fn

    def annotate_video(self):
        self.bodies = list()
        for frame in self.video:
            for body in frame:
                self.bodies.append(body)
        self.annotate_bodies(self.bodies)

    def update(self, i):
        self.current_index += i

    @property
    def current_body(self):
        return self.bodies[self.current_index]

    def next_body_batch(self, batch_size=20):
        i = self.current_index
        return self.bodies[i:i+batch_size]

    def prev_body_batch(self, batch_size=20):
        i = self.current_index
        return self.bodies[i-batch_size:i]

    @property
    def current_annotation(self):
        return self.current_body.annotations

    def annotate_bodies(self, bodies):
        self.bodies = bodies
        
        if self.shuffle:
            random.shuffle(self.bodies)
        
        self.annotated_bodies = count_annotated_bodies(self.bodies, self.options)
        self.current_index = self.annotated_bodies
        
        sort_func = functools.partial(isBodyCompletedAnnotated, self.options)
        self.bodies = sorted(self.bodies, key=lambda x: not sort_func(x))

        self.preload_bodies(command="next")
        self.preload_bodies(command="prev")
        
        self.label_options, self.label_captions = self.split_options(self.options)
        
        self.count_label = HTML()
        self.set_status_text()
        display(self.count_label)

        if type(self.options) == dict:
            task_type = 'classification'
        else:
            raise ValueError('Invalid options. Must be classification dictionary.')


        self.options_buttons = self.plot_options_buttons(self.label_options)

        self.label_textareas = self.plot_textareas(self.label_captions) 
            
        self.actions_buttons = self.plot_action_buttons()

        self.out = Output()
        display(self.out)

        self.show()

        return

    def split_options(self, options):
        """
        Split options in buttons and Text Areas
        """
        label_options = dict() # Buttons

        label_captions = dict() # Text Areas

        for key, opt in options.items():
            if isinstance(opt, list):
                label_options[key] = opt
            if isinstance(opt, str):
                label_captions[key] = opt
        return label_options, label_captions

    def plot_options_buttons(self, label_options):
        options_buttons = defaultdict(dict)
        for key, values in self.label_options.items():
            for label in values:
                btn = MultiLabelButton(description=label, task_name=key)
                def on_click(label, task_name, btn):
                    # if button has a color, clear it! if not, give it a color!
                    if btn.style.button_color is self.DEACTIVE:
                        self.clear_task_button(task_name)
                        btn.style.button_color = self.ACTIVE
                    else:
                        btn.style.button_color = self.DEACTIVE

                btn.button.on_click(functools.partial(on_click, label, btn.task_name))
                options_buttons[key][label] = btn.button

            box = HBox([Label(str(key),layout=Layout(width='75px')),*options_buttons[key].values()])
            display(box)
        return options_buttons

    def plot_textareas(self, label_captions):
        label_tas = dict() 
        for key, values in label_captions.items():
            label_tas[key] = Text()
            box = HBox([Label(str(key),layout=Layout(width='75px')),label_tas[key]])
            display(box)
        return label_tas

    def plot_action_buttons(self):
        actions_buttons = list()

        btn = Button(description='record')
        btn.on_click(self.record)
        actions_buttons.append(btn)

        btn = Button(description='back')
        btn.on_click(self.back)
        actions_buttons.append(btn)

        btn = Button(description='clear current')
        btn.on_click(self.clear_annotation)
        actions_buttons.append(btn)

        btn = Button(description='skip')
        btn.on_click(self.skip)
        actions_buttons.append(btn)
        
        btn = Button(description='save')
        btn.on_click(self.save)
        actions_buttons.append(btn)

        box = HBox(actions_buttons)
        display(box)

        return actions_buttons


    def set_status_text(self, command = None):
        if (self.current_index < len(self.bodies) and command == None):
            self.count_label.value = (
                '{} examples annotated, {} examples left. Current Index {}.<br>Asset name: <code>{}</code><hr>'.format(
                    self.annotated_bodies, len(self.bodies) - self.annotated_bodies, self.current_index, body_info_str(self.current_body)
                )
            )
        elif command == "saving file":
            self.count_label.value = 'Saving... file at {}.'.format(self.output_file)
        else:
            self.count_label.value = '{} examples annotated<br>Annotation done.<hr>'.format(self.annotated_bodies)


    def collect_annotations(self):
        ann = dict()
        for task, task_buttons in self.options_buttons.items():
            for option, button in task_buttons.items():
                if button.style.button_color == self.ACTIVE:
                    ann[task] = option
        for key, ta in self.label_textareas.items():
            ann[key] = ta.value
        return ann

    def record(self, btn):
        user_annotations = self.collect_annotations()
        if isAnnotationComplete(user_annotations, self.options):
            record_annotation(self.current_body, user_annotations)
            self.annotated_bodies += 1
            self.show_next()
        else:
            print ('Havent annotated all classes, please try again')
            return 0

    def show(self, default=True):
        self.clear_colors()
        self.set_status_text()
        with self.out:
            clear_output(wait=True)
            if isBodyAnnotated(self.current_body):
                self.display_body_annotations()
            elif default:
                self.default_buttons()
            self.display_fn(self.current_body)
    
    def show_next(self):
        if self.current_index + 1 >= len(self.bodies):
            print('cannot go next')
            return
        self.current_index += 1
        self.show()
        self.preload_bodies(command="next")

    def go_back(self):
        if self.current_index - 1 < 0:
            print('cannot go back')
            return
        self.current_index -= 1
        self.show(default=False)
        self.preload_bodies(command="prev")

    #Sets the values of the annotation to the ones defined as the default ones
    #And then colors the buttons that have those values
    def default_buttons(self):
        for task, option in self.defaults.items():
            if task in self.options_buttons:
                self.options_buttons[task][option].style.button_color = self.ACTIVE
            elif task in self.label_textareas:
                self.label_textareas[task].value = option
            
            

    #Displays the current annotations with the buttons
    def display_body_annotations(self):
        
        body_annotations = self.current_body.annotations
        
        for key, value in body_annotations.items():
            if key in self.options_buttons:
                self.options_buttons[key][value].style.button_color = self.ACTIVE
            elif key in self.label_textareas:
                self.label_textareas[key].value = value
            
            
    def save(self, btn):
        self.deactivate_buttons()
        
        if self.output_file is None:
            filename, ext = os.path.splitext(self.video.video_name)
            self.output_file = "annotated_" + filename + ".json"
        
        self.set_status_text(command="saving file")
        print("Saving video at {}".format(self.output_file))
        self.video.save(self.output_file)
        print("Video saved.")
        self.activate_buttons()
        self.set_status_text()

          
    def skip(self, btn):
        self.show_next()

    def go_next(self, btn):
        self.show_next()

    def back(self, btn):
        self.go_back()

    def clear_annotation(self, btn):
        self.clear_colors()
        if isBodyCompletedAnnotated(self.options, self.current_body):
            self.annotated_bodies -= 1
        self.current_body._annotations = dict()
        self.set_status_text()

        
    def clear_colors(self):
        for task_buttons in self.options_buttons.values():
            for button in task_buttons.values():
                button.style.button_color = None
        for key, ta in self.label_textareas.items():
            ta.value = ""
                
    def deactivate_buttons(self):
        for task_buttons in self.options_buttons.values():
            for button in task_buttons.values():
                button.disabled = True
        for button in self.actions_buttons:
            button.disabled = True
            
        for key, ta in self.label_textareas.items():
            ta.disabled = True
            
    def activate_buttons(self):
        for task_buttons in self.options_buttons.values():
            for button in task_buttons.values():
                button.disabled = False
        for button in self.actions_buttons:
            button.disabled = False
        for key, ta in self.label_textareas.items():
            ta.disabled = False
            
    def clear_task_button(self, task_name):
        for button in self.options_buttons[task_name].values():
            button.style.button_color = None

    def preload_bodies(self, command="next"):
        """
        preload images in the lru_cache
        """
        if command == "next":
            bodies = self.next_body_batch()
        elif command == "prev":
            bodies = self.prev_body_batch()
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(preload_bodies, bodies)
