from plotbee.video import Video
from plotbee.frame import Frame
from plotbee.body import Body
from tqdm import tqdm
from plotbee.utils import read_json, save_json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage import io



def parse_name(name):
    parse = dict()
    parse["colony"] = int(name[1:3])
    parse["year"] = 2000 + int(name[4:6])
    parse["month"] = int(name[6:8])
    parse["day"] = int(name[8:10])
    parse["hour"] = int(name[10:12])
    return parse


def hour_histogram(df, figsize=(10, 20), ax=None):
    adf = df.copy()
    adf.loc[:, "c"] = 1
    table = pd.pivot_table(adf, "c", "id", "hour", aggfunc=np.sum).fillna(0)
    btable = table.clip(10, 100)
    mask = (table == 0)
    if ax is None:
        plt.figure(figsize=figsize)
        sns.heatmap(btable, mask=mask, linewidths=.5, cbar=True)
        plt.suptitle("Occurences of Tag ID by Hour", fontsize=30)
        plt.title("White means no occurences", fontsize=15)
        plt.xlabel("Hour", fontsize=20)
        plt.ylabel("Tag ID", fontsize=20)
        plt.xticks(fontsize=14);
        plt.yticks(fontsize=14,  rotation='horizontal');
    else:
        a = sns.heatmap(btable, mask=mask, linewidths=.5, cbar=True, ax=ax)
#         plt.suptitle("Occurences of Tag ID by Hour", fontsize=30)
        a.set_title("White means no occurences", fontsize=15)
        a.set_xlabel("Hour", fontsize=20)
        a.set_ylabel("Tag ID", fontsize=20)
        

def days_histogram(df, figsize=(10, 20), ax=None):
    
    adf = df.copy()
    adf.loc[:, "c"] = 1
    table = pd.pivot_table(adf, "c", "id", "day", aggfunc=np.sum).fillna(0)
    btable = table.clip(10, 100)
    mask = (table == 0)
    if ax is None:
        plt.figure(figsize=figsize)
        sns.heatmap(btable, linewidths=.5, cbar=True, vmax=100.0,  mask=mask)
        plt.suptitle("Occurences (clipped at 100) of Tag ID by Day", fontsize=30)
        plt.title("White means no occurences", fontsize=15)
        plt.xlabel("Day", fontsize=20)
        plt.ylabel("Tag ID", fontsize=20)
        plt.xticks(fontsize=14);
        plt.yticks(fontsize=14,  rotation='horizontal');
    else:
        
        a = sns.heatmap(btable, linewidths=.5, cbar=True, vmax=100.0,  mask=mask, ax=ax)
    #     plt.suptitle("Occurences (clipped at 100) of Tag ID by Day", fontsize=30)
        a.set_title("White means no occurences", fontsize=15)
        a.set_xlabel("Day", fontsize=20)
        a.set_ylabel("Tag ID", fontsize=20)
    #     ax.set_xticks(fontsize=14);
    #     ax.set_yticks(fontsize=14,  rotation='horizontal');

class TagVideoCollection():


    @classmethod
    def load(cls, video_folder, json_path):
        tagged_videos =  read_json(json_path)

        video_collection = dict()

        for video_name, bodies in tagged_videos.items():
            # print("Loading annotation from {} ...".format(video_name))
            frames = dict()
            for body in bodies:
                frame_id = body["frameid"]

                if  frame_id not in frames:
                    new_frame = Frame([], frame_id)
                    frames[frame_id] = new_frame

                b = Body.load_body(body, frames[frame_id])
                frames[frame_id].update([b])


            config = {
                "VIDEO_PATH": os.path.join(video_folder, video_name + ".mp4")
            }

            video = Video(list(frames.values()), [], config)
            video_collection[video_name] = video

        return cls(video_collection)
            

    def __init__(self, video_collection):
        self.video_collection = video_collection

    def __repr__(self):
        return repr(list(self.video_collection.keys()))

    def __getitem__(self, key):
        return self.video_collection[key]

    def DataFrame(self):
        instances = list()
        for video_name, video in self.video_collection.items():
            parsed = parse_name(video_name)
            for frame in video:
                for body in frame:
                    body_dict = dict()
                    body_dict["id"] = body.tag_id
                    body_dict["frame"] = body.frameid
                    body_dict["video_name"] = video_name
                    body_dict.update(parsed)
                    instances.append(body_dict)

        return pd.DataFrame(instances).sort_values(["year", "month", "day", "hour"])
    
    def json(self):
        collection_json = dict()
        for video_name, video in self.video_collection.items():
            collection_json[video_name] = video.json()
        return collection_json
    
    def save(self, path):
        collection_json = self.json()
        save_json(path, collection_json)
            

    def bodies(self):
        bodies = list()

        for video_name, video in self.video_collection.items():
            for frame in video:
                for body in frame:
                    bodies.append(body)
        return bodies


    def get_body(self, i):
        bodies = list()

        for video_name, video in self.video_collection.items():
            for frame in video:
                for body in frame:
                    if body.tag_id == i:
                        bodies.append(body)
        return bodies


    def get_occurence(self, video_name, frame_id, tag_id):
        video = self.video_collection[video_name]
        for frame in video:
            if frame.id == frame_id:
                for body in frame:
                    if body.tag_id == tag_id:
                        return body
        return None

    def filter(self, func):
        

        video_collection = dict()

        for video_name, video in self.video_collection.items():
            video_collection[video_name] = dict()
            for frame in video:
                for body in frame:
                    if func(body):
                        if frame.id not in video_collection[video_name]:
                            video_collection[video_name][frame.id] = Frame([], frame.id)
                        video_collection[video_name][frame.id].update([body])

        out_collection = dict()

        for vname in video_collection:
            config = self.video_collection[vname].config 
            frames = video_collection[vname].values()
            if len(frames) == 0:
                continue
            out_collection[vname] = Video(list(frames), [], config)
        
        return TagVideoCollection(out_collection)

    def select(self, df):
        ndf = df.sort_values(["video_name", "frame", "id"])

        video_collection = dict()

        for vname, vdf in ndf.groupby("video_name"):
            video = self.video_collection[vname]
            video_collection[vname] = dict()
            frame_indx = 0
            for frameid, fdf in vdf.groupby("frame"):
                video_collection[vname][frameid] = Frame([], frameid)
                bodies = list()
                tag_ids = fdf.id.values
                frame = video[frame_indx]
                while frame.id != frameid:
                    frame_indx += 1
                    frame = video[frame_indx]

                for body in frame:
                    if body.tag_id in tag_ids:
                        bodies.append(body)
                video_collection[vname][frameid].update(bodies)
                
        out_collection = dict()

        for vname in video_collection:
            config = self.video_collection[vname].config 
            frames = video_collection[vname].values()
            out_collection[vname] = Video(list(frames), [], config)
        
        return TagVideoCollection(out_collection)

    def export(self, folder):

        os.makedirs(folder, exist_ok=True)
        body_folder = os.path.join(folder, "bodies")
        tag_folder = os.path.join(folder, "tags")
        json_fname = os.path.join(folder, "tags.json")
        self.save(json_fname)
        for video_name, video in self.video_collection.items():
            for frame in video:
                for body in frame:
                    body_path = os.path.join(body_folder, str(body.tag_id))
                    tag_path = os.path.join(tag_folder, str(body.tag_id))
                    if not os.path.exists(body_path):
                        os.makedirs(body_path)
                        os.makedirs(tag_path)

                    # path = os.path.join(path, video_name)
                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                    
                    body_path = os.path.join(body_path, video_name + "_" + str(frame.id) + ".jpg")
                    tag_path = os.path.join(tag_path, video_name + "_" + str(frame.id) + ".jpg")
                    body.save(body_path)
                    io.imsave(tag_path, body.tag_image())





            




