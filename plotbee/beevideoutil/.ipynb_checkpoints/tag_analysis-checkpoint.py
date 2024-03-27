import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def loadtagsdf(item, tagsroot):
    file = os.path.join(tagsroot,item.tagsfile)
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def tags_stats(tags):
    annotated_frames = len(tags['data'])
    data = tags['data']
    total_count=0
    tags_counts={}
    for frame in data:
        tags1=data[frame]['tags']
        total_count+=len(tags1)
        for tag in tags1:
            id0=tag['id']
            tags_counts[id0] = tags_counts.get(id0,0)+1
    return annotated_frames,total_count,tags_counts

def tags_frame_id(tags):
    frames = []
    ids =  []
    data = tags['data']
    for frame in data:
        tags1=data[frame]['tags']
        for tag in tags1:
            id0=int(tag['id'])
            if (id0==-1): continue
            frames.append(int(frame))
            ids.append(int(id0))
    return np.array(frames),np.array(ids)

def tags_frame_id_x_y(tags):
    frames = []
    ids =  []
    x =  []
    y =  []
    data = tags['data']
    for frame in data:
        tags1=data[frame]['tags']
        for tag in tags1:
            #print(tag)
            id0=int(tag['id'])
            if (id0==-1): continue
            frames.append(int(frame))
            ids.append(int(id0))
            x.append(tag['c'][0])
            y.append(tag['c'][1])
    return np.array(frames),np.array(ids),np.array(x),np.array(y)

def tags_frame_id_df(tags):
    x,y=tags_frame_id(tags)
    df=pd.DataFrame(dict(frame=x,id=y))
    return df
def tags_frame_id_x_y_df(tags):
    f,i,x,y=tags_frame_id_x_y(tags)
    df=pd.DataFrame(dict(frame=f,id=i,x=x,y=y))
    return df

def load_tags_df(df, tagsroot, verbose=False):
    ttdf = pd.DataFrame()
    with tqdm(total=df.shape[0]) as pbar:
        for key,item in df.iterrows():
            if (verbose):
                print(item.tagsfile)
            tags = loadtagsdf(item, tagsroot)
            tdf = tags_frame_id_x_y_df(tags)
            tdf['hh']=item.hh
            tdf['DD']=item.DD
            tdf['daystamp']=f"{item.YY:02}{item.MM:02}{item.DD:02}"
            tdf['col']=item.newcol
            tdf['videokey']=key
            ttdf=ttdf.append(tdf)
            pbar.update()
    return ttdf

