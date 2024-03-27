from . import main as vu
# Displays
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cv2

from os.path import join

from IPython.display import display
from PIL import Image

def getFrame(fullfile,frame=0,fps=20,cap=None):
    if (cap is not None):
        cap1=cap
    else:
        cap1=cv2.VideoCapture(fullfile)
    ts=frame/fps*1000
    #print(ts/1000)
    cap1.set(cv2.CAP_PROP_POS_MSEC, ts);
    ret, img = cap1.read()
    if (cap is None):
        cap1.release()
    if img is None: 
        return None
    img = np.ascontiguousarray(img[...,::-1], dtype=np.uint8)
    return img

def get_avi_frame(item, frame=0, aviroot=None):
    return getFrame(join(aviroot,item.avifile),frame=frame,fps=item.fps,cap=None)
def get_mp4_frame(item, frame=0, mp4root=None):
    return getFrame(join(mp4root,item.mp4file),frame=frame,fps=item.fps,cap=None)


from IPython.display import display

def addFrameMetainfo(img, vidname=None, f=None, s=0.25):
    I=cv2.resize(img, None, fx=s, fy=s)
    I=cv2.copyMakeBorder(I,0,40,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    h,w=I.shape[0:2]
    fs=0.5
    fs2=0.5
    #frame=cv2.putText(frame,"Col {}/{}".format(col,vidname).format(col),(50,310),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),5)
    if (vidname is not None):
        frame=cv2.putText(I,"{}".format(vidname),(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),1)
    if (f is not None):
        frame=cv2.putText(I,"F{}".format(f),(w-60,h-10),cv2.FONT_HERSHEY_SIMPLEX,fs2,(0,0,0),1)
    return I
def get_frame_plus_metainfo(fullfile,frame=0,fps=20,cap=None, vidname=None, scale=0.25):
    img=getFrame(fullfile,frame=frame,fps=fps,cap=cap)
    if (img is None): return None
    return addFrameMetainfo(img, vidname, frame, s=scale)
def display_frame_plus_metainfo(fullfile,frame=0,fps=20,cap=None, vidname=None, scale=0.25):
    I=get_frame_plus_metainfo(fullfile, frame, fps, cap, vidname=vidname, scale=scale)
    display(Image.fromarray(I))

def framedisplay(imgs, scale=1.0):
    def scal(img,s):
        return cv2.resize(img, None, fx=s, fy=s)
    if (isinstance(imgs,list)):
        #display(*[Image.fromarray(I) for I in imgs])
        I=np.concatenate([Image.fromarray(scal(I,scale)) for I in imgs], axis=1)
        display(Image.fromarray(I))
    else:
        I=imgs
        display(Image.fromarray(scal(I,scale)))
        
        
def plot1Frame(df, videos, videoid, frame, debug=False, focus=False, focus_center=None):

    vidfile=videos[videoid]
    if (debug):
        print(vidfile)
    I=getFrame(vidfile,frame)
    
    df0=df[(df.videoid==videoid)&(df.frame==frame)]
    
    if (debug):
        print(df0['trackid,frame,detid,cx,cy,vx,vy,predictx,predicty'.split(',')])

    if (focus):
        xmin=focus_center[0]-200
        xmax=focus_center[0]+200
        ymin=focus_center[1]-200
        ymax=focus_center[1]+200
        
    plt.imshow(I)
    plt.plot(df0.cx,df0.cy, 'ro', fillstyle='none')
    ax=plt.gca()
    for k, item in df0.iterrows():
        if (focus and ((item.cx<xmin) or (item.cx>xmax) or (item.cy<ymin) or (item.cy>ymax))):
            continue
        if (item.virtual):
            col='b'
        else:
            col='r'
        ax.annotate(str(item.trackid), (item.cx, item.cy), xytext=(5,0), textcoords='offset points', color=col)
        plt.plot([item.cx+item.vx,item.cx],[item.cy+item.vy,item.cy], 'g-')
    dff=df0[df0.frame==df0.track_startframe]
    plt.plot(dff.cx,dff.cy, 'r+')
    dff=df0[df0.frame==df0.track_endframe]
    plt.plot(dff.cx,dff.cy, 'rx')
    
    if (focus):
        plt.xlim(xmin,xmax)
        plt.ylim(ymax,ymin)
            
    plt.title(vidfile.split('/')[-1]+f' frame {frame}')
    
import seaborn as sns
    
def plottrack(ddf, videos, trackid, frame=None):
    Tdf=ddf[ddf.trackid==trackid]
    item=Tdf.iloc[0]
    N=item.track_endframe-item.track_startframe
    print(f'N={N}')

    fig,ax=plt.subplots(1,3,figsize=(18,4), constrained_layout=True)
    ax=ax.ravel()
    W,H=2580,1480
    
    if (frame is None):
        frame=(item.track_endframe+item.track_startframe)//2
        
    s = 10*(2-Tdf.virtual)

    plt.sca(ax[0])
    plot1Frame(ddf,videos, 3, frame, debug=False)
    plt.scatter(Tdf.cx,Tdf.cy, c=Tdf.frame)
    plt.sca(ax[1])
    #plt.scatter(Tdf.frame,Tdf.cy, s=s, c=Tdf.frame, label='cy')
    sns.scatterplot(data=Tdf, x='frame',y='cy',hue='frame',style='virtual',palette='viridis',legend=False,markers=['o','x'],linewidth=1)
    plt.ylim(H,0);
    #plt.gca().set_aspect((N/H)*(H/W), adjustable='box')
    plt.sca(ax[2])
    plt.scatter(Tdf.frame, Tdf.cx, s=s, c=Tdf.frame, label='cx')
    sns.scatterplot(data=Tdf, x='frame',y='cx',hue='frame',style='virtual',palette='viridis',legend=False)
    plt.ylim(0,W)
    #plt.gca().set_aspect((W/N)*(H/W))
    #ax[1].sharey(ax[0])
    #ax[2].sharex(ax[0])