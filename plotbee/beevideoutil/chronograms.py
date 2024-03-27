from . import main as vu
# Displays
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def plottaggingdf(taggingdf, cols):
    df=taggingdf[taggingdf.newcol.isin(cols)]#[['newcol','timestamp','nvrseq','frames','mm','ss','fps','realfps']]
    newcol = df.newcol
    
    plt.plot(df.datetime.to_list(), newcol.to_list(), 'vc', label='Tagging')

    plt.legend()
    plt.title('AVI files: date vs status, newcol={}'.format( cols ))
    vu.xaxis_datetime_setup(plt.gca());
    dloc = vu.DayLocator()
    plt.gca().xaxis.set_minor_locator(dloc)
    #plt.gca().yaxis.set_ticks([0,200,400,600,800,1000,1200])
    plt.grid(True,'both','x')
    #plt.xlim(datetime.datetime(2019,6,20),datetime.datetime(2019,8,25))
    #plt.xlim(datetime.datetime(2019,7,4),datetime.datetime(2019,7,13))
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
def plotstatuscols(afidf, cols):
    df=avidf[avidf.newcol.isin(cols)]#[['newcol','timestamp','nvrseq','frames','mm','ss','fps','realfps']]
    hasdup=df.duplicated(subset=['newcol','timestamp'],keep=False)
    is1h=(df.frames/df.realfps > 3600-20)   # miss less than 10s at the end. more than 01:59:50 at 20fps
    isaligned=(df.mm==0)&(df.ss==0)
    weirdfps=(df.realfps!=20)
    newcol=df.newcol

    isnormal = isaligned & is1h & ~weirdfps
    z=pd.Series(np.zeros((df.shape[0],)),index=df.index,name='zero')
    plt.plot(vu.timestamp_column_to_datetime(df[isnormal].timestamp).to_list(), newcol[isnormal]+0, '.g', label='Normal')
    plt.plot(vu.timestamp_column_to_datetime(df[~isaligned].timestamp).to_list(), newcol[~isaligned]+0, '.r', label='Start not aligned')
    plt.plot(vu.timestamp_column_to_datetime(df[~is1h].timestamp).to_list(), newcol[~is1h]+0, '.m', label='Not 1h long')
    plt.plot(vu.timestamp_column_to_datetime(df[hasdup].timestamp).to_list(), newcol[hasdup]-0.1, '.b', label='Has duplicate')
    plt.plot(vu.timestamp_column_to_datetime(df[weirdfps].timestamp).to_list(), newcol[weirdfps]+0, '.', mec=(0.25,0.75,0), label='Not 20 realfps')
    plt.plot(vu.timestamp_column_to_datetime(df[df.trimmed].timestamp).to_list(), newcol[df.trimmed]+0.4, '.', mec=(1,0.8,0), label='Trimmed end(s)')
    plt.plot(vu.timestamp_column_to_datetime(df[df.corrupted].timestamp).to_list(), newcol[df.corrupted]+0.35, '.', mec=(1,0.25,0), label='Corrupted content')
    plt.legend()
    plt.title('AVI files: date vs status, newcol={}'.format( cols ))
    vu.xaxis_datetime_setup(plt.gca());
    dloc = vu.DayLocator()
    plt.gca().xaxis.set_minor_locator(dloc)
    #plt.gca().yaxis.set_ticks([0,200,400,600,800,1000,1200])
    plt.grid(True,'both','x')
    #plt.xlim(datetime.datetime(2019,6,20),datetime.datetime(2019,8,25))
    #plt.xlim(datetime.datetime(2019,7,4),datetime.datetime(2019,7,13))
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
def plotstatuscols1h(avidf, cols, taggingdf=None):
    df=avidf[avidf.newcol.isin(cols)]#[['newcol','timestamp','nvrseq','frames','mm','ss','fps','realfps']]
    isaligned=(df.mm==0)&(df.ss==0)
    is1h=(df.frames/df.realfps > 3600-20)   # miss less than 10s at the end. more than 01:59:50 at 20fps
    
    df2 = df[~(isaligned & is1h)].copy()
    df=df[isaligned & is1h]
    
    hasdup=df.duplicated(subset=['newcol','timestamp'],keep=False)
    weirdfps=(df.realfps!=20)
    newcol=df.newcol
    
    trimmedstart=(~df.hasframe0)&(df.hasframeN)
    trimmedend=(df.hasframe0)&(~df.hasframeN)
    trimmedboth=(~df.hasframe0)&(~df.hasframeN)

    isnormal = (~weirdfps)&(~df.trimmed)&(~df.corrupted)
    z=pd.Series(np.zeros((df.shape[0],)),index=df.index,name='zero')
    plt.plot(vu.timestamp_column_to_datetime(df[isnormal].timestamp).to_list(), newcol[isnormal]+0, '.g', label='Normal')
    plt.plot(vu.timestamp_column_to_datetime(df[hasdup].timestamp).to_list(), newcol[hasdup]-0.1, '.b', label='Has duplicate')
    plt.plot(vu.timestamp_column_to_datetime(df[weirdfps].timestamp).to_list(), newcol[weirdfps]+0, '.', mec=(0.25,0.75,0), label='Not 20 realfps')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedstart].timestamp).to_list(), newcol[trimmedstart]+0.2, '.', mec=(0.7,0.7,0), label='Trimmed start')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedend].timestamp).to_list(), newcol[trimmedend]+0.2, '.', mec=(0.7,1,0), label='Trimmed end')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedboth].timestamp).to_list(), newcol[trimmedboth]+0.2, '.', mec=(0.7,0.5,0), label='Trimmed start & end')
    plt.plot(vu.timestamp_column_to_datetime(df[df.corrupted].timestamp).to_list(), newcol[df.corrupted]+0.3, '.', mec=(1,0.1,0), label='Corrupted content')
    
    plt.plot(vu.timestamp_column_to_datetime(df2.timestamp).to_list(), df2.newcol+0.1, '.', mec=(0.25,0.5,0.5), label='Unaligned and short videos')
    
    if (taggingdf is not None):
        df=taggingdf[taggingdf.newcol.isin(cols)]
        newcol = df.newcol
    
        plt.plot(df.datetime.to_list(), newcol+0.4, 'vc', label='Tagging')
    
    plt.legend()
    plt.title('AVI files: date vs status, newcol={}'.format( cols ))
    vu.xaxis_datetime_setup(plt.gca());
    dloc = vu.DayLocator()
    plt.gca().xaxis.set_minor_locator(dloc)
    #plt.gca().yaxis.set_ticks([0,200,400,600,800,1000,1200])
    plt.grid(True,'both','x')
    #plt.xlim(datetime.datetime(2019,6,20),datetime.datetime(2019,8,25))
    #plt.xlim(datetime.datetime(2019,7,4),datetime.datetime(2019,7,13))
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
def plotstatus_mp4_1h(mp4df, cols=None, taggingdf=None):
    df = mp4df
    if (cols is None):
        cols = np.sort(df.newcol.unique())
        df = df.newcol.isin(cols)

    isaligned=(df.mm==0)&(df.ss==0)
    is1h=(df.frames/df.realfps > 3600-20)   # miss less than 10s at the end. more than 01:59:50 at 20fps
    
    df2 = df[~(isaligned & is1h)].copy()
    df=df[isaligned & is1h]
    
    hasdup=df.duplicated(subset=['newcol','timestamp'],keep=False)
    weirdfps=(df.realfps!=20)
    newcol=df.newcol
    
    trimmedstart=(~df.hasframe0)&(df.hasframeN)
    trimmedend=(df.hasframe0)&(~df.hasframeN)
    trimmedboth=(~df.hasframe0)&(~df.hasframeN)

    exist = df['mp4exist']
    
    isnormal = (~weirdfps)&(~df.trimmed)&(~df.corrupted)&(exist)
    
    ismissing = ~exist
    
    z=pd.Series(np.zeros((df.shape[0],)),index=df.index,name='zero')
    plt.plot(vu.timestamp_column_to_datetime(df[isnormal].timestamp).to_list(), newcol[isnormal]+0, '.g', label='Normal')
    plt.plot(vu.timestamp_column_to_datetime(df[ismissing].timestamp).to_list(), newcol[ismissing]+0, '.', mec=[0.7,0.7,0.7], label='Missing')
    plt.plot(vu.timestamp_column_to_datetime(df[hasdup].timestamp).to_list(), newcol[hasdup]-0.1, '.b', label='Has duplicate')
    plt.plot(vu.timestamp_column_to_datetime(df[weirdfps].timestamp).to_list(), newcol[weirdfps]+0, '.', mec=(0.25,0.75,0), label='Not 20 realfps')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedstart].timestamp).to_list(), newcol[trimmedstart]+0.2, '.', mec=(0.7,0.7,0), label='Trimmed start')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedend].timestamp).to_list(), newcol[trimmedend]+0.2, '.', mec=(0.7,1,0), label='Trimmed end')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedboth].timestamp).to_list(), newcol[trimmedboth]+0.2, '.', mec=(0.7,0.5,0), label='Trimmed start & end')
    plt.plot(vu.timestamp_column_to_datetime(df[df.corrupted].timestamp).to_list(), newcol[df.corrupted]+0.3, '.', mec=(1,0.1,0), label='Corrupted content')
    
    plt.plot(vu.timestamp_column_to_datetime(df2.timestamp).to_list(), df2.newcol+0.1, '.', mec=(0.25,0.5,0.5), label='Unaligned and short videos')
    
    if (taggingdf is not None):
        df=taggingdf[taggingdf.newcol.isin(cols)]
        newcol = df.newcol
    
        plt.plot(df.datetime.to_list(), newcol+0.4, 'vc', label='Tagging')
    
    plt.legend()
    plt.title('MP4 files: date vs status, newcol={}'.format( cols ))
    vu.xaxis_datetime_setup(plt.gca());
    dloc = vu.DayLocator()
    plt.gca().xaxis.set_minor_locator(dloc)
    #plt.gca().yaxis.set_ticks([0,200,400,600,800,1000,1200])
    plt.grid(True,'both','x')
    #plt.xlim(datetime.datetime(2019,6,20),datetime.datetime(2019,8,25))
    #plt.xlim(datetime.datetime(2019,7,4),datetime.datetime(2019,7,13))
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
def plotstatus_tags_1h(tagsdf, cols=None, taggingdf=None):
    df = tagsdf
    if (cols is None):
        cols = np.sort(df.newcol.unique())
        df = df.newcol.isin(cols)

    isaligned=(df.mm==0)&(df.ss==0)
    is1h=(df.frames/df.realfps > 3600-20)   # miss less than 10s at the end. more than 01:59:50 at 20fps
    
    df2 = df[~(isaligned & is1h)].copy()
    df=df[isaligned & is1h]
    
    hasdup=df.duplicated(subset=['newcol','timestamp'],keep=False)
    weirdfps=(df.realfps!=20)
    newcol=df.newcol
    
    trimmedstart=(~df.hasframe0)&(df.hasframeN)
    trimmedend=(df.hasframe0)&(~df.hasframeN)
    trimmedboth=(~df.hasframe0)&(~df.hasframeN)

    exist = df['tagsexist']
    
    isnormal = (~weirdfps)&(~df.trimmed)&(~df.corrupted)&(exist)
    
    ismissing = ~exist
    
    z=pd.Series(np.zeros((df.shape[0],)),index=df.index,name='zero')
    plt.plot(vu.timestamp_column_to_datetime(df[isnormal].timestamp).to_list(), newcol[isnormal]+0, '.g', label='Normal')
    plt.plot(vu.timestamp_column_to_datetime(df[ismissing].timestamp).to_list(), newcol[ismissing]+0, '.', mec=[0.9,0.9,0.9], label='Missing')
    plt.plot(vu.timestamp_column_to_datetime(df[hasdup].timestamp).to_list(), newcol[hasdup]-0.1, '.b', label='Has duplicate')
    plt.plot(vu.timestamp_column_to_datetime(df[weirdfps].timestamp).to_list(), newcol[weirdfps]+0, '.', mec=(0.25,0.75,0), label='Not 20 realfps')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedstart].timestamp).to_list(), newcol[trimmedstart]+0.2, '.', mec=(0.7,0.7,0), label='Trimmed start')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedend].timestamp).to_list(), newcol[trimmedend]+0.2, '.', mec=(0.7,1,0), label='Trimmed end')
    plt.plot(vu.timestamp_column_to_datetime(df[trimmedboth].timestamp).to_list(), newcol[trimmedboth]+0.2, '.', mec=(0.7,0.5,0), label='Trimmed start & end')
    plt.plot(vu.timestamp_column_to_datetime(df[df.corrupted].timestamp).to_list(), newcol[df.corrupted]+0.3, '.', mec=(1,0.1,0), label='Corrupted content')
    
    plt.plot(vu.timestamp_column_to_datetime(df2.timestamp).to_list(), df2.newcol+0.1, '.', mec=(0.25,0.5,0.5), label='Unaligned and short videos')
    
    if (taggingdf is not None):
        df=taggingdf[taggingdf.newcol.isin(cols)]
        newcol = df.newcol
    
        plt.plot(df.datetime.to_list(), newcol+0.4, 'vc', label='Tagging')
    
    plt.legend()
    plt.title('Tags files: date vs status, newcol={}'.format( cols ))
    vu.xaxis_datetime_setup(plt.gca());
    dloc = vu.DayLocator()
    plt.gca().xaxis.set_minor_locator(dloc)
    #plt.gca().yaxis.set_ticks([0,200,400,600,800,1000,1200])
    plt.grid(True,'both','x')
    #plt.xlim(datetime.datetime(2019,6,20),datetime.datetime(2019,8,25))
    #plt.xlim(datetime.datetime(2019,7,4),datetime.datetime(2019,7,13))
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
def plotstatus_1h_avi_mp4_tags(tagsdf, cols=None, taggingdf=None):
    df = tagsdf
    if (cols is None):
        cols = np.sort(df.newcol.unique())
        df = df.newcol.isin(cols)

    isaligned=(df.mm==0)&(df.ss==0)
    is1h=(df.frames/df.realfps > 3600-20)   # miss less than 10s at the end. more than 01:59:50 at 20fps
    
    #df2 = df[~(isaligned & is1h)].copy()
    df=df[isaligned & is1h]
    
    newcol=df.newcol
    
    trimmedstart=(~df.hasframe0)&(df.hasframeN)
    trimmedend=(df.hasframe0)&(~df.hasframeN)
    trimmedboth=(~df.hasframe0)&(~df.hasframeN)

    isnormal = (~df.trimmed)&(~df.corrupted)
    
    aviexist = df['aviexist']
    mp4exist = df['mp4exist']
    tagsexist = df['tagsexist']
    
    avinormal = isnormal & aviexist
    mp4normal = isnormal & mp4exist
    tagsnormal = isnormal & tagsexist
        
    z=pd.Series(np.zeros((df.shape[0],)),index=df.index,name='zero')
    plt.plot(vu.timestamp_column_to_datetime(df[avinormal].timestamp).to_list(), newcol[avinormal]+0, '.', mec=[0.5,0.5,0.5], label='Has avi')
    plt.plot(vu.timestamp_column_to_datetime(df[mp4normal].timestamp).to_list(), newcol[mp4normal]+0.2, '.b', label='Has mp4')
    plt.plot(vu.timestamp_column_to_datetime(df[tagsnormal].timestamp).to_list(), newcol[tagsnormal]+0.4, '.r', label='Has tags')
    plt.plot(vu.timestamp_column_to_datetime(df[~isnormal].timestamp).to_list(), newcol[~isnormal]-0.1, '.', mec=(0.7,0.7,0.7), label='Trimmed or corrupted')
    
    if (taggingdf is not None):
        df=taggingdf[taggingdf.newcol.isin(cols)]
        newcol = df.newcol
    
        plt.plot(df.datetime.to_list(), newcol+0.4, 'vc', label='Tagging')
    
    plt.legend()
    plt.title('1h aligned videos: date vs status, newcol={}'.format( cols ))
    vu.xaxis_datetime_setup(plt.gca());
    dloc = vu.DayLocator()
    plt.gca().xaxis.set_minor_locator(dloc)
    #plt.gca().yaxis.set_ticks([0,200,400,600,800,1000,1200])
    plt.grid(True,'both','x')
    #plt.xlim(datetime.datetime(2019,6,20),datetime.datetime(2019,8,25))
    #plt.xlim(datetime.datetime(2019,7,4),datetime.datetime(2019,7,13))
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()