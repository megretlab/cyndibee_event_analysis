import os
import glob
import math
import sys
import pandas as pd
import csv
import functools
import time
import re
from stat import *
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import fnmatch

def timestamp12_to_24(timestamp12, ampm):
    hh=int(timestamp12[6:8])
    if (ampm=='AM'): 
        if (hh==12): hh=0    # 12AM == 00
    if (ampm=='PM'): 
        if (hh!=12): hh=hh+12  # 12PM == 12, 01PM == 13
    return timestamp12[:6]+'{:02d}'.format(hh)+timestamp12[8:]

def list_avi_files(avidir):
    files = []
    for file in os.listdir(avidir):
        if fnmatch.fnmatch(file, '*.avi'):
            files.append(file)
    return files
def parse_avi_filename(filename, path):
    obj={'filename':filename, 'name':filename[:-4]}
    
    res_raw24 = re.search(r'([0-9]+)_([0-9]+)_R_([0-9]{12}).avi', filename)
    res_raw24_suf = re.search(r'([0-9]+)_([0-9]+)_R_([0-9]{12})(.+).avi', filename)
    res_raw12 = re.search(r'([0-9]+)_([0-9]+)_R_([0-9]{12})(AM|PM).avi', filename)
    
    res_ignore = re.search(r'.*.scale4.mp4', filename)
    
    if (res_ignore is not None):
        obj['ignore']=True
        return obj

    suf=''
    if (res_raw12 is not None):
        seq, cam, timestamp, ampm = res_raw12.groups()
        timestamp=timestamp12_to_24(timestamp,ampm)
        nameformat='raw12'
    elif (res_raw24 is not None):
        seq, cam, timestamp = res_raw24.groups()
        nameformat='raw24'
    elif (res_raw24_suf is not None):
        seq, cam, timestamp, suf = res_raw24_suf.groups()
        nameformat='raw24_suf'
    else:
        obj['ignore']=True
        return obj
    
    D=timestamp[0:6]
    T=timestamp[6:12]
    newname = 'C{cam}_{timestamp}'.format(cam=cam,timestamp=timestamp)

    try:
        info = os.stat(os.path.join(path,filename))
    except:
        info = None

    obj['cam']=cam
    obj['timestamp']=timestamp
    if (info is not None):
        obj['filesize']=info.st_size
    obj['nameformat']=nameformat
    obj['newname']=newname
    obj['ignore']=False
    obj['suffix']=suf
    obj['nvrseq']=float(seq)
    
    obj.update(YY=int(timestamp[:2]),MM=int(timestamp[2:4]),DD=int(timestamp[4:6]),
               hh=int(timestamp[6:8]),mm=int(timestamp[8:10]),ss=int(timestamp[10:12]))
    #print('{} => {}'.format(filename,newname))
    
    return obj
def parse_avi_files(avifiles, avidir, only_valid=True):
    objs=[]
    for f in avifiles:
        obj=parse_avi_filename(f, avidir)
        if (only_valid and obj['ignore']):
            continue
        objs.append(obj)
    return pd.DataFrame(objs)[['filename','name','newname','cam','timestamp','YY','MM','DD','hh','mm','ss',
                                'filesize','nameformat','ignore','suffix','nvrseq']]
def parse_avi_dir(avidir, only_valid=True):
    return parse_avi_files(list_avi_files(avidir), avidir, only_valid=only_valid)

def list_mp4_files(mp4dir):
    files = []
    for file in os.listdir(mp4dir):
        if fnmatch.fnmatch(file, '*.mp4'):
            files.append(file)
    return files
def parse_mp4_filename(filename, path):
    obj={'filename':filename, 'name':filename[:-4]}
    
    res_raw24 = re.search(r'([0-9]+)_([0-9]+)_R_([0-9]+).mp4', filename)
    res_raw12 = re.search(r'([0-9]+)_([0-9]+)_R_([0-9]+)(AM|PM).mp4', filename)
    res_mp4 = re.search(r'C([0-9]+)_([0-9]+).mp4', filename)
    res_ignore = re.search(r'.*.scale4.mp4', filename)
    
    if ((res_ignore is not None) or (res_mp4 is None and res_raw24 is None and res_raw12 is None)):
        obj['ignore']=True
        return obj
    
    if (res_mp4 is not None):
        cam, timestamp = res_mp4.groups()
        nameformat='camtime' 
    elif (res_raw24 is not None):
        _, cam, timestamp = res_raw24.groups()
        nameformat='raw24'
    elif (res_raw12 is not None):
        _, cam, timestamp, ampm = res_raw12.groups()
        timestamp=timestamp12_to_24(timestamp,ampm)
        nameformat='raw12'
    else:
        raise ValueError('Internal error on filename: not cam_timestamp, nor raw name.') 
        
    D=timestamp[0:6]
    T=timestamp[6:12]
    newname = 'C{cam}_{timestamp}'.format(cam=cam,timestamp=timestamp)
    info = os.stat(os.path.join(path,filename))
    obj['cam']=cam
    obj['timestamp']=timestamp
    obj['filesize']=int(info.st_size)
    obj['nameformat']=nameformat
    obj['newname']=newname
    obj['ignore']=False
    
    obj.update(YY=int(timestamp[:2]),MM=int(timestamp[2:4]),DD=int(timestamp[4:6]),
               hh=int(timestamp[6:8]),mm=int(timestamp[8:10]),ss=int(timestamp[10:12]))
    #print('{} => {}'.format(filename,newname))
    return obj
def parse_mp4_files(mp4files, mp4dir, only_valid=True):
    objs=[]
    for f in mp4files:
        obj=parse_mp4_filename(f, mp4dir)
        if (only_valid and obj['ignore']):
            continue
        objs.append(obj)
    return pd.DataFrame(objs)[['filename','name','newname','cam','timestamp','YY','MM','DD','hh','mm','ss',
                                'filesize','nameformat','ignore']]
def parse_mp4_dir(mp4dir, only_valid=True):
    return parse_mp4_files(list_mp4_files(mp4dir), mp4dir, only_valid=only_valid)

class TagDB:
    def __init__(self,tagdir):
        self.dir=tagdir
    def list_filenames(self,pattern='Tags-*.json'):
        files = []
        for file in os.listdir(self.dir):
            if fnmatch.fnmatch(file, pattern):
                files.append(file)
        return files
    def parse_filename(self,filename):
        path=self.dir
        obj={'filename':filename}

        res_tag = re.search(r'Tags-C([0-9]+)_([0-9]+).json', filename)

        if (res_tag is not None):
            cam, timestamp = res_tag.groups()
            name='C{cam}_{timestamp}'.format(cam=cam,timestamp=timestamp)
            nameformat='camtime' 
        else:
            raise ValueError('Internal error on filename: expected Tags-Cxx_YYMMDDhhmmss.json') 

        D=timestamp[0:6]
        T=timestamp[6:12]
        newname = 'C{cam}_{timestamp}'.format(cam=cam,timestamp=timestamp)
        info = os.stat(os.path.join(path,filename))        
        obj['name']=name
        obj['cam']=cam
        obj['timestamp']=timestamp
        obj['filesize']=int(info.st_size)
        obj['nameformat']=nameformat
        obj['newname']=newname
        obj['ignore']=False

        obj.update(YY=int(timestamp[:2]),MM=int(timestamp[2:4]),DD=int(timestamp[4:6]),
                   hh=int(timestamp[6:8]),mm=int(timestamp[8:10]),ss=int(timestamp[10:12]))
        #print('{} => {}'.format(filename,newname))
        return obj
    def parse_files(self, filenames, only_valid=True):
        objs=[]
        for f in filenames:
            obj=self.parse_filename(f)
            if (only_valid and obj['ignore']):
                continue
            objs.append(obj)
        return pd.DataFrame(objs)[['filename','name','newname','cam','timestamp','YY','MM','DD','hh','mm','ss',
                                    'filesize','nameformat','ignore']]
    def parse_dir(self, only_valid=True):
        filenames=self.list_filenames()
        return self.parse_files(filenames, only_valid=only_valid)

def get_fullname(serie, dir):
    return serie.apply(lambda x: os.path.join(dir,x))

def finddups(df):
    df=df.sort_values(by=['newname','nameformat','filesize'],ascending=[True,False,True])
    #df=df.sort_values(by=['newname','filesize'],ascending=[True,True])
    dupnum = df.groupby(['newname'])['newname'].transform('count')
    dupids = df.duplicated('newname',keep=False)
    nodupsids=~dupids
    duptokeep=~df.duplicated('newname',keep='last')

    dupdf=df[['filename','newname','filesize']]
    dupdf = dupdf.assign(isdup=dupids, tokeep=nodupsids | duptokeep, dupcount=dupnum)
    return dupdf
def finddups_by(df,by=['cam','timestamp'],order=[]):
    df=df.sort_values(by=by)
    dupnum = df.groupby(by)[by].transform('count')
    dupids = df.duplicated(by,keep=False)
    nodupsids=~dupids
    duptokeep=~df.duplicated('newname',keep='last')

    dupdf=df[['filename','newname','filesize']]
    dupdf = dupdf.assign(isdup=dupids, tokeep=nodupsids | duptokeep, dupcount=dupnum)
    return dupdf

import shutil

def do_move(movdf, dryrun=True):
    for idx,item in movdf.iterrows():
        #print(item)
        f = item['old']
        newf = item['new']
        try:
            info = os.stat(f)
        except:
            print('SKIP:',f,"=>",newf,'FILE NOT FOUND')
            continue
        ## DANGER !!!
        if os.path.exists(newf):
            print('SKIP:',f,"=>",newf,'FILE EXIST')
            continue
        else:
            print('MOVE:',f,"=>",newf,'size=',info.st_size)
                
        if (not dryrun):
            shutil.move(f,newf)
            
def build_movdf(dupdf,srcdir,dupdir):
    df = dupdf[~dupdf['tokeep']]
    movdf = pd.DataFrame(data={'name':df['filename'],'tokeep':df['tokeep'],
                               'old':get_fullname(df['filename'], srcdir),
                               'new':get_fullname(df['filename'], dupdir)})
    return movdf
def build_movdf_normalizename_mp4(df,srcdir,newdir):
    movdf = pd.DataFrame(data={'oldname':df['name'],'newname':df['newname'],
                               'old':get_fullname(df['filename'], srcdir),
                               'new':get_fullname(df['newname']+'.mp4', newdir)})
    return movdf

def timestamp_column_to_YYMMDDhhmmss(ts):
    obj = {'YY': ts.apply(lambda x: int('20'+x[0:2])),
           'MM': ts.apply(lambda x: int(x[2:4])),
           'DD': ts.apply(lambda x: int(x[4:6])),
           'hh': ts.apply(lambda x: int(x[6:8])),
           'mm': ts.apply(lambda x: int(x[8:10])),
           'ss': ts.apply(lambda x: int(x[10:12]))}
    return obj

def timestamp_column_to_datetime(ts):
    dt = ts.apply(lambda x: datetime.datetime(int('20'+x[0:2]),int(x[2:4]),int(x[4:6]),int(x[6:8]),int(x[8:10]),int(x[10:12])))
    return dt

def to_datetime(s):
    # Convert hour of the day as datetime.time into datetime starting on 1900/01/01
    return s.apply(lambda x: datetime.datetime(1900,1,1,x.hour, x.minute, x.second))

def timestamp_column_to_times(ts):
    #if (ts.dtype is not 'str'):
    #    raise ValueError('ts should be Series of str')
    df=pd.DataFrame()
    ts = ts[~ts.isna()]
    df['datetime'] = ts.apply(lambda x: datetime.datetime(int('20'+x[0:2]),int(x[2:4]),int(x[4:6]),int(x[6:8]),int(x[8:10]),int(x[10:12])))
    df['date'] = ts.apply(lambda x: datetime.date(int('20'+x[0:2]),int(x[2:4]),int(x[4:6])))
    df['time'] = ts.apply(lambda x: datetime.time(int(x[6:8]),int(x[8:10]),int(x[10:12])))
    return df

def plot_timestamp_cam(df, style='.', **kargs):
    #df.sort_values(['timestamp','newname'])
    dt = timestamp_column_to_datetime(df['timestamp'])
    
    df = df[~dt.isna()]
    dt = dt[~dt.isna()].to_list()
    
    camint = df['cam'].astype(int).to_list()

    #df.plot('datetime','camint','scatter',)
    #ts = dt.to_list()
    #cam = df['camint'].to_list()
    #df.plot('datetime','camint','scatter')
    return plt.plot(dt,camint, style,**kargs)

import matplotlib.dates as dates

from matplotlib.dates import YearLocator, MonthLocator, WeekdayLocator, DayLocator, HourLocator, AutoDateLocator, DateFormatter
import matplotlib.dates as mdates


def xaxis_datetime_setup(ax):
    mloc = MonthLocator()
    dloc = WeekdayLocator(dates.MO)
    #dloc = DayLocator()
    
    # See format strings: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    mformat = DateFormatter('%b %Y')
    #mformat = DateFormatter('%Y/%m')
    #dformat = DateFormatter('%b %d\n%a')
    dformat = DateFormatter('%b %d')
    
    ax.xaxis.set_major_locator(mloc)
    ax.xaxis.set_minor_locator(dloc)
    ax.xaxis.set_major_formatter(mformat)
    ax.xaxis.set_minor_formatter(dformat)
    ax.grid(True, which='both',axis='x')
    
    plt.setp(ax.xaxis.get_minorticklabels(), ha='left', rotation=-90);
    plt.setp(ax.xaxis.get_minorticklabels(), y=0.01);
    plt.setp(ax.xaxis.get_majorticklabels(), ha='left');
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0);
    plt.setp(ax.xaxis.get_majorticklabels(), y=-0.02);
    ax.tick_params(axis='x', which='minor', length=5)
    ax.tick_params(axis='x', which='major', length=40)
    plt.xlabel('day')
    
def xaxis_day_hours_setup(ax):

    dloc = DayLocator()
    dformat = DateFormatter('%Y %h %d')
    ax.xaxis.set_major_locator(dloc)
    ax.xaxis.set_major_formatter(dformat)

    dloc = HourLocator()
    dformat = DateFormatter('%H:%M')
    ax.xaxis.set_minor_locator(dloc)
    ax.xaxis.set_minor_formatter(dformat)
    
    plt.setp(ax.xaxis.get_minorticklabels(), ha='left', rotation=-90);
    plt.setp(ax.xaxis.get_minorticklabels(), y=0.01);
    plt.setp(ax.xaxis.get_majorticklabels(), ha='left');
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0);
    plt.setp(ax.xaxis.get_majorticklabels(), y=-0.02);
    ax.tick_params(axis='x', which='minor', length=5)
    ax.tick_params(axis='x', which='major', length=40)
    plt.xlabel('time')

def yaxis_hours_setup(ax):
    hformat = DateFormatter('%H:%M')
    
    #ax.yaxis.set_major_locator(mdates.HourLocator(byhour=range(24),interval=1))
    ax.yaxis.set_major_locator(mdates.HourLocator(byhour=[0,6,12,18,24]))
    ax.yaxis.set_minor_locator(mdates.HourLocator())
    #ax.yaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=6))
    ax.yaxis.set_major_formatter(hformat)
    #plt.yticks([datetime.time(hour=h,minute=0) for h in range(0,24)]);
    #plt.yticks([datetime.datetime(1900,1,1,hour=h,minute=0) for h in range(0,24)]+[datetime.datetime(1900,1,2,hour=0,minute=0)]);
    ax.grid(True, which='both',axis='y')
    ax.grid(which='minor',axis='y', lw=0.3)

    plt.ylabel('hour of the day')
    plt.ylim(datetime.datetime(1900,1,1,0,0,0)-datetime.timedelta(minutes=30),
             datetime.datetime(1900,1,1,0,0,0)+datetime.timedelta(hours=24,minutes=30))
    
def xlim_timestamp(ax,xmin,xmax, deltamin=2, deltamax=23):
    # Expect xmin and xmax as timestamp strings
    
    def ts_to_date(x):
        if (len(x)==6): x=x+'000000'
        return datetime.datetime(int('20'+x[0:2]),int(x[2:4]),int(x[4:6]),int(x[6:8]),int(x[8:10]),int(x[10:12]))
    
    t1 = ts_to_date(xmin)
    t2 = ts_to_date(xmax)

    d1 = datetime.timedelta(hours=deltamin)
    d2 = datetime.timedelta(hours=deltamax)
    
    ax.set_xlim(t1-d1, t2+d2)
    
def ylim_hours(ax,xmin,xmax, deltamin=0.5, deltamax=0.9):
    # Expect xmin and xmax as timestamp strings
    
    def hour_to_date(x):
        if (len(x)==2): x=x+'0000'
        if (len(x)==4): x=x+'00'
        if (len(x)!=6): raise ValueError('x should be in format HH, HHMM or HHMMSS')
        return datetime.datetime(1900,1,1,int(x[0:2]),int(x[2:4]),int(x[4:6]))
    
    t1 = hour_to_date(xmin)
    t2 = hour_to_date(xmax)

    d1 = datetime.timedelta(hours=deltamin)
    d2 = datetime.timedelta(hours=deltamax)
    
    ax.set_ylim(t1-d1, t2+d2)
    

def yticks_hours(ax,ticks):
    # Expect timestamp strings
    
    def hour_to_date(x):
        if (isinstance(x, int)):
            x='{:02d}'.format(x)
        if (len(x)==2): x=x+'0000'
        if (len(x)==4): x=x+'00'
        if (len(x)!=6): raise ValueError('x should be in format int, HH, HHMM or HHMMSS')
        return datetime.datetime(1900,1,1,int(x[0:2]),int(x[2:4]),int(x[4:6]))
    
    ticks = [hour_to_date(h) for h in ticks]
    
    ax.set_yticks(ticks);
    

def camhour_y_from_series(cam,hour,maxdelta=0.8):
    return pd.Series( cam.apply(lambda x:int(x))+maxdelta/24*hour.apply(lambda x:int(x)) )

def camhour_y(df,maxdelta=0.8):
    hh = df['hh'] #df['hh_avi'].where(df['hh_avi'].notnull(),df['hh_mp4'])
    cam = df['cam'] #df['cam_avi'].where(df['cam_avi'].notnull(),df['cam_mp4'])
    return camhour_y_from_series(cam,hh,maxdelta=maxdelta)

# From https://stackoverflow.com/a/12329993
# reorder columns
def set_column_sequence(dataframe, seq, front=True):
    '''Takes a dataframe and a subsequence of its columns,
       returns dataframe with seq as first columns if "front" is True,
       and seq as last columns if "front" is False.
    '''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            if front: #we want "seq" to be in the front
                #so append current column to the end of the list
                cols.append(x)
            else:
                #we want "seq" to be last, so insert this
                #column in the front of the new column list
                #"cols" we are building:
                cols.insert(0, x)
    return dataframe[cols]


from tqdm.notebook import tqdm

def checkfileexist(relpath_series, rootpath):
    #df=df.iloc[[0]]
    fileexist = pd.Series(index=relpath_series.index, dtype="bool")
    with tqdm(total=relpath_series.size) as pbar:
        for key, relpath in relpath_series.iteritems():
            full = os.path.join(rootpath,relpath)
            exist = os.path.isfile(full)
            #print(full, exist)
            fileexist.loc[key] = exist
            pbar.update()
    return fileexist

def get_filesize(relpath_series, rootpath):
    #df=df.iloc[[0]]
    filesize = pd.Series(index=relpath_series.index, dtype="int")
    for key, relpath in tqdm(relpath_series.iteritems()):
        full = os.path.join(rootpath,relpath)
        exist = os.path.isfile(full)
        if (not exist):
            filesize.loc[key] = -1
        else:
            info = os.stat(full)   
            filesize.loc[key] = int(info.st_size)
    return fileexist

def daystamp(item):
    if (isinstance(item,pd.DataFrame)):
        return item.apply(daystamp,axis=1)
    return "{:02}{:02}{:02}".format(item.YY, item.MM, item.DD)
#mp4df.apply(daystamp, axis=1)