import json
import pandas as pd
import argparse
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def load_json(filename):
    with open(filename,"r") as f:
        T=json.load(f)
    return T

def load_tracks_json(filename):
    with open(filename,"r") as f:
        T=json.load(f)
    return T

def load_tags_json(filename):
    with open(filename,"r") as f:
        Tags=json.load(f)
    return Tags


def tracks_to_df(T):
    df = pd.DataFrame(columns=['frame','id','leaving','entering','pollen','walking','fanning','FA'])
    df = df.astype("bool")
    df[['frame','id']] = df[['frame','id']].astype("int64")
    
    all_labels=[]
    all_ids=[]
    
    for frameDict in T:
        if (frameDict is None): continue
        for id in frameDict:
            item=frameDict[id]
            if (item['labels'] == ''):
                labels=[]
            else:
                labels=item['labels'].split(',')
            for l in labels:
                if (l not in all_labels): all_labels.append(l)
            if (id not in all_ids): all_ids.append(id)
            df=df.append(dict(frame=int(item['frame']),
                              id=int(item['ID']),
                              labels=item['labels'],
                              leaving='leaving' in labels,
                              entering='entering' in labels,
                              pollen='pollen' in labels,
                              walking='walking' in labels,
                              fanning='fanning' in labels,
                              FA='falsealarm' in labels or 'wronglabel' in labels
                              )
                        ,ignore_index=True)   
    #print(all_labels)
    #print(all_ids)
    return df
    
def tags_to_df(Tags):
    df = pd.DataFrame(columns=['frame','id','hamming','c','p'])
    df[['frame','id']]=df[['frame','id']].astype(np.int32)
    
    all_ids=[]
    
    for framestr in Tags:
        frame=int(framestr)
        frameRecord = Tags[framestr]
        if (frameRecord is None): continue
        L = frameRecord['tags']
        for item in L:
            #item=L[i]
            if (item['id'] not in all_ids): all_ids.append(item['id'])
            D=dict(frame=frame, id=item['id'])
            D['hamming'] = item.get('hamming')
            D['c'] = item.get('c')
            D['p'] = item.get('p')
            df=df.append(D,ignore_index=True)   
    #print(all_labels)
    #print(all_ids)
    return df

def timestamping(df,timestring,fps=20):
    t0=pd.Timestamp(timestring)
    #df = df.reindex(columns = np.append( df.columns.values, ['time']))
    #df['time'] = df['time'].astype(pd.Timestamp)
    df['time'] = t0+pd.to_timedelta(df['frame']/fps,unit='s')
    
    df['datetime']=df['time'].apply(lambda d: pd.to_datetime(d))
    
    return df

def timestamp_from_filename(filename):
    result = re.match(r'.*?C(\d\d)_((\d\d\d\d\d\d)(\d\d\d\d\d\d))', filename)
    if (result is None): return None # Default
    videoname=result.group(0)
    camera_id=result.group(1)
    timestamp='20'+result.group(2)
    date=result.group(3)
    time=result.group(4)
    return timestamp

def load_fileset(inputlist):
    '''
    ex: inputlist="/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python/inputlist.csv"
    '''
    L = pd.read_csv(inputlist,
                header=0,names=['filename'])
    L[['timestamp']]=L[['filename']].applymap(timestamp_from_filename)
    
    df=pd.DataFrame()
    for index, row in L.iterrows():
        filename=row['filename']
        print("Loading {}...".format(filename))
        T=load_tracks_json(filename)
        df1=tracks_to_df(T)
        df1=timestamping(df1, row['timestamp'])
        df=df.append(df1,ignore_index=True)

    df[['datetime']]=df[['time']].applymap(lambda d: pd.to_datetime(d))
        
    #df = df.query('FA!=True')
    
    return df

def plot_activities(df, tagids=None):

    df = df.copy()
    
    if (tagids is None):
        ids=df['id'].unique()
        ids.sort()
    else:
        ids = tagids
    
    rmap = {int(id): int(i) for i,id in enumerate(ids)}    
    df['uid']=df['id'].apply(lambda x: rmap[int(x)])
    
    #fig=plt.figure()
    idx=df.query('not FA and not walking and not fanning and not pollen and not entering and not leaving').index
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'.',c='k',label='other')
    idx=df.index[df['FA'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'x',c='#a0a0a0',label='FA/WId',mfc='none')
    idx=df.index[df['walking'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'o',c='k',label='walking',mfc='none')
    idx=df.index[df['fanning'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'s',c='brown',label='fanning',mfc='none')
    idx=df.index[df['pollen'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'s',c='#EEC000',label='pollen',linewidth=3)
    idx=df.index[df['entering'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'<',c='r',label='entering',mfc='none')
    idx=df.index[df['leaving'].astype(bool)]
    plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'>',c='b',label='leaving',mfc='none')
    ax=plt.gca()
    ax.set_xlim(df.iloc[0]['datetime'],df.iloc[-1]['datetime'])
    plt.xticks(rotation='vertical')
    
    days = mdates.DayLocator()  
    hours = mdates.HourLocator()  
    dayFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    hourFmt = mdates.DateFormatter('%H:%M')
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dayFmt)
    #ax.xaxis.set_minor_locator(hours)
    #ax.xaxis.set_minor_formatter(hourFmt)

    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids)
    ax.set_ylim(-0.5,len(ids)-0.5)
    ax.grid(color='#888888', linestyle='-', linewidth=1)
    ax.legend()
    
    #return fig

import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
def plot_activities_df(in_df, plot_interval=False, tagids=None, leaving_symbol = '>', entering_symbol = '<'):
    
    def plot_intervals(ax, df, color):
        segs = [((mdates.date2num(item.track_starttime), item.uid),(mdates.date2num(item.track_endtime), item.uid)) for k,item in df.iterrows()]
        #print(segs)
        line_segments = LineCollection(segs, colors=color, linestyle='solid', linewidth=3)
        ax.add_collection(line_segments)

    df=in_df.copy()
    df.fillna(-1,inplace=True)
    
    df['tseq']=df.groupby(['track_tagid']).cumcount()
    
    if (tagids is None):
        ids=df['track_tagid'].unique()
        ids.sort()
    else:
        ids = tagids
        df = df[df['track_tagid'].isin(ids)]
    
    rmap = {int(id): int(i) for i,id in enumerate(ids)}    
    #print(rmap)
    df['uid']=df['track_tagid'].apply(lambda x: rmap[int(x)])
    
    df['uid']=df['uid'].astype(float)+0.2*(df.tseq%10)/10-0.1 #+np.random.normal(size=df['uid'].shape)*0.1
    
    if (plot_interval != 'only'):
        
        df1=df[df.walking]
        plt.plot(df1['datetime'].tolist(),df1['uid'],'o',c='k',label='visible',mfc='none')

        idx=df.index[df.leaving]
        plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],leaving_symbol,c='b',label='leaving',mfc='none')

        idx=df.index[df.entering]
        plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],entering_symbol,c='r',label='entering',mfc='none')

        idx=df.index[df.pollen]
        #plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],'s',ms=6,c='orange',mfc='orange',label='pollen')
        plt.plot(df['datetime'][idx].tolist(),df['uid'][idx],entering_symbol,c='orange',label='pollen', mfc='none')
        
    if (plot_interval):
        plot_intervals(plt.gca(), df[df.walking], 'k')
        plot_intervals(plt.gca(), df[df.leaving], 'b')
        plot_intervals(plt.gca(), df[df.entering], 'r')
    
    ax=plt.gca()
    ax.set_xlim(df.iloc[0]['datetime'],df.iloc[-1]['datetime'])
    plt.xticks(rotation='vertical')
    
    days = mdates.DayLocator()  
    hours = mdates.HourLocator()  
    dayFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    hourFmt = mdates.DateFormatter('%H:%M')
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dayFmt)
    #ax.xaxis.set_minor_locator(hours)
    #ax.xaxis.set_minor_formatter(hourFmt)

    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids)
    #ax.set_ylim(-0.5,len(ids)-0.5)
    ax.set_ylim(len(ids)-0.5,-0.5)   # top-to-bottom
    ax.grid(color='#888888', linestyle='-', linewidth=1)
    ax.legend()
    
    # Hide data inside the axes for interactive UI
    ax.activity_data_ = dict( ids=ids, rmap=rmap, df=df )
    
def format_multiday(ax=None):
    if (ax is None):
        ax=plt.gca()
    def format_date(x, pos=None):
        d=mdates.num2date(x)
        if d.hour==12:
            return d.strftime('%H\n%y-%m-%d')
        else:
            return d.strftime('%H')
    ax.xaxis.set_major_locator(mdates.HourLocator([0]))
    ax.xaxis.set_minor_locator(mdates.HourLocator([6,12,18]))
    #ax.xaxis.set_major_formatter(dates.DateFormatter('%y-%m-%d'))
    #ax.xaxis.set_minor_formatter(dates.DateFormatter('%H'))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(format_date))
    plt.legend(loc=4)
    #plt.xticks(pd.date_range('2017-06-21 00:00','2017-06-27 00:00'),rotation=0)
    plt.xticks(horizontalalignment="left")
    ax.tick_params('x', length=30, width=2, which='major',color='gray')

    import matplotlib.transforms
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=0) 

    ax.grid(b=True, which='major', axis='x', linewidth=2)

    # Create offset transform by 5 points in x direction
    dx = 5/72.; dy = 15/72. 
    fig = plt.gcf()
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    
def define_fileset(files):
    '''
    ex: inputlist="/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python/inputlist.csv"
    '''
   
    L = pd.DataFrame(files, columns=['filename'])
    L['timestamp']=L['filename'].apply(timestamp_from_filename)
    
    L=pd.concat([L, pd.DataFrame(L['timestamp'].apply(lambda x: pd.Series(dict(YY=int(x[2:4]),MM=int(x[4:6]),DD=int(x[6:8]),hh=int(x[8:10]),mm=int(x[10:12]),ss=int(x[12:14]))) ))],axis=1)

    L['daystamp']=L.apply(lambda x: f"{x.YY:02}{x.MM:02}{x.DD:02}", axis=1)
    
    return L

    
def load_annotations(L):
    '''
    ex: inputlist="/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python/inputlist.csv"
    '''
   
    df=pd.DataFrame()
    for index, row in L.iterrows():
        filename=row['filename']
        print("Loading {}...".format(filename))
        T=load_tracks_json(filename)
        df1=tracks_to_df(T)
        df1=timestamping(df1, row['timestamp'])
        df=df.append(df1,ignore_index=True)

    df[['datetime']]=df[['time']].applymap(lambda d: pd.to_datetime(d))
        
    #df = df.query('FA!=True')
    
    return df


from datetime import datetime as dt, time, timedelta
from tqdm.autonotebook import tqdm
import re
def fix_P_o_l_l_e_n(T):
    P_o_l_l_e_n = 'Pollen' #[L for L in 'Pollen']
    data=T['data']
    for f in data.keys():
        Tf=data[f]
        for item in Tf:
            if (item['labels']==P_o_l_l_e_n):
                item['labels']=['Pollen']
def extract_events(T, starttime, videoid):
    out=[]
    data=T['data']
    H0=starttime #dt(2017,6,21)
    for f in tqdm(data.keys(),mininterval=1.0):
        Tf=data[f]
        for item in Tf:
            CEL=item['classifier']['entering_leaving']
            EL=list(CEL.keys())[0]
            #if (EL=='Not_Event'):
            #    pass#continue
            CELL=CEL[EL]
            L=CELL['length']
            sp=CELL['start_position']
            ep=CELL['end_position']
            f=int(item['frame'])
            P='Pollen' in item['labels']
            Tag=item['tag']
            if (Tag is None):
                tagid=pd.NA
                taghamming=pd.NA
                tx,ty=pd.NA,pd.NA
                tagdm=pd.NA
            else:
                tagid=int(Tag['id'])
                taghamming=int(Tag['hamming'])
                tx,ty=Tag['c']
                tagdm=Tag['dm']
            frame=int(item['frame'])
            x=item['cx']; y=item['cy']
            out.append(dict(videoid=int(videoid),
                            trackid=int(item['id']),
                            frame=int(frame),
                            datetime=H0+timedelta(seconds=f/20),
                            #entering_leaving=EL,
                            #labels=';'.join(item['labels']),
                            cx=x, cy=y,
                            entering= EL=='entering',
                            leaving= EL=='leaving',
                            walking= EL=='Not_Event',
                            pollen=P,
                            tagid=tagid,
                            taghamming=taghamming,
                            tx=tx,ty=ty,
                            tagdm=tagdm,
                            #tag=Tag,
                            track_class=EL,
                            track_length=L,
                            track_startx=sp[0],
                            track_starty=sp[1],
                            track_starta=sp[2],
                            track_endx=ep[0],
                            track_endy=ep[1],
                            track_enda=ep[2]
                           ))
    return out
def extract_events_from_files(files):
    allout=[]
    for i,f in enumerate(files):
        videoid=i
        M=re.search('_C\d{2}_(\d{12})_',f)
        ts=M.group(1)
        YY,MM,DD,hh,mm,ss=[int(x) for x in [ts[0:2],ts[2:4],ts[4:6],ts[6:8],ts[8:10],ts[10:12]]]
        starttime=dt(2000+YY,MM,DD,hh,mm,ss)
        print(f, starttime)
        print('  Load')
        #T=lb.load_tracks_json(f)
        T=load_tracks_json(f)
        print('  Extract')
        fix_P_o_l_l_e_n(T)
        out = extract_events(T, starttime, videoid)
        allout = allout+out
    return pd.DataFrame(allout)
def augment_df(df):
    df['hastag']=~(df.tagid.isnull())
    g=df.groupby(['videoid','trackid'])
    df['track_haspollen']=g.pollen.transform('any')
    df['track_hastag']=g.hastag.transform('any')
    df['track_tagid']=g.tagid.transform(lambda x: np.nan if x.isnull().all() else int(pd.Series.mode(x)[0]))
    df['track_startframe']=g.frame.transform('min')
    df['track_endframe']=g.frame.transform('max')
    df['track_starttime']=g.datetime.transform('min')
    df['track_endtime']=g.datetime.transform('max')
    return df



def detections_ivan_to_df(detections, debug=False):
    out = []
    
    def qprint(*args):
        if (debug):
            print(*args)
    
    keylist = list( detections.keys() )
    for i in tqdm(range(len(keylist))):
        key = keylist[i]
        frame=int(key)
        
        PP=detections[key]['parts']
        
        #qprint(f'### FRAME {frame}')
        #qprint(PP)
        
        for partid in PP: # dict {'partid': P, ...}
            P = PP[partid]
            #qprint(f'  partid {partid}')
            #qprint(P)
            for detid,det in enumerate(P): # array [detection0, ...]
                #qprint(f'  detid {detid}')
                #qprint(p)
                out.append(dict(frame=frame, partid=int(partid), detid=int(detid), cx=det[0], cy=det[1], score=det[2]))
        
    df = pd.DataFrame.from_records(out,columns='frame partid detid cx cy score'.split())
    return df

def hungarian_tracking_with_prediction(detections, max_dist,part=str(2), nms_min_dist=50, max_step=1, debug=False, progress=True):
    """
    This function executes the hungarian algorithm. It is expecting to receive an instance of detections. Please check documentation for data structure. It will also consider the maximum distance and it outputs in a new file with the data structure explained in documentation. 
    
    Inputs: 
        - detections_path: Path to find detections file
        - cost : Maximum distance allowed
        - output: Optional, if '', will use the same path as were the detections are. 
        - part : Over what part perform the tracking. By default thorax or '2'
    
    """
    #if output =='':
    #    output=detections_path
        
    #detections= load_json(detections_path)
    keylist = list( detections.keys() ) # Frames
    #framelist = sorted([int(k) for k in detections.keys()])
    
    def dprint(*args):
        if (debug):
            print(*args)
    
    #cmaps={}
    
    #track_id={}
        
    track_raw={}
    activetracks=[]
    lasttrackid=-1
    key_prev=-1
    #parts_prev=np.zeros( (0,4) )
    #vel_prev=np.zeros( (0,2) )
    state_prev=np.zeros( (0,4) )  # [x,y,vx,vy]
    
    for i in tqdm(range(len(keylist)), disable=~progress):
        key = keylist[i]
        frame=int(key)
        if (key_prev==-1):
            framedelta = 1
        else:
            framedelta = int(key)-int(key_prev)
        key_prev=frame
        dprint(f"### FRAME {frame}, delta={framedelta}")
          
        #print(f'active {activetracks}')
        #print(f'state_prev {state_prev}')
        
        # TERMINATE: Kill all tracks that didnt have detection for some time
        nactive=len(activetracks)
        keepactive=np.zeros((nactive,),dtype=bool)
        for j in range(nactive): 
            tj = activetracks[j]
            gap=int(key)-track_raw[tj]['endframe']
            keepactive[j]= gap<=max_step
            if (not keepactive[j]):
                dprint(f"Close Track {tj} (Active {j}), gap {gap} frames")
        for j in np.nonzero(~keepactive)[0]:
            tj = activetracks[j]
            # Eliminate virtual detections that were not confirmed
            T=track_raw[tj]
            del T['virtual']
            # Eliminate if too short
            #if (T['endframe']-T['startframe']<min_track_len):
            #    del track_raw[tj]
            
        activetracks = [activetracks[i] for i in range(len(keepactive)) if keepactive[i]]
        state_prev = state_prev[keepactive,...]
        nactive=len(activetracks)
        dprint(f'Active Tracks {activetracks}')
        
        # PREDICT all active tracks for current frame (constant velocity)
        state_prev[:,0:2] = state_prev[:,0:2]+state_prev[:,2:4]*framedelta # [x,y]
        #state_prev[:,2:4] = state_prev[:,2:4]
        #print(state_prev)
        
        # GATHER OBSERVATIONS
        # Get all thoraxes
        parts = np.array(detections[key]['parts'][part]) # Nx3 (x,y,score)
        parts = np.concatenate([parts,np.arange(parts.shape[0]).reshape(-1,1)],axis=1) # Nx4 (x,y,score,k)
        
        # NMS
        keep = nms_euclid(parts, nms_min_dist)
        #keep = np.ones((parts.shape[0],),dtype=bool)
        keepids, = np.nonzero(keep)
        partsnms = parts[keep,...]
        #print(f'parts {parts}')
        #print(f'ndet {len(keep)}')
        #print(f'keepids {keepids}')
        nNMS=len(keepids)
        
        dprint(f'nms delete {np.nonzero(~keep)}')
        
        # MATCH
        cmap = cost_matrix_tracks_vectorized(state_prev[:,:2], partsnms[:,:2], max_dist)
        #cmaps[key]=cmap # Log for debug
        _,idx=hungarian(cmap)
        revidx=np.zeros_like(idx)
        revidx[idx]=np.arange(idx.size)
        dprint(f"idx {idx[:nactive]} | {idx[nactive:]}")
        dprint(f"revidx {revidx[:nNMS]} | {revidx[nNMS:]}")
        unmatched_det = np.flatnonzero(revidx[:len(keepids)]>=nactive) # kidx[keepid]=j
        dprint(f"unmatched_det {unmatched_det}")
        
        # Extend existing tracks
        for j in range(nactive): 
            tj = activetracks[j]   # track id tj
            T = track_raw[tj]  # Existing track
            
            predictpos = state_prev[j,0:2].tolist()
            predictvel = state_prev[j,2:4].tolist()
            
            # Matched existing track tj with detection k
            k=idx[j]  # column k in cmap == id k within NMS detections
            matching_dist=cmap[j,k]
            if (matching_dist<max_dist): # MATCHED for j
                dk=keepids[k]  # id dk within all detections for frame key
                
                dprint(f'Track {tj} (Active {j}) += Det {dk}  (NMS {k})  pos {parts[dk,0:2]}')  
                
                # Insert intermediate virtual detections
                T['data'].update(T['virtual'])
                T['virtual']={}
                
                # Update from observation
                # Recursive estimate of velocity with exponential decay
                alpha = 0.5
                state_prev[j,2:4] = (1.0-alpha)*state_prev[j,2:4]+alpha*(parts[dk,0:2]-state_prev[j,0:2])/framedelta # [vx,vy]
                # Update position to observation
                state_prev[j,0:2] = parts[dk,0:2] # [x,y]
                
                T['endframe']=frame
                T['nbdetections']+=1
                T['data'][frame]=dict(detid=dk, pos=parts[dk,0:2].tolist(), score=parts[dk,2], cost=matching_dist, vel=state_prev[j,2:4].tolist(),
                                      predictpos=predictpos, predictvel=predictvel)

            else: # NO MATCH for j
                T['virtual'][frame]=dict(detid=-1, pos=state_prev[j,0:2].tolist(), score=-1, cost=-1, vel=state_prev[j,2:4].tolist(),
                                         predictpos=predictpos, predictvel=predictvel)
                
                dprint(f'Track {tj} (Active {j}) += Virtual Det    pos {state_prev[j,0:2]}')
            
        # Create tracks for unmatched detections
        
        #print(f'unmatched_det {unmatched_det}')
        for k in unmatched_det:
            dk=keepids[k]
            # Detection without track
            lasttrackid+=1
            T = {}; track_raw[lasttrackid]=T
            T['startframe']=frame
            T['endframe']=frame
            T['nbdetections']=1
            T['data']={}
            T['data'][frame]=dict(detid=dk, pos=parts[dk,0:2].tolist(), score=parts[dk,2], cost=0, vel=[0.0,0.0])
            T['virtual']={}
            
            # Append to active tracks
            activetracks.append(lasttrackid)
            state = np.array([[parts[dk,0], parts[dk,1], 0.0, 0.0]])
            state_prev = np.append(state_prev, state, axis=0)
            #print(state_prev)
            
            dprint(f'New Track {lasttrackid} (Active {len(activetracks)-1}) += Det {dk} (NMS {k}) pos {parts[dk,0:2]}')
         
    # CLEANUP
    for j in range(len(activetracks)):
        tj = activetracks[j]
        # Eliminate virtual detections that were not confirmed
        T=track_raw[tj]
        del T['virtual']
            
    return track_raw

def hungarian_tracking_with_prediction_df(detdf, max_dist, nms_min_dist=50, max_step=1, debug=False, progress=True, dets=None):
    """
    This function executes the hungarian algorithm. It is expecting to receive an instance of detections. Please check documentation for data structure. It will also consider the maximum distance and it outputs in a new file with the data structure explained in documentation. 
    
    Inputs: 
        - detection df, with fields frame,detid,cx,cy,score
        - cost : Maximum distance allowed
        - output: Optional, if '', will use the same path as were the detections are. 
        - part : Over what part perform the tracking. By default thorax or '2'
    
    """

    #detdf = detdf.sort_values(['frame','detid'])
    framelist = detdf.frame.unique() # Frames
    
    #assert(len(detdf.partid.unique())==1, 'Only one type of part allowed for tracking')
    
    def dprint(*args):
        if (debug):
            print(*args)
    
    if (dets is None):
        dets = detdf[['cx','cy','score','detid','frame']].groupby('frame').apply(lambda df: df.to_numpy()) # Nx5 (x,y,score,detid, frame)))
        
    track_raw={}
    activetracks=[]
    lasttrackid=-1
    frame_prev=-1
    
    state_prev=np.zeros( (0,4) )  # [x,y,vx,vy]
    
    for i in tqdm(range(len(framelist)), disable=~progress):
        frame=framelist[i]
        if (frame_prev==-1):
            framedelta = 1
        else:
            framedelta = frame-frame_prev
        frame_prev=frame
        dprint(f"### FRAME {frame}, delta={framedelta}")
          
        #print(f'active {activetracks}')
        #print(f'state_prev {state_prev}')
        
        # TERMINATE: Kill all tracks that didnt have detection for some time
        nactive=len(activetracks)
        keepactive=np.zeros((nactive,),dtype=bool)
        for j in range(nactive): 
            tj = activetracks[j]
            gap=frame-track_raw[tj]['endframe']
            keepactive[j]= gap<=max_step
            if (not keepactive[j]):
                dprint(f"Close Track {tj} (Active {j}), gap {gap} frames")
        for j in np.nonzero(~keepactive)[0]:
            tj = activetracks[j]
            # Eliminate virtual detections that were not confirmed
            T=track_raw[tj]
            del T['virtual']
            # Eliminate if too short
            #if (T['endframe']-T['startframe']<min_track_len):
            #    del track_raw[tj]
            
        activetracks = [activetracks[i] for i in range(len(keepactive)) if keepactive[i]]
        state_prev = state_prev[keepactive,...]
        nactive=len(activetracks)
        #print(f'active after kill {activetracks}')
        dprint(f'Active Tracks {activetracks}')
        
        # PREDICT all active for current frame 
        # Keep same velocity
        #state_prev[:,2:4] = state_prev[:,2:4]
                
        # Predict current position with constant velocity
        state_prev[:,0:2] = state_prev[:,0:2]+state_prev[:,2:4]*framedelta # [x,y]
        #print(state_prev)
        
        # GATHER OBSERVATIONS
        # Get all thoraxes
        #partsdf = tupledf.get_group(frame)
        #parts = np.array(partsdf[['cx','cy','score','detid']]) # Nx3 (x,y,score,detid)
        parts = dets[frame][:,:4] # Nx3 (x,y,score,detid)
        #parts = np.concatenate([parts,np.arange(parts.shape[0]).reshape(-1,1)],axis=1) # Nx4 (x,y,score,k)
        
        # NMS
        keep = nms_euclid(parts, nms_min_dist)
        #keep = np.ones((parts.shape[0],),dtype=bool)
        keepids, = np.nonzero(keep)
        partsnms = parts[keep,...]
        #print(f'parts {parts}')
        #print(f'ndet {len(keep)}')
        #print(f'keepids {keepids}')
        nNMS=len(keepids)
        
        dprint(f'nms delete {np.nonzero(~keep)}')
        
        # MATCH
        cmap = cost_matrix_tracks_vectorized(state_prev[:,:2], partsnms[:,:2], max_dist)
        #cmaps[key]=cmap # Log for debug
        _,idx=hungarian(cmap)
        revidx=np.zeros_like(idx)
        revidx[idx]=np.arange(idx.size)
        dprint(f"idx {idx[:nactive]} | {idx[nactive:]}")
        dprint(f"revidx {revidx[:nNMS]} | {revidx[nNMS:]}")
        unmatched_det = np.flatnonzero(revidx[:len(keepids)]>=nactive) # kidx[keepid]=j
        dprint(f"unmatched_det {unmatched_det}")
        
        # Extend existing tracks
        for j in range(nactive): 
            tj = activetracks[j]   # track id tj
            T = track_raw[tj]  # Existing track
            
            predictpos = state_prev[j,0:2].tolist()
            predictvel = state_prev[j,2:4].tolist()
            
            # Matched existing track tj with detection k
            k=idx[j]  # column k in cmap == id k within NMS detections
            matching_dist=cmap[j,k]
            if (matching_dist<max_dist): # MATCHED for j
                dk=keepids[k]  # id dk within all detections for frame key
                
                dprint(f'Track {tj} (Active {j}) += Det {dk}  (NMS {k})  pos {parts[dk,0:2]}')  
                
                # Insert intermediate virtual detections
                T['data'].update(T['virtual'])
                T['virtual']={}
                
                # Update from observation
                # Recursive estimate of velocity with exponential decay
                alpha = 0.5
                state_prev[j,2:4] = (1.0-alpha)*state_prev[j,2:4]+alpha*(parts[dk,0:2]-state_prev[j,0:2])/framedelta # [vx,vy]
                # Update position to observation
                state_prev[j,0:2] = parts[dk,0:2] # [x,y]
                
                T['endframe']=frame
                T['nbdetections']+=1
                T['data'][frame]=dict(detid=dk, pos=parts[dk,0:2].tolist(), score=parts[dk,2], cost=matching_dist, vel=state_prev[j,2:4].tolist(),
                                      predictpos=predictpos, predictvel=predictvel)

            else: # NO MATCH for j
                T['virtual'][frame]=dict(detid=-1, pos=state_prev[j,0:2].tolist(), score=-1, cost=-1, vel=state_prev[j,2:4].tolist(),
                                         predictpos=predictpos, predictvel=predictvel)
                
                dprint(f'Track {tj} (Active {j}) += Virtual Det    pos {state_prev[j,0:2]}')
            
        # Create tracks for unmatched detections
        
        #print(f'unmatched_det {unmatched_det}')
        for k in unmatched_det:
            dk=keepids[k]
            # Detection without track
            lasttrackid+=1
            T = {}; track_raw[lasttrackid]=T
            T['startframe']=frame
            T['endframe']=frame
            T['nbdetections']=1
            T['data']={}
            T['data'][frame]=dict(detid=dk, pos=parts[dk,0:2].tolist(), score=parts[dk,2], cost=0, vel=[0.0,0.0])
            T['virtual']={}
            
            # Append to active tracks
            activetracks.append(lasttrackid)
            state = np.array([[parts[dk,0], parts[dk,1], 0.0, 0.0]])
            state_prev = np.append(state_prev, state, axis=0)
            #print(state_prev)
            
            dprint(f'New Track {lasttrackid} (Active {len(activetracks)-1}) += Det {dk} (NMS {k}) pos {parts[dk,0:2]}')
         
    # CLEANUP
    for j in range(len(activetracks)):
        tj = activetracks[j]
        # Eliminate virtual detections that were not confirmed
        T=track_raw[tj]
        del T['virtual']
            
    return track_raw

def hungarian_tracking_with_prediction_from_file(detections_path, max_dist, part=str(2), nms_min_dist=50, max_step=1, debug=False, progress=True):
    params=locals(); del params['detection_path']
    detections = read_json(detections_path)
    track_raw = hungarian_tracking_with_prediction_new(detections, **params)
    
    # TODO: convert track_raw to Ivan's format


#import numba
from scipy.optimize import linear_sum_assignment as hungarian

def nms_euclid(det, min_dist):
    """
    det is Nx3 array, each row [x,y,score,...]
    
    return bool array, True if keep, False if suppress
    """
    # if there are no boxes, return an empty list
    if (det.shape[0]==0):
        #return np.zeros((0,),dtype=bool)
        return det[:,0]  # already empty
 
    # grab the coordinates of the bounding boxes
    x = det[:,(0,)] # Make sure to keep 2 dims
    y = det[:,(1,)]
    score = det[:,(2,)]
    
    # Vectorized comparison of all pairs
    M_close = ((x-x.T)**2+(y-y.T)**2)<(min_dist**2)
    M_upper = np.triu(np.ones(M_close.shape,dtype=bool),k=1)
    M_lowerscore = (score < score.T) | ((score==score.T)& M_upper)  # M[i,j] True if score[i]<score[j]
    # Corner case: is same score, arbitrarily keep the node j>i
    
    dominated = (M_lowerscore & M_close).any(axis=1) # Node of row `i` is dominated if any node `j`  dominates it
    
    #print("M_close",M_close)
    #print("M_upper",M_upper)
    #print("M_lowerscore",M_lowerscore)
    #print("dominated",dominated)
    
    # Caution: corner case
    # with this approach, one local maximum can suppress maxima with less score
    # up to arbitrary radius if the sequence of scores is decreasing from neigbor to neirbor
    # as each max in the sequence will suppress the next one
    # I guess this is what we want?
    
    # return boolean array of detections to keep
    return ~dominated

def cost_matrix_tracks_vectorized(ground_t,detections,threshold):
    ground_t=ground_t.reshape(-1,2)
    detections=detections.reshape(-1,2)
    Ng = ground_t.shape[0]
    Nd = detections.shape[0]
    total = Ng+Nd
    cost_m = np.zeros((total,total))+threshold
    cost_m[:Ng,:Nd] = np.sqrt( (ground_t[:,[0]]-detections[:,[0]].T)**2 + (ground_t[:,[1]]-detections[:,[1]].T)**2 )
    return cost_m
    
    
def hungarian_tracking_with_prediction_new(detections, max_dist=100, part='2', nms_min_dist=50, max_step=1, min_track_length=1, decay_alpha=0.5, 
                                           debug=False, progress=True):
    """
    This function executes the hungarian algorithm. It is expecting to receive an instance of detections. Please check documentation for data structure. 
    It will also consider the maximum distance and it outputs in a new file with the data structure explained in documentation. 
    
    Inputs: 
        - detections_path: Path to find detections file
        - cost : Maximum distance allowed
        - output: Optional, if '', will use the same path as were the detections are. 
        - part : Over what part perform the tracking. By default thorax or '2'
    
    """
    keylist = list( detections.keys() ) # Frames
    #framelist = sorted([int(k) for k in detections.keys()])
    
    def dprint(*args):
        if (debug):
            print(*args)
            
    def get_detections(detections, key, part):
        # Customizable function to use different input structures
        dets0 = np.array(detections[key]['parts'][part]) # Nx3 (x,y,score)
        # Add column with original detection id `dk` in the frame
        dets0 = np.concatenate([dets0,np.arange(dets0.shape[0]).reshape(-1,1)],axis=1) # Nx4 (x,y,score,dk)
        return dets0
    
    def close_track(track_raw, tj, min_track_length=None):
        T = track_raw[tj]
        del T['virtual']
        # Eliminate if too short
        if (min_track_length is not None):
            if (T['endframe']-T['startframe']+1<min_track_length):
                #dprint(f"Delete Short Track {tj} (Active {j}), gap {gap} frames")
                del track_raw[tj]            

    track_raw =  {}
    activetracks = []  # Keep as a list  (TODO: see if np.array would be better?)
    lasttrackid = -1
    frame_prev = -1
    tracker_state = np.zeros( (0,4) )  # [x,y,vx,vy]
    
    # FOR EACH FRAME:
    #for i in tqdm(range(len(keylist)), disable=~progress):
    for i in range(len(keylist)):
        key = keylist[i]
        frame=int(key)
        
        framedelta = 0 if (frame_prev==-1) else frame-frame_prev
        frame_prev=frame
        #dprint(f"### FRAME {frame}, delta={framedelta}")
        
        #dprint(f'activetracks {activetracks}')
        #dprint(f'tracker_state {tracker_state}')
        
        # 0. DELETE LOST TRACKS
        keepactive=np.zeros_like(activetracks,dtype=bool)
        
        for j,tj in enumerate(activetracks): 
            T = track_raw[tj]  # Existing track
            gap = frame-T['endframe']
            isactive = (gap <= max_step)
            keepactive[j] = isactive
            if (not isactive):
                #dprint(f"Close Track {tj} (Active {j}), gap {gap} frames")
                close_track(track_raw, tj, min_track_length=min_track_length)
        
        activetracks = [activetracks[i] for i in np.flatnonzero(keepactive)]
        tracker_state = tracker_state[keepactive,...]
        nactive = len(activetracks)
        #dprint(f'Active Tracks {activetracks}')
        
        # 1. PREDICT all active tracks for current frame (constant velocity)
        #dprint(f'# PREDICT')
        tracker_state[:,0:2] = tracker_state[:,0:2]+tracker_state[:,2:4]*framedelta # [x,y]
        #tracker_state[:,2:4] = tracker_state[:,2:4]
        #dprint(f'tracker_state {tracker_state}')
        
        # 2. GATHER OBSERVATIONS
        # Get all thoraxes (or reference part defined by `part`)
        #dets0 = get_detections(detections, key, part)   # Array with each row as [x,y,score,dk]
        dets0 = np.array(detections[key]['parts'][part]) # Nx3 (x,y,score)
        # Add column with original detection id `dk` in the frame
        dets0 = np.concatenate([dets0,np.arange(dets0.shape[0]).reshape(-1,1)],axis=1) # Nx4 (x,y,score,dk)
        # NMS
        if (nms_min_dist is not None):
            keep = nms_euclid(dets0, nms_min_dist)
            dets = dets0[keep,...]
            #dprint(f'nms delete [{np.nonzero(~keep)}]')
        else:
            dets=dets0
        detids = dets[:,3]  # 4th column is original detection id
        
        ndets = dets.shape[0]
        nactive = len(activetracks)
        #dprint(f'ndets {ndets}')
        #dprint(f'nactive {nactive}')
        
        # 3. MATCH DETECTIONS AND TRACKS
        cmat = cost_matrix_tracks_vectorized(tracker_state[:,:2], dets[:,:2], max_dist)
        _,idx = hungarian(cmat)
        revidx = np.zeros_like(idx)
        revidx[idx] = np.arange(idx.size)
        unmatched_det = np.flatnonzero(revidx[:ndets]>=nactive)
        
        #dprint(f"idx {idx[:nactive]} | {idx[nactive:]}")
        #dprint(f"revidx {revidx[:ndets]} | {revidx[ndets:]}")
        #dprint(f"unmatched_det {unmatched_det}")
        
        # 4. EXTEND EXISTING TRACKS
        keepactive=np.ones((nactive,),dtype=bool)
        for j,tj in enumerate(activetracks): 
            T = track_raw[tj]  # Existing track
            
            predictpos = tracker_state[j,0:2].tolist()
            predictvel = tracker_state[j,2:4].tolist()
            
            # Matched existing track tj with detection k
            k=idx[j]  # column k in cmap == id k within NMS detections
            matching_dist=cmat[j,k]
            if (matching_dist<max_dist): # MATCHED track tj to detection k
                dk=detids[k]  # id dk within all detections for frame key
                
                #dprint(f'Track {tj} (Active {j}) += Det {dk}  (NMS {k})  pos {dets[k,0:2]}')
                
                # Insert intermediate virtual detections
                T['data'].update(T['virtual'])
                T['virtual']={}
                
                # Update from observation
                # Recursive estimate of velocity with exponential decay (use existinting state)
                tracker_state[j,2:4] = (1.0-decay_alpha)*tracker_state[j,2:4]+decay_alpha*(dets[k,0:2]-tracker_state[j,0:2])/framedelta # [vx,vy]
                # Update position to observation
                tracker_state[j,0:2] = dets[k,0:2] # [x,y]
                
                T['endframe']=frame  # Last frame with a matching detection
                T['nbdetections']+=1
                T['data'][frame]=dict(detid=dk, pos=dets[k,0:2].tolist(), score=dets[k,2], cost=matching_dist, vel=tracker_state[j,2:4].tolist(),
                                      predictpos=predictpos, predictvel=predictvel)

            else: # NO MATCH for track tj
                # Still alive, but need to create a virtual detection
                T['virtual'][frame]=dict(detid=-1, pos=predictpos, score=-1, cost=-1, vel=predictvel,
                                     predictpos=predictpos, predictvel=predictvel)
                #dprint(f'Track {tj} (Active {j}) += Virtual Det    pos {tracker_state[j,0:2]}')
            
        # 6. CREATE NEW TRACKS FOR UNMATCHED DETECTIONS
        #dprint(f'unmatched_det {unmatched_det}')
        for k in unmatched_det:
            dk=detids[k]  # Original detection id before NMS (not really used here, for debugging purpose)
            
            lasttrackid+=1
            T = {}; track_raw[lasttrackid]=T
            T['startframe'] = frame
            T['endframe'] = frame
            T['nbdetections'] = 1
            T['data'] = {}
            T['data'][frame] = dict(detid=dk, pos=dets[k,0:2].tolist(), score=dets[k,2], cost=0, vel=[0.0,0.0])
            T['virtual'] = {}
            
            # Append to active tracks
            activetracks.append(lasttrackid)
            state_k = np.array([[dets[k,0], dets[k,1], 0.0, 0.0]])
            tracker_state = np.append(tracker_state, state_k, axis=0)
            #print(tracker_state)
            
            #dprint(f'New Track {lasttrackid} (Active {len(activetracks)-1}) += Det {dk} (NMS {k}) pos {dets[k,0:2]}')
        nactive = len(activetracks)
         
    # CLEANUP
    for _,tj in enumerate(activetracks):
        close_track(track_raw, tj, min_track_length=min_track_length)
        # Do not clean tracker_state as it is not used anymore
            
    return track_raw

def trackraw_to_df(track_raw, videoid):
#     'videoid', 'trackid', 'frame', 'datetime', 'cx', 'cy', 'entering',
#        'leaving', 'walking', 'pollen', 'tagid', 'taghamming', 'tx', 'ty',
#        'tagdm', 'track_class', 'track_length', 'track_startx', 'track_starty',
#        'track_starta', 'track_endx', 'track_endy', 'track_enda', 'hastag',
#        'track_haspollen', 'track_hastag', 'track_tagid', 'track_startframe',
#        'track_endframe', 'track_starttime', 'track_endtime'
    out = []
    for trackid in tqdm(track_raw.keys()):
        T=track_raw[trackid]
        TD=T['data']
        for fstr in TD:
            frame=int(fstr)
            #print(trackid,frame)
            D=TD[fstr]
            predict=D.get('predictpos',[0,0])
            out.append(dict(videoid=videoid,
                           trackid=trackid,
                           frame=frame,
                           cx=D['pos'][0],cy=D['pos'][1],detid=D['detid'],score=D['score'],
                           entering=False,leaving=False,walking=False,pollen=False,
                           tagid=pd.NA,taghamming=pd.NA,tx=pd.NA,ty=pd.NA,tagdm=pd.NA,
                           track_class=pd.NA,track_length=T['endframe']-T['startframe']+1,
                           hastag=False,track_haspollen=False,track_hastag=False,track_tagid=pd.NA,
                           track_startframe=T['startframe'], track_endframe=T['endframe'],
                           virtual=D['cost']==-1, vx=D['vel'][0],vy=D['vel'][1],
                           predictx=predict[0], predicty=predict[1]))
    return pd.DataFrame(out)  
            

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj,np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj) 

def main(args):
  file=pd.read_json(annotations,output_file)
  with open(output_file,"w") as f:
      json.dump(Annotations,f)

if __name__ == "__main__": 
	parser = argparse.ArgumentParser()
	parser.add_argument('-il',dest="inputlist",help="Input list as CSV")
	parser.add_argument('-o',dest="output",help="Output file")
	args = parser.parse_args()
	
	main(args)

