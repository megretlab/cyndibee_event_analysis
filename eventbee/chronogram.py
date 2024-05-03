import pandas as pd
import numpy as np
#import json
#import argparse
#import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection



## Generation of activity chronogram plot

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

    ax.grid(visible=True, which='major', axis='x', linewidth=2)

    # Create offset transform by 5 points in x direction
    dx = 5/72.; dy = 15/72. 
    fig = plt.gcf()
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


## Interactivity with activity chronogram plot

def find_closest_item(vdf, axes, mpl_event):
    #print('find_closest_item',mpl_event)
    #saveme['event']=event
    
    # CONVERT RAW COORDINATES X,Y INTO DATETIME AND ID
    x,y = mpl_event.xdata, mpl_event.ydata
    #ts = num2date(x, tz=None)
    ts = pd.Timestamp(mdates.num2date(x)).tz_localize(None)  # avoid error "Cannot compare tz-naive and tz-aware datetime-like objects"
    ry = round(y)
    ids = axes.activity_data_['ids']
    if ( (ry>=0) and (ry < len(ids)) ):
        tagid = ids[ry]
    else:
        tagid = None
    #print("ID=",tagid,"date=",ts)
    if (id is None):
        return
    
    # FIND CLOSEST EVENT FOR SELECTED ID
    # https://stackoverflow.com/questions/42264848/pandas-dataframe-how-to-query-the-closest-datetime-index
    ddf0 = vdf[vdf.track_tagid==tagid]
    ddf0 = ddf0[ ddf0.entering | ddf0.leaving | ddf0.pollen ]   # Keep only specific types of events
    iloc0_idx = (ddf0['track_starttime']-ts).abs().argsort().iloc[0]
    loc_idx = ddf0.index[iloc0_idx]
    #print('Event loc=',loc_idx)
    closest_item = ddf0.iloc[iloc0_idx]
    #print(closest_item)
    #iloc_idx = vdf.index.get_indexer([loc_idx]) # Need list of target values
    #print('Event iloc=',iloc_idx, ' loc=',loc_idx)
    # Check Right click in output then "Show Log Console" in Jupyterlab for output

    return closest_item

def build_chronogram_click_callback(vdf, axes, item_clicked_cb):
    # Build a callback function that can be used to get the clicked event
    # fig.canvas.mpl_connect('button_press_event', on_click_cb)
    # When clicking activities chronogram, first find the closest item from vdf and axes, then call
    #   item_clicked_cb(clicked_item)
    # Requires axes to have axes.activity_data_ user data defined to work

    def click_callback(event):
        # Closure function that serve as a matplotlib click callback
        clicked_item = find_closest_item(vdf, axes, event)
        item_clicked_cb(clicked_item)

    return click_callback

def register_chronogram_event_click(fig, vdf, axes, item_clicked_cb):
    # Register the callback
    click_cb = build_chronogram_click_callback(vdf, axes, item_clicked_cb)
    click_cb_id = fig.canvas.mpl_connect('button_press_event', click_cb)
    return click_cb_id  # Return the id if user wants to deregister later

