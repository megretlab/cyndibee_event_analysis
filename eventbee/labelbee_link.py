

def labelbee_url(event):
    videoid = event['labelbee_videoid']
    frame = event['track_startframe']
    url = f"http://136.145.54.85/webapp/labelbee/gui#video_id={videoid}&frame={frame}" 
    return url