
from scipy import stats

def track_classification(video, inside=300, outside=600, threshold=5, pollen_score="average", tag="mode"):

    for track in video.tracks.values():
        _, start = track.start.center
        _, end = track.end.center

        if len(track) > threshold:
            if start < inside and end < inside :
                for body in track:
                    x, y = body.center
                    if y > inside:
                        track.event = 'walking'
                        track.track_shape = 'inside_inside'
                        break
                if track.track_shape is None:
                    track.track_shape = 'inside'
            # Outside 
            elif start > outside and end > outside:
                for body in track:
                    x, y = body.center
                    if y < outside:
                        track.event = 'entering_leaving'
                        track.track_shape = 'outside_outside'
                        break
                if track.track_shape is None:
                    track.track_shape = 'outside'
            # Inside-outside 
            elif start < inside and end > outside: 
                track.track_shape = 'inside_out'
                track.event = 'leaving'
            # Ramp Ramp
            elif start > inside and start < outside and end >inside and end < outside :
                track.track_shape = 'ramp_ramp'
            # Ramp - Outside
            elif start> inside  and start < outside and end > outside: 
                track.track_shape = 'ramp_outside'
                track.event = 'leaving'
            # Ramp - Inside
            elif start < outside and start > inside and end <inside: 
                track.track_shape = 'ramp-inside'
            # Outside Inside 
            elif start > outside and end < inside: 
                track.track_shape = 'outside_inside'
                track.event = 'entering'
            # Outside Ramp
            elif start > outside  and end > inside and end < outside:
                track.track_shape = 'outside_ramp'
                track.event = 'entering'
            #Inside Ramp
            elif start < inside and end >inside and end < outside: 
                track.track_shape = 'inside_ramp'
        else:
            track.track_shape = 'noise'

        body = track.start
        pollen_count = 0
        pollen_sum = 0
        tags = list()
        while body.next is not None:
            if body.pollen:
                pollen_count += 1
            pollen_sum += body.pollen_score
            if body.tag != None:
                tags.append(body.tag)
            body = body.next

        if pollen_score == "mode":
            track.pollen_score = pollen_count/len(track)
        elif pollen_score == "average":
            track.pollen_score = pollen_sum/len(track)

        if tag == "mode":
            tids = list()
            for t in tags:
                tids.append(t["id"])
            track._tag = stats.mode(tids)



