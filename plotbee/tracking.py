from scipy.optimize import linear_sum_assignment as hungarian
import numpy as np
from plotbee.sort import Sort, KalmanBoxTracker
from plotbee.track import Track
from plotbee.body import Body
from collections import defaultdict
from tqdm import tqdm


class IntegerIDGen():

    def __init__(self):
        self.next_available_id = 0

    def __call__(self):
        i = self.next_available_id
        self.next_available_id += 1
        return i

def body_distance(body_a, body_b):
    x_a, y_a = body_a.center
    x_b, y_b = body_b.center

    return np.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)

def cost_matrix_tracks_skeleton(frame, next_frame, threshold):
    total = len(frame)+len(next_frame)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(frame) and j <len(next_frame):
                    cost_m[i][j] = body_distance(frame[i], next_frame[j])
            else:
                cost_m[i][j] = threshold

    return cost_m

def body_cbox_overlaping_ratio(body_a, body_b):
    x1_i, y1_i, x2_i, y2_i = body_a.cbox()
    x1_j, y1_j, x2_j, y2_j = body_b.cbox()

    area_i = (x2_i - x1_i + 1) * (y2_i - y1_i + 1)

    intersec_x1 = max(x1_i, x1_j)
    intersec_y1 = max(y1_i, y1_j)

    intersec_x2 = min(x2_i, x2_j)
    intersec_y2 = min(y2_i, y2_j)

    w = max(0, intersec_x2 - intersec_x1 + 1)
    h = max(0, intersec_y2 - intersec_y1 + 1)

    overlap_ratio = float(w*h)/area_i

    return overlap_ratio

def non_max_supression_video(video, overlapThreshold):
    for frame in video:
        non_max_supression(frame, overlapThreshold)
    return




def non_max_supression(frame, overlapThreshold):
    """
    This function filter out overlaping bodies using a Overlapping threshold.

    """

    # Sort bees by y2
#     sorted_bodies = sorted(frame, key=lambda body: body.cbox()[-1])
    
    num_bodies = len(frame)
    
    for i in range(num_bodies):
        body_a = frame[i]
        
        if body_a.suppressed:
            continue
            
        for j in range(i + 1, num_bodies):
            
            body_b = frame[j]
            
            if body_b.suppressed:
                continue
                
            overlap_ratio = body_cbox_overlaping_ratio(body_a, body_b)

            if overlap_ratio > overlapThreshold:
                body_b.suppressed = True
        
    return


def hungarian_tracking(video, cost=200, nms_overlap_fraction=0.6):


    getId = IntegerIDGen()
    # Supress bodies
    non_max_supression_video(video, nms_overlap_fraction)

    video._tracks = {}  # Init tracks

    for i, body in enumerate(video[0].valid_bodies):
        body.set_id(getId())
        video._tracks[body.id] = Track(body)
#             print(body)


    for i in tqdm(range(len(video) - 1)):
        current_frame = video[i].valid_bodies
        next_frame = video[i + 1].valid_bodies

        cmap = cost_matrix_tracks_skeleton(current_frame, next_frame, cost)
        _, idx = hungarian(cmap)

        for j in range(len(current_frame)):
            if cmap[j,idx[j]]<cost:

                # Create New ID
                if current_frame[j].id == -1:

                    current_frame[j].set_id(getId())
                    video._tracks[current_frame[j].id] = Track(current_frame[j])

                # Match Next Frame Detections
                next_frame[idx[j]].set_id(current_frame[j].id)
                next_frame[idx[j]].prev = current_frame[j]
                current_frame[j].next = next_frame[idx[j]]
    return


def matchIds(bboxes, predbboxes):
    a = (bboxes[:, 0:2] + bboxes[:, 2:4])/2
    b = (predbboxes[:, 0:2] + predbboxes[:, 2:4])/2

    m = np.zeros((bboxes.shape[0], predbboxes.shape[0]))

    for i, p1 in enumerate(a):
        for j, p2 in enumerate(b):
            d = np.sqrt(np.sum((p1 - p2)**2))
            m[i, j] = d
            
    bbox_ids, pred_ids = hungarian(m)
    # print(predbboxes.shape, pred_ids.shape)
    return bbox_ids, predbboxes[pred_ids, 4]


def sort_tracking(video, bbox=200, nms_overlap_fraction=0.6):
    # getId = IntegerIDGen()
    # Supress bodies
    non_max_supression_video(video, nms_overlap_fraction)
    mot_tracker = Sort()
    KalmanBoxTracker.count=0
    prev_track = defaultdict(lambda: None)

    for frame in video:
        valid_bodies = [body for body in frame if not body.suppressed]
        bboxes = [body.cbox(bbox, bbox) for body in valid_bodies]
        bboxes = np.array(bboxes)

        predbboxes = mot_tracker.update(bboxes)
        # if(predbboxes.shape[0] != bboxes.shape[0]):
        #     print(predbboxes.shape, bboxes.shape)

        bodiesIds, predIds = matchIds(bboxes, predbboxes) 

        for i, body in zip(predIds, bodiesIds):
            body = valid_bodies[body]
            body.set_id(int(i))
            # print(box_id)
            

            # Update Track LinkList DataStructure
            body.prev = prev_track[body.id]

            if prev_track[body.id] is not None:
                prev_track[body.id].next = body
            else:
                video._tracks[body.id] = Track(body)


            prev_track[body.id] = body

    return




from scipy.optimize import linear_sum_assignment as hungarian

def nms_euclid(det, min_dist):
    """
    det is Nx3 array, each row [x,y,score,...]
    
    return bool vector, True if keep, False if suppress
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
    
    
def hungarian_tracking_with_prediction(video, max_dist=100, part='2', nms_min_dist=50, 
                                       max_step=4, min_track_length=1,
                                       decay_alpha=0.5, 
                                       debug=False, progress=True):
    """
    This function track detections in time by matching prediction with detections using the hungarian algorithm. 
    It is expecting to receive an instance of detections. Please check documentation for data structure. 
    
    Inputs: 
        - video : Video datastructure from which to extract the detections
        - max_dist : Maximum euclidean distance allowed
        NOT USED - part : Id of the part to use as reference keypoint in a body (default thorax '2')
        NOT USED - nms_min_dist : Distance below which non-maximum keypoints are suppressed
        - max_step : largest frame step since last observed detection (1 does not allow any gap)
        - min_track_length : minimum number of nodes to keep a track (1 keep all tracks)
        - decay_alpha : exponential decay for velocity estimation
    
    Return:
        - track_raw data structure indexed as track_raw[trackid][fieldname]
          where fieldname in startframe,endframe,data...
          and track_raw[trackid]['data'][frame] is a dict 
          with fields detid, pos,score,vel,cost, predictpos,predictvel
    
    """
    
    def dprint(*args):
        if (debug):
            print(*args)
            
    def get_detections_and_bodies(frameObj):
        allbodies = frameObj.bodies
        bodies = frameObj.valid_bodies
        dets = np.zeros( (len(bodies),5) )  # x,y,score,detid, angle
        k=0
        for dk,body in enumerate(allbodies):
            if (body.suppressed): continue
            C = body.center
            angle = body.angle
            dets[k,:] = (C[0], C[1], -1, dk, angle)  # Caution: k is within valid_bodies, dk within all bodies
            k+=1
        return dets, bodies
    
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
    tracker_state = np.zeros( (0,5) )  # [x,y,vx,vy,angle]
    
    video._tracks = {}  # Init tracks
    for frameObj in tqdm(video):  # Remove all preexisting virtual bodies
        frameObj.delete_virtual_bodies()
    
    print("Tracking...")    
    
    # FOR EACH FRAME:
    #for i in tqdm(range(len(keylist)), disable=~progress):
    #for i in range(len(keylist)):
    for frameObj in tqdm(video):
        frame_id = frameObj.id
        
        for body in frameObj.bodies:
            body.set_id(-1)  # Reset all track ids, including suppressed bodies
        
        framedelta = 0 if (frame_prev==-1) else frame_id-frame_prev
        frame_prev=frame_id
        #dprint(f"### FRAME {frame_id}, delta={framedelta}")
        
        #dprint(f'activetracks {activetracks}')
        #dprint(f'tracker_state {tracker_state}')
        
        # 0. DELETE LOST TRACKS
        keepactive=np.zeros_like(activetracks,dtype=bool)
        
        for j,tj in enumerate(activetracks): 
            T = track_raw[tj]  # Existing track
            gap = frame_id-T['endframe']
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
        #tracker_state[:,2:4] = tracker_state[:,2:4]  # vx,vy
        #tracker_state[:,4] = tracker_state[:,4]      # a
        #dprint(f'tracker_state {tracker_state}')
        
        # 2. GATHER OBSERVATIONS
        # Get all thoraxes (or reference part defined by `part`)
        dets, bodies = get_detections_and_bodies(frameObj)   # Array with each row as [x,y,score,dk,angle]
        # NMS
        #if (nms_min_dist is not None):
        #    keep = nms_euclid(dets0, nms_min_dist)
        #    dets = dets0[keep,...]
        #    #dprint(f'nms delete [{np.nonzero(~keep)}]')
        #else:
        #    dets=dets0
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
            predictangle = tracker_state[j,4].tolist()
            
            # Matched existing track tj with detection k
            k=idx[j]  # column k in cmap == id k within NMS detections
            matching_dist=cmat[j,k]
            if (matching_dist<max_dist): # MATCHED track tj to detection k
                dk = detids[k]  # id dk within all detections for frame key
                body = bodies[k]
                
                body.set_id(tj) # Assign this body to track tj
                
                #dprint(f'Track {tj} (Active {j}) += Det {dk}  (NMS {k})  pos {dets[k,0:2]}')
                
                # Insert intermediate virtual detections
                T['data'].update(T['virtual'])
                T['virtual']={}
                
                # Update from observation
                # Recursive estimate of velocity with exponential decay (use existinting state)
                tracker_state[j,2:4] = (1.0-decay_alpha)*tracker_state[j,2:4]+decay_alpha*(dets[k,0:2]-tracker_state[j,0:2])/framedelta # [vx,vy]
                # Update position to observation
                tracker_state[j,0:2] = dets[k,0:2] # [x,y]
                
                angle = dets[k,4]
                tracker_state[j,4] = angle # [angle]
                
                T['endframe']=frame_id  # Last frame with a matching detection
                T['nbdetections']+=1
                T['data'][frame_id]=dict(body=body,detid=dk, pos=dets[k,0:2].tolist(), angle=angle, score=dets[k,2], 
                                      cost=matching_dist, vel=tracker_state[j,2:4].tolist(),
                                      predictpos=predictpos, predictvel=predictvel, predictangle=predictangle)

            else: # NO MATCH for track tj
                # Still alive, but need to create a virtual detection
                x,y=predictpos[0],predictpos[1]
                aa=predictangle/180*np.pi
                body = Body({1:[(int(x-np.sin(aa)*100),int(y-np.cos(aa)*100))], 3: [(x,y)]}, center=3,
                            connections=[],angle_conn=[1,3], frame=frameObj, 
                            body_id=tj, suppressed=False, pollen_score=0.0, 
                            tag=None, features=None,
                            virtual=True)
                angle = body.angle
                T['virtual'][frame_id]=dict(body=body,detid=-1, pos=predictpos, angle=angle, score=-1, cost=-1, vel=predictvel,
                                     predictpos=predictpos, predictvel=predictvel, predictangle=predictangle)
                #dprint(f'Track {tj} (Active {j}) += Virtual Det    pos {tracker_state[j,0:2]}')
            
        # 6. CREATE NEW TRACKS FOR UNMATCHED DETECTIONS
        #dprint(f'unmatched_det {unmatched_det}')
        for k in unmatched_det:
            dk=detids[k]
            body = bodies[k]
            
            angle = body.angle
            
            lasttrackid+=1
            T = {}; track_raw[lasttrackid]=T
            T['id']=lasttrackid
            T['startframe'] = frame_id
            T['endframe'] = frame_id
            T['nbdetections'] = 1
            T['data'] = {}
            T['data'][frame_id] = dict(body=body, detid=dk, pos=dets[k,0:2].tolist(), angle=angle, score=dets[k,2], cost=0, vel=[0.0,0.0])
            T['virtual'] = {}
            
            # Append to active tracks
            activetracks.append(lasttrackid)
            state_k = np.array([[dets[k,0], dets[k,1], 0.0, 0.0, dets[k,4]]])
            tracker_state = np.append(tracker_state, state_k, axis=0)
            #print(tracker_state)
            
            #dprint(f'New Track {lasttrackid} (Active {len(activetracks)-1}) += Det {dk} (NMS {k}) pos {dets[k,0:2]}')
        nactive = len(activetracks)
         
    # CLEANUP
    for _,tj in enumerate(activetracks):
        close_track(track_raw, tj, min_track_length=min_track_length)
        # Do not clean tracker_state as it is not used anymore
        
    print("Convert to Track format")    
    
    # Convert track_raw to Track
    for tj,T in track_raw.items():
        #tj = T['id']
        data = T['data']
        body = data[T['startframe']]['body']
        body.set_id(tj)
        TT = Track( body ) # Init with first body
        video._tracks[tj] = TT
        #TT.id = tj   # Already set in Track.__init__
        for frame_id in range(T['startframe']+1,T['endframe']+1):
            body = data[frame_id]['body']
            TT[frame_id] = body
            
            if (body.virtual): # Add virtual bodies to frameObj
                frameObj = video._get_frame(frame_id)
                frameObj.update([body])
            
    return track_raw