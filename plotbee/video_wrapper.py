import cv2
import numpy as np

class VideoCaptureWrapper():
    
    def __init__(self, video_path, start=0, stop=np.inf, step=1):
        self.video = cv2.VideoCapture(video_path)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        self.step = step
        self.stop = stop
        self.start = start
        self.i = start - 1
        
    def read(self):
        if self.i < self.stop:
            self.i += 1
            ret, frame = self.video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_id = self.i

            # if self.step > 1:
            #     self.i += self.step 
            #     self.video.set(cv2.CV_CAP_PROP_POS_FRAMES, self.i)
            
            for _ in range(self.step - 1):
                if self.i >= self.stop:
                    break
                self.video.read()
                self.i += 1 # update for the next query
            
            return frame_id, frame
        return False, None
    
    def release(self):
        self.video.release()