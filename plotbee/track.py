import bisect

class Track():
    
    def __init__(self, body):
        self._start = body  # marked for obsolescence
        self._end = body    # marked for obsolescence
        
        # New data structure: explicit dict track[frameid] -> body at frameId
        self.id = body.id
        self.startframe = body.frameid
        self.endframe = body.frameid
        self._data = {body.frameid: body}
        
        self._event = None
        self._track_shape = None   # What for?
        self.pollen_score = 0.0
        self._tag = None

    @property
    def tag(self):
        return self._tag

    @property
    def pollen(self):
        if self.pollen_score > 0.5:
            return True
        return False

#  Now baked in the Track fields
#     @property
#     def id(self):
#         return self._start.id

    @property
    def tag_id(self):
        if self.tag is None:
            return None
        else:
            return self.tag["id"]

    @property
    def end(self):
        return self[self.endframe]
#         if self._end.next is None:
#             return self._end
#         while self._end.next is not None:
#             self._end = self._end.next
#         return self._end

    def params(self):
        p = {
            "event": self._event,
            "track_shape": self._track_shape,
            "pollen": self.pollen
        }

        return p

    @property
    def start(self):
        #return self._start
        return self[self.startframe]

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, value):
        self._event = value

    @property
    def track_shape(self):
        return self._track_shape

    @track_shape.setter
    def track_shape(self, value):
        self._track_shape = value
    

    def __len__(self):
        return self.endframe-self.startframe+1
#         self._size = 1
#         x = self._start
#         while x.next is not None:
#             x = x.next
#             self._size += 1
#         return self._size
    
    def __getitem__(self, index):
        return self._data[index]
#         if index < self._size:
#             x = self._start
#             for _ in range(index):
#                 x = x.next
#             return x
#         else:
#             raise IndexError("Index out of range.")

    def __setitem__(self, index, body):
        if (index < self.startframe-1 or index > self.endframe+1):
            raise IndexError("Index out of range, can only set to +/- 1 of existing range.")
        self._data[index] = body
        if (index < self.startframe): 
            self.startframe = index
        if (index > self.endframe): 
            self.endframe = index
        body.set_id(self.id) # Enforce consistency between Track and Body
            
        # Marked for obsolescence: update _start and _end
        if (hasattr(self, '_start')):
            if (index == self.startframe):
                self._start = body
            if (index == self.endframe):
                self._end = body
        # Marked for obsolescence: update next and prev in the body nodes
        if (hasattr(body, '_prev')):
            if (body is not self._start):
                body._prev = self._data[index-1]
                self._data[index-1]._next = body
            if (body is not self._end):
                body._next = self._data[index-1]
                self._data[index+1]._prev = body

    def __iter__(self):
        for index in range(self.startframe,self.endframe+1):
            yield self[index]
#         x = self._start

#         yield x

#         while x.next is not None:
#             x = x.next
#             yield x
    
    def __repr__(self):
        return "Track({}, len={})".format(self.id, len(self))