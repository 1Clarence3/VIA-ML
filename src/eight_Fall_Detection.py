import numpy as np
import cv2
import sys
sys.path.append("./../Neuropl")
import neuropl
import time

class FallDet:
    # hard coded parameters for fall detection
    model1_input_shape = [1,224,224,3]
    model1_output_shape = [1,2]
    model2_input_shape = [1,8]
    model2_output_shape = [1,1]
    input_type = np.uint8
    threshold = 130

    model1 = None
    model2 = None
    probs = []
    ### added for diff filter
    prev_frame_gray = None
    is_first_frame = True
    diff_threshold = 43
    #########################
    
    def __init__(self):
        self.model1 = neuropl.Model("./../dla_models/eight_m1.dla") # model1 in: uint8 (1x224x224x3) out: uint8 (1x2)
        self.model2 = neuropl.Model("./../dla_models/eight_m2.dla") # model2 in: uint8 (1x16) out: uint8 (1x1)
        self.probs = []
        ### added for diff filter
        prev_frame_gray = None
        is_first_frame = True
        #########################
        
    def cropFrameToSquare(self, frame):
        h, w, _ = frame.shape
        target_len = min(h,w)
        start_x, start_y = (w - target_len)//2, (h - target_len)//2
        return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]

    def predictFrame(self, frame):
        # match model input shape
        start = time.time()
        frame = self.cropFrameToSquare(frame)
        frame = cv2.resize(frame, (self.model1_input_shape[1], self.model1_input_shape[2]), interpolation=cv2.INTER_AREA) # resize frame to 224x224
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size [224x224x3] -> [1x224x224x3]
        # match model input type
        frame_rgb = frame_rgb.astype(self.input_type)

        ### added for diff filter
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.is_first_frame:
            self.prev_frame_gray = curr_frame_gray
            self.is_first_frame = False
        frame_diff = cv2.absdiff(self.prev_frame_gray, curr_frame_gray)
        frame_avg_diff = np.sum(frame_diff) / (self.model1_input_shape[1]*self.model1_input_shape[2])
        print(frame_avg_diff)
        #########################

        # predict using neuropl
        output_data = self.model1.predict(frame_rgb)[0] # API outputs [[movingprob, stillprob]]

        output_probs = np.exp(output_data.astype(float))/np.sum(np.exp(output_data.astype(float))) # without tf
        predicted_index = np.argmax(output_data)
        class_labels = ["Moving", "Still"]
        predicted_class = class_labels[predicted_index]

        prob = np.around(max(output_probs), decimals = 2) # without tf
        if predicted_class == "Still": self.probs += [1-prob]
        ### added for diff filter
        elif frame_avg_diff < self.diff_threshold: self.probs += [1-prob] # small value
        #########################
        else: self.probs += [prob]

        result = False # default = not falling
        print(len(self.probs))
        if len(self.probs) == 8: # go to 2nd model if collected 16 frames
            result = self.predictVid()
            self.probs = self.probs[1:]
        
        end = time.time()
        timediff = end-start
        return result, timediff
    
    def predictVid(self): # only called by predictFrame(self, frame)
        model2_in = np.array(self.probs).reshape((1, 8))
        vid_preds = self.model2.predict(model2_in)[0] # [[uint8]]
        self.model2.print_profiled_qos_data()
        print(f"predicted val = {vid_preds[0]}")
        return (vid_preds.reshape((1, len(vid_preds))) > self.threshold)[0][0] # bool
