import numpy as np
import cv2
import sys
sys.path.append("Neuropl")
import neuropl
import time

class SingleModel:
  model = None
  input_type = np.uint8 # manually specified by client
  output_type = np.float # manually specified by client
  input_shape = [[1,300,300,3]] # manually specified by client
  output_shape = [[1,1917,21], [1,1917,4]] # manually specified by client

  def __init__(self, model_path, input_shape, output_shape, input_type, output_type):
    self.model = neuropl.Model(model_path) # .dla
    self.input_shape, self.output_shape = input_shape, output_shape
    self.input_type, self.output_type = input_type, output_type

  def cropFrameToSquare(self, frame):
    h, w, _ = frame.shape
    target_len = min(h,w)
    start_x, start_y = (w - target_len)//2, (h - target_len)//2
    return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]

  def predictFrame(self, frame):
    # match model input shape
    start = time.time()
    frame = self.cropFrameToSquare(frame)
    frame = cv2.resize(frame,(self.input_shape[0][1], self.input_shape[0][2]))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
    frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size
    # match model input type
    input = frame_rgb.astype(self.input_type)

    # predict
    ans = self.model.predict(input) # [output arr 1, output arr 2, ...]
    self.model.print_profiled_qos_data()
    
    end = time.time()
    timediff = end-start
    return ans, timediff
