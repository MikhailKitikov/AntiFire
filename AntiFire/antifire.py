import tkinter
from tkinter import messagebox, IntVar, Radiobutton, S
import cv2
import PIL.Image, PIL.ImageTk
import time
import argparse
import os
from keras import backend as K
import tensorflow as tf
from model_utils import *

 
# app class
class App:
    def __init__(self, window, window_title, base_model, video_source=0):
        self.window = window
        self.window.title(window_title)        
        self.window.bind('<Escape>', lambda e: self.quit())        
        self.video_source = video_source

        # open video stream
        self.vid = VideoStream(self.video_source)

        # create canvas
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # grid mode radiobutton
        self.enable_grid = IntVar()
        Radiobutton(self.window, text="Single", variable=self.enable_grid, value=0).pack(anchor=S)
        Radiobutton(self.window, text="Grid", variable=self.enable_grid, value=1).pack(anchor=S)
        Radiobutton(self.window, text="Special", variable=self.enable_grid, value=2).pack(anchor=S)

        # build model
        if base_model == 'mobilenet':
            self.model = load_mobilenetv2()
        elif base_model == 'nasnet':
            self.model = load_nasnetmobile()
        elif base_model == 'resnet':
            self.model = load_resnet50()
        elif base_model == 'firenet':
            self.model = load_FireNet()

        # build grid model
        self.grid_model = load_FireNetMobile()

        # build stack models
        self.stack_model = load_FireNetStack()
        self.stack_grid_model = load_FireNetMobileStack()
        self.lstm_model = load_LSTM()
        
        # variables
        self.classes = ['fire', 'normal']
        self.Q = deque(maxlen=5)
        self.stack_Q = deque(maxlen=10)
        self.cnt = 0
        self.cell_size = 64
        self.grid_size = 5
        self.stack_grid_size = 3
        
        # settings
        self.delay = 5
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.mainloop()

 
    def update(self):        
        ret, frame = self.vid.get_frame()
        if not ret:
            return

        output = frame.copy()
            
        # predict 
        if self.cnt % 10 == 0:           

            if self.enable_grid.get() == 0:    
                # preprocess
                frame = cv2.resize(frame, (224, 224)).astype('float32')
                frame = (frame - frame.mean()) / frame.std()
                
                # predict
                pred = self.model.predict(np.array([frame]))[0]
                self.Q.append(pred)
                self.label = 'normal' if np.mean(self.Q) > 0.5 else 'fire'

                # text
                text = "{}".format(self.label)
                text_color = (0, 255, 0) if self.label == 'normal' else (255, 0, 0)
                cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 5)

                # show
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(output))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)


            elif self.enable_grid.get() == 1: 
                # preprocess
                frame = cv2.resize(frame, (self.cell_size * self.grid_size, self.cell_size * self.grid_size)).astype('float32')
                frame = (frame - frame.mean()) / frame.std()

                delta_h = output.shape[0] // self.grid_size
                delta_w = output.shape[1] // self.grid_size

                to_draw = []

                # predict
                h, hh = 0, 0
                for i in range(self.grid_size):
                    w, ww = 0, 0
                    for j in range(self.grid_size):
                        pred = self.grid_model.predict(np.expand_dims(frame[h: h + self.cell_size, w: w + self.cell_size], axis=0))[0][0]
                        if pred < 0.5:
                            to_draw.append([(ww, hh), (ww + delta_w, hh + delta_h)])
                        w += self.cell_size
                        ww += delta_w
                    h += self.cell_size
                    hh += delta_h

                # draw
                for coords_x, coords_y in to_draw:
                    output = cv2.rectangle(output, coords_x, coords_y, (255, 0, 0), 3)

                # show
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(output))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)


            elif self.enable_grid.get() == 2:               
                # preprocess
                frame = cv2.resize(frame, (224, 224)).astype('float32')
                frame = (frame - frame.mean()) / frame.std()

                # predict whole
                prediction = list(self.stack_model.predict(np.expand_dims(frame, axis=0))[0])

                # predict grid
                y_pos = 0
                while y_pos < 224:
                    x_pos = 0
                    while x_pos < 224:
                        prediction.extend(list(self.stack_grid_model.predict(np.expand_dims(frame[x_pos: x_pos+64, y_pos: y_pos+64], axis=0))[0]))
                        x_pos += 80
                    y_pos += 80

                # add prediction to queue
                self.stack_Q.append(prediction)
                while len(self.stack_Q) < 10:
                    self.stack_Q.append(prediction)

                # final prediction
                arr = np.expand_dims(np.array(self.stack_Q), axis=0)
                pred = self.lstm_model.predict(arr)   
                self.label = 'normal' if pred > 0.5 else 'fire'             

                # text
                text = "{}".format(self.label)
                text_color = (0, 255, 0) if self.label == 'normal' else (255, 0, 0)
                cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 5)

                # show
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(output))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)


        self.cnt += 1
        self.window.after(self.delay, self.update)
        
    def quit(self):
        if self.vid:
            del self.vid
        print("[INFO] Stream closed")  
        self.window.destroy()
 
 
# stream class
class VideoStream:
    def __init__(self, video_source=0):
        # open video
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("[INFO] Unable to open video source", video_source)

        # get dimensions
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print("[INFO] Stream opened successfully") 
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# driver
if __name__ == '__main__':
    # start session
    NUM_PARALLEL_EXEC_UNITS = 4
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4,\
                           allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
    session = tf.compat.v1.Session(config=config)
    
    # parse command line arguments    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--stream", help="stream path")
        parser.add_argument("--model", help="base model")
        args = parser.parse_args()
        stream = 0
        
        if args.stream:
            stream = args.stream
            try:
                stream = int(stream)
            except:
                pass

        base_model = args.model if args.model else 'firenet'

        print("[INFO] Opening stream " + str(stream) + " ...")

        #  create window
        app = App(tkinter.Tk(), "AntiFire (© Mikhail Kitikov)", base_model=base_model, video_source=stream)   

    except Exception as e:
        print('[INFO] Stream failed (' + str(e) + ')')
