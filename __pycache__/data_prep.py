import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List
from utils.face_mesh import *
import cv2
import os
mesh = FaceMesh()

class Data_Prep:
    def __init__(self) -> None:
        self.vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!1234567890 "]
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)

    def get_points(self, vid_path:str) -> List[int]:
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(25, 70))

        res, frame = cap.read()
        try:
            lm = mesh.get_landmarks(frame, vid_path)
            return lm[132][1], lm[132][2], lm[433][1], lm[378][2]
        except (TypeError, IndexError):
            print(f'Error in {vid_path}')
            lm = mesh.get_landmarks(frame, vid_path)
            return lm[132][1], lm[132][2], lm[433][1], lm[378][2]

        


    ##convert video to image frames and capture mouth movement
    def vid_frames(self, vid_path:str)  -> List[float]:
        x1,y1,x2,y2 = self.get_points(vid_path)
        
        cap = cv2.VideoCapture(vid_path)
        frames = []
        for vid_frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
            ret, frame = cap.read()
            try:
              trial=frame[y1:y2,x1-10:x2+10,:]
            except TypeError:
              print(vid_path)
              continue
            frame = frame[y1:y2,x1-10:x2+10,:]
            frame = cv2.resize(frame, (140, 46))
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame)
        cap.release()
        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        return tf.cast(frames, tf.float32)
    
    #take  written data from alignment folder and encode it
    def align_encode(self, path:str) -> List[str]:
        with open(path, 'r') as f:
            lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil': 
                tokens = [*tokens,' ',line[2]]
        return self.char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]  
    

    #Converts encoded data to letters. USED TO PREDICT RESULTS
    def decoder(self, enc_data:int) -> str:
        decode = [bytes.decode(x) for x in self.num_to_char(enc_data).numpy()]
        return "".join(decode)
    
    #Converts tensorflow encoded data to letters. USED TO PREDICT RESULTS
    def tf_decoder(self, enc_data:int) -> str:
            decoded = tf.keras.backend.ctc_decode(enc_data, input_length=[75], greedy=True)[0]
            return tf.strings.reduce_join([self.num_to_char(decoded)]).numpy()
    

    #Gets video file and align file with same name and folder
    def load_data(self, path: str): 
        #GET FILE NAME
        path = bytes.decode(path.numpy())
        filename = path.split('\\')[-1].split('.')[0]
        foldername = path.split('\\')[2]
        # names = path.split('/')
        # foldername = names[-2]
        # filename = names[-1].split('.')[0]
        video_path = os.path.join('data','videos', f'{foldername}',f'{filename}.mpg')
        
        ##LOAD DATA
        alignment_path = os.path.join('data','alignments',f'{foldername}',f'{filename}.align')
        frames = self.vid_frames(video_path) 
        alignments =  self.align_encode(alignment_path)
        
        return frames, alignments


    #tensorflow mappable function
    def mappable_function(self, path:str) ->List[str]:
        result = tf.py_function(self.load_data, [path], (tf.float32, tf.int64))
        return result


    #Combines above data preparation functions to one pipeline
    def data_pipeline(self, path:str, batch_size:int, split:float):    
        data = tf.data.Dataset.list_files(path)
        data = data.shuffle(500, reshuffle_each_iteration=False)
        data = data.map(self.mappable_function)
        data = data.padded_batch(batch_size, padded_shapes=([75,None,None,None],[40]))
        # data = data.prefetch(tf.data.AUTOTUNE)
        # data = data.cache()
        # return data
        train = data.take(int(len(data)*split)) 
        #everything after total data in s1+s2+s3/padded batch         
        val = data.skip(int(len(data)*split))
        return train, val



    # def get_data(self, df, split:float):
    #     #total data in s1+s2+s3/padded batch
    #     train = df.take(int(len(df)*split)) 
    #     #everything after total data in s1+s2+s3/padded batch         
    #     val = df.skip(int(len(df)*split))
    #     return train, val