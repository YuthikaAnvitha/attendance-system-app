import numpy as np
import pandas as pd
import cv2
import os

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

#time 
import time
from datetime import datetime

# Connect to Redis Client
hostname='redis-14346.c261.us-east-1-4.ec2.redns.redis-cloud.com'
port=14346
password='75dtohB74B9ja7IPxVmkr8qsVSrH76BQ'

r = redis.StrictRedis(host=hostname,
                      port=port,
                      password=password)

#Retrieve data from database

def retrieve_data(name):
    retrieve_dict=r.hgetall(name)
    retrieve_series=pd.Series(retrieve_dict)
    retrieve_series=retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index=retrieve_series.index
    index=list(map(lambda x: x.decode(),index))
    retrieve_series.index=index
    retrieve_df=retrieve_series.to_frame().reset_index()
    retrieve_df.columns=['name_role','facial_features']
    retrieve_df[['Name','Role']]=retrieve_df['name_role'].apply(lambda x:x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Role','facial_features']]



# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['Name','Role'],thresh=0.5):
    dataframe=dataframe.copy()
    x_list=dataframe[feature_column].tolist()
    x=np.asarray(x_list)

    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe['coisine']=similar_arr

    data_filter=dataframe.query(f'coisine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['coisine'].argmax()
        name_cos,role_cos = data_filter.loc[argmax][name_role]
        
    else:
        name_cos= 'Unknown'
        role_cos= 'Unknown'
    
    return name_cos,role_cos

class RealTimePred():
    def __init__(self):
        self.logs=dict(name=[],role=[],current_time=[])
        
        
    def reset_dict(self):
        self.logs=dict(name=[],role=[],current_time=[])
        
    def saveLogs_redis(self):
        #step1: create a logs dataframe
        dataframe=pd.DataFrame(self.logs)
        #step2: drop the duplicate info
        dataframe.drop_duplicates('name',inplace=True)
        #step3: push data to redis datbase(list)
        #encode the data
        name_list=dataframe['name'].tolist()
        role_list=dataframe['role'].tolist()
        ctime_list=dataframe['current_time'].tolist()
        encoded_data=[]
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name!='Unknown':
                concat_string = f'{name}@{role}@{ctime}'
                encoded_data.append(concat_string)
                
        if len(encoded_data) >0:
            r.lpush('attendance:logs',*encoded_data)
            
        self.reset_dict()
                    
    def face_prediction(self,test_image, dataframe,feature_column,name_role=['Name','Role'],thresh=0.5):
        #step1: take the test image and apply insightface 
        current_time=str(datetime.now())
        
        results=faceapp.get(test_image)
        test_copy=test_image.copy()
        
        #step2:use for loop and extract each embedding and pass it to ml_search algorithm
        for res in results:
            x1,y1,x2,y2= res['bbox'].astype(int)
            embeddings=res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if person_name=='Unknown':
                color=(0,0,255)
            else:
                color=(0,255,0)
            
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen= person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            #save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
            
        return test_copy


###Registration form
#get results from insightface model

class RegistrationForm():
    def __init__(self):
        self.sample = 0
        
    def reset(self):
        self.sample = 0
        
    def get_embedding(self,frame):
        results=faceapp.get(frame,max_num=1)
        embeddings=None
        for res in results:
            self.sample+=1
            x1,y1,x2,y2= res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            text=f'samples = {self.sample}'
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            embeddings=res['embedding']
            
        return frame,embeddings
    
    def save_data_in_redis_db(self,name,role):
        #validation name
        if name is not None:
            if name.strip()!='':
                key=f'{name}@{role}'
                
            else:
                return 'name_false'
            
        else:
            return 'name_false'
        
        #if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
            
        
        #step-1: load "face_embedding.txt"
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32)
        
        #step-2: convert into array(proper shape)
        received_samples=int(x_array.size/512)
        x_array= x_array.reshape(received_samples,512)
        x_array= np.asarray(x_array)
        
        
        #step-3:cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean= x_mean.astype(np.float32)
        x_mean_bytes= x_mean.tobytes()
        
        #step-4: save this into redis database
        #redis hashes
        r.hset(name='academy:register',key=key,value=x_mean_bytes)
        
        
        os.remove('face_embedding.txt')
        self.reset()
        
        return True
