#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''UrbanSound8k Model (dataset was divided based on K-Folds with k=10, are at cluster 13 loc='rhea/UrbanSound8k/')'''


#Importing Libraries

import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#Libraries for Classification Models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Libraries needed for this project 
import os
import librosa
import librosa.display
import glob 
import skimage
import csv


# In[4]:


df = pd.read_csv("UrbanSound8k/UrbanSound8K_csv.csv")
df.head()


# In[5]:


print("Count in each class of urban sound 8k dataset=\n")
print(df['class'].value_counts())
print("\nNumber of training sets and val sets= ",df.shape[0],"\nNumber of classes= ",len(df['class'].unique()))


# In[1]:


#Feature Extraction and mel_spectogram function of librosa to extract the spectogram data as a numpy array
feature = []
label = []

def parser():
    # Function to load files and extract features
    for i in range(1,8732):
        file_name = 'UrbanSound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data , first extracting 3s per file....ie 22050*0.5=11025
        #applying segmentation:
        segment_1s=[X[ k*22050 : k*22050+22050 ] for k in range(int(len(X)/22050))]
        for audio_1s in segment_1s:
            mels = librosa.feature.melspectrogram(y=audio_1s, sr=sample_rate).T  
            feature.append(mels)
            label.append(df["classID"][i])
    return [feature,label]
    


# In[5]:


temp = parser()


# In[6]:


temp = np.array(temp)
data = temp.transpose()


# In[7]:


data.shape


# In[8]:


X_ = data[:, 0]
Y = data[:, 1]
#print(X_.shape, Y.shape)
X = np.empty([X_.shape[0],44,128])

print(X_.shape)
print(X_[0].shape)
print(X.shape)


# In[9]:


for i in range(len(X_)):
    X[i] = X_[i]


# In[10]:


X_[0].shape


# In[11]:


Y = to_categorical(Y,num_classes=10)


# In[12]:


# Final Data
print(X.shape)
print(Y.shape)


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)


# In[14]:


len(X_test)


# In[15]:


print(len(X_train))
print(len(Y_train))


# In[16]:


X_train = X_train.reshape(-1, 128, 44, 1)
X_test = X_test.reshape(-1, 128, 44, 1)


# In[17]:


len(Y_test)


# In[18]:


input_dim = (128, 44, 1)


# In[19]:



#CREATING A KERAS MODEL AND TESTING

model = Sequential()
model.add(Conv2D(64, (3, 3), padding = "same", activation = "relu", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dense(10, activation = "softmax"))


# In[20]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[21]:


model.summary()


# In[22]:


model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))


# In[23]:


#Testing on test data
predictions_1 = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)


# In[24]:


#testing on train data
predictions_2 = model.predict(X_train)
score = model.evaluate(X_train, Y_train)
print(score)


# In[25]:


#Priting confusion matrix to show accuracy 
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test.argmax(axis=1),predictions_1.argmax(axis=1))


# In[ ]:


##TESTING FOR CITY.WAV


# In[26]:


video_name='video6'


# In[28]:


city, srate= librosa.load('UrbanSound8k/video_testing/'+video_name+'.wav')
print(city.shape)


# In[29]:


#audio files name is city.wav that has to be used to predict on each sec
segments=[city[i*22050:i*22050+22050] for i in range(int(city.shape[0]/22050))]


# In[30]:


#segmenting the audio file to 1 sec each
feature_city=[]
for i in segments:
    mels = librosa.feature.melspectrogram(y=i, sr=srate).T       
    feature_city.append(mels)   


# In[31]:


print(len(feature_city))


# In[32]:


np.shape(feature_city[0])


# In[33]:


feature_city=np.array(feature_city)
print(feature_city.shape)


# In[34]:


feature_city= feature_city.reshape(-1, 128, 44, 1)
predictions_city = model.predict(feature_city)
print(predictions_city.shape)


# In[35]:


import math
def Top3labels(listt):
    first = -math.inf
    second = -math.inf
    third =-math.inf
    fi=0;si=0;ti=0
    for i in range(0, len(listt)): 
        if (listt[i] > first): 
            third = second
            ti=si 
            second = first 
            si=fi
            first = listt[i]
            fi=i
        elif (listt[i] > second): 
            third = second 
            ti=si
            second = listt[i]
            si=i
        elif (listt[i] > third): 
            third = listt[i] 
            ti=i
    index_labels=[fi,si,ti]
    value_labels=[first,second,third]
        
    #print(index_labels)
    #print(value_labels)
    return [index_labels , value_labels]


# In[36]:


index_labels=[]  #stores the index of the max 3 
value_labels=[]  #stores the values of the hot encoding of those max 3

for class_values in predictions_city:
    class_values1 = class_values.tolist()
    tdl=Top3labels(class_values1)
    index_labels.append(tdl[0])
    value_labels.append(tdl[1])


for i in value_labels:
    #print(i)
    norm = np.linalg.norm(i)
    normal_array = i/norm
    #print(normal_array)
    

for i in range(len(index_labels)):
    print(index_labels[i]," = ",value_labels[i])


# In[ ]:





# In[50]:


major_classes=['air_conditioner', 'animal_sounds','car_horn', 'drilling', 'human_sounds', 'gun_shot','human_sounds','street_music' , 'siren','jackhammer']


select_classes=[]

for i in index_labels:
    select_j=[]
    for j in i:
        select_j.append(major_classes[j])
    select_classes.append(select_j)

#for i in select_classes:
#    print(i)


# In[51]:


select_classes


# In[52]:


value_labels


# In[53]:


#writing selected_class_labels into a csv
  
import csv      
# field names  
fields = ['first', 'second', 'third']  
    
# data rows of csv file  
rows =select_classes
    
# name of csv file (uncomment the next line if you want to rewrite) 
filename = 'UrbanSound8k/video_testing_csv/class_'+video_name+'_labels.csv'
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)          
    # writing the fields  
    csvwriter.writerow(fields)          
    # writing the data rows  
    csvwriter.writerows(rows) 


# In[54]:


#writing selected_class_label's value_counts(probablity) into a csv
  
import csv      
# field names  
fields = ['first', 'second', 'third']  
    
# data rows of csv file  
rows =value_labels
    
# name of csv file  (uncomment the next line if you want to rewrite) 
filename = 'UrbanSound8k/video_testing_csv/value_'+video_name+'_plot.csv'

    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)          
    # writing the fields  
    csvwriter.writerow(fields)          
    # writing the data rows  
    csvwriter.writerows(rows) 


# In[ ]:





# In[55]:


main_labels=[]
main_values=[]

for i in select_classes:
    for j in i:
        main_labels.append(j)
        

for i in value_labels:
    for j in i:
        main_values.append(j)


# In[56]:


value_labels


# In[57]:


#to store graphs in the direcory location
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import os
i=0
itr=0
directory='UrbanSound8k/video_testing_plots/plot_'+video_name+'_graph/'
if not os.path.exists(directory):
    os.makedirs(directory)
while(i<len(main_labels)):  
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar([main_labels[i],main_labels[i+1],main_labels[i+2]],[main_values[i],main_values[i+1],main_values[i+2]])
    ax.set_yticks(np.arange(0.0, 1.0, 0.2))
    plt.savefig(os.path.join(directory,"plot_"+str(itr)+".jpg"),bbox_inches='tight')
    itr+=1
    i=i+3
    
    #plt.show()
    #print(i)
    
    
    


# In[58]:


import numpy as np
import cv2
import os
import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

cap = cv2.VideoCapture('UrbanSound8k/video_testing/'+video_name+'.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False): 
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Find FPS
fps = cap.get(cv2.CAP_PROP_FPS)
#fourcc = cv2.cv.CV_FOURCC(*'XVID')


fourcc = cv2.VideoWriter_fourcc(*'MP4V')

out = cv2.VideoWriter('UrbanSound8k/video_testing/output_'+video_name+'.mp4',fourcc, fps, (frame_width,frame_height))
#fig, ax = plt.subplots(1,1)

font = cv2.FONT_HERSHEY_TRIPLEX
org = (50, 50) 
fontScale = 1
color = (138, 43, 226) 
thickness = 2

i = 0
fig = plt.figure()
counter = 0
itr=0

while(True):
    ret, frame = cap.read()
    counter = counter + 1
    if ret == True: 
        if(counter % 23 == 0):
            i=i+3
            itr+=1
        print(i)
        if(i>=len(main_labels)):
            break
        #frame = cv2.putText(frame, 'Class='+main_labels[i]+', '+main_labels[i+1]+', '+main_labels[i+2], org, font, fontScale, color, thickness, cv2.LINE_AA)
        h,w = frame.shape[:2]
        plot_image=cv2.imread('UrbanSound8k/video_tesing_plots/plot_'+video_name+'_graph/plot_'+str(itr)+'.jpg')
        h1, w1 = plot_image.shape[:2]
        #set top left position of the resized image
        pip_h = 10
        pip_w = 10
        frame[pip_h:pip_h+h1,pip_w:pip_w+w1] = plot_image  # make it PIP
        out.write(frame)
        #cv2.imwrite(r""+itr+".png",frame)
        #cv2.imwrite(r'.\plot_'+itr+'.png',img)
        print(itr)
    else:
        break
cap.release()
out.release()


cv2.destroyAllWindows() 
   


# In[ ]:




