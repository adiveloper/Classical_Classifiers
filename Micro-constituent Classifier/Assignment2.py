#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG16(weights="imagenet", include_top = False)
Model1 = Model(inputs = base_model.input, outputs = base_model.get_layer("block1_pool").output)
Model2 = Model(inputs = base_model.input, outputs = base_model.get_layer("block2_pool").output)
Model3 = Model(inputs = base_model.input, outputs = base_model.get_layer("block3_pool").output)
Model4 = Model(inputs = base_model.input, outputs = base_model.get_layer("block4_pool").output)
Model5 = Model(inputs = base_model.input, outputs = base_model.get_layer("block5_pool").output)

Model1.summary()


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv("/Users/moeinrazavi/Educational/TAMU Courses/Pattern Recognition/Final Project/Data/micrograph.csv")


# In[4]:


pearlite = data[data.primary_microconstituent =="pearlite"]
train_pearlite = pearlite[0:100]
test_pearlite = pearlite[100:]

spheroidite = data[data.primary_microconstituent =="spheroidite"]
train_spheroidite = spheroidite[0:100]
test_spheroidite = spheroidite[100:]

network = data[data.primary_microconstituent =="network"]
train_network = network[0:100]
test_network = network[100:]




train_ps = np.concatenate((train_pearlite, train_spheroidite))
test_ps = np.concatenate((test_pearlite, test_spheroidite))

train_pn = np.concatenate((train_pearlite, train_network))
test_pn = np.concatenate((test_pearlite, test_network))

train_sn = np.concatenate((train_spheroidite, train_network))
test_sn = np.concatenate((test_spheroidite, test_network))

train_set = np.concatenate((train_pearlite, train_spheroidite, train_network))
test_set = np.concatenate((test_pearlite, test_spheroidite, test_network))


# In[5]:


path_train_ps = train_ps[:,1]
y_train_ps = train_ps[:,-1]
path_test_ps = test_ps[:,1]
y_test_ps = test_ps[:,-1]

path_train_pn = train_pn[:,1]
y_train_pn = train_pn[:,-1]
path_test_pn = test_pn[:,1]
y_test_pn = test_pn[:,-1]

path_train_sn = train_sn[:,1]
y_train_sn = train_sn[:,-1]
path_test_sn = test_sn[:,1]
y_test_sn = test_sn[:,-1]

path_train = train_set[:,1]
y_train = train_set[:,-1]
path_test = test_set[:,1]
y_test = test_set[:,-1]


# In[6]:


img_train_ps = [0 for i in range(len(train_ps))]
img_test_ps = [0 for i in range(len(test_ps))]
x_train_ps = [0 for i in range(len(train_ps))]
x_test_ps = [0 for i in range(len(test_ps))]

img_train_pn = [0 for i in range(len(train_pn))]
img_test_pn = [0 for i in range(len(test_pn))]
x_train_pn = [0 for i in range(len(train_pn))]
x_test_pn = [0 for i in range(len(test_pn))]

img_train_sn = [0 for i in range(len(train_sn))]
img_test_sn = [0 for i in range(len(test_sn))]
x_train_sn = [0 for i in range(len(train_sn))]
x_test_sn = [0 for i in range(len(test_sn))]

img_train = [0 for i in range(len(train_set))]
img_test = [0 for i in range(len(test_set))]
x_train = [0 for i in range(len(train_set))]
x_test = [0 for i in range(len(test_set))]


train_features_ps_M1 = [0 for i in range(len(train_ps))]
train_features_pn_M1 = [0 for i in range(len(train_pn))]
train_features_sn_M1 = [0 for i in range(len(train_sn))]
train_features_M1 = [0 for i in range(len(train_set))]

train_features_ps_M2 = [0 for i in range(len(train_ps))]
train_features_pn_M2 = [0 for i in range(len(train_pn))]
train_features_sn_M2 = [0 for i in range(len(train_sn))]
train_features_M2 = [0 for i in range(len(train_set))]

train_features_ps_M3 = [0 for i in range(len(train_ps))]
train_features_pn_M3 = [0 for i in range(len(train_pn))]
train_features_sn_M3 = [0 for i in range(len(train_sn))]
train_features_M3 = [0 for i in range(len(train_set))]

train_features_ps_M4 = [0 for i in range(len(train_ps))]
train_features_pn_M4 = [0 for i in range(len(train_pn))]
train_features_sn_M4 = [0 for i in range(len(train_sn))]
train_features_M4 = [0 for i in range(len(train_set))]

train_features_ps_M4 = [0 for i in range(len(train_ps))]
train_features_pn_M4 = [0 for i in range(len(train_pn))]
train_features_sn_M4 = [0 for i in range(len(train_sn))]
train_features_M4 = [0 for i in range(len(train_set))]

train_features_ps_M5 = [0 for i in range(len(train_ps))]
train_features_pn_M5 = [0 for i in range(len(train_pn))]
train_features_sn_M5 = [0 for i in range(len(train_sn))]
train_features_M5 = [0 for i in range(len(train_set))]



for i in range(len(train_ps)):

    img_train_ps[i] = image.load_img("./Data/micrograph/" + path_train_ps[i], target_size=(224, 224))
    x_train_ps[i] = image.img_to_array(img_train_ps[i])
    x_train_ps[i] = x_train_ps[i][0:484,:,:] # crop the bottom subtitles
    x_train_ps[i] = np.expand_dims(x_train_ps[i], axis=0)
    x_train_ps[i] = preprocess_input(x_train_ps[i])
    
for i in range(len(train_ps)):
    train_features_ps_M1[i] = Model1.predict(x_train_ps[i])
    train_features_ps_M2[i] = Model2.predict(x_train_ps[i])
    train_features_ps_M3[i] = Model3.predict(x_train_ps[i])
    train_features_ps_M4[i] = Model4.predict(x_train_ps[i])
    train_features_ps_M5[i] = Model5.predict(x_train_ps[i])
    
###################

for i in range(len(train_pn)):

    img_train_pn [i] = image.load_img("./Data/micrograph/" + path_train_pn[i], target_size=(224, 224))
    x_train_pn[i] = image.img_to_array(img_train_pn[i])
    x_train_pn[i] = x_train_pn[i][0:484,:,:] # crop the bottom subtitles
    x_train_pn[i] = np.expand_dims(x_train_pn[i], axis=0)
    x_train_pn[i] = preprocess_input(x_train_pn[i])
    
for i in range(len(train_pn)):
    
    train_features_pn_M1[i] = Model1.predict(x_train_pn[i])
    train_features_pn_M2[i] = Model2.predict(x_train_pn[i])
    train_features_pn_M3[i] = Model3.predict(x_train_pn[i])
    train_features_pn_M4[i] = Model4.predict(x_train_pn[i])
    train_features_pn_M5[i] = Model5.predict(x_train_pn[i])
    
###################

for i in range(len(train_sn)):

    img_train_sn[i] = image.load_img("./Data/micrograph/" + path_train_sn[i], target_size=(224, 224))
    x_train_sn[i] = image.img_to_array(img_train_sn[i])
    x_train_sn[i] = x_train_sn[i][0:484,:,:] # crop the bottom subtitles
    x_train_sn[i] = np.expand_dims(x_train_sn[i], axis=0)
    x_train_sn[i] = preprocess_input(x_train_sn[i])
    
for i in range(len(train_sn)):
    train_features_sn_M1[i] = Model1.predict(x_train_sn[i])
    train_features_sn_M2[i] = Model2.predict(x_train_sn[i])
    train_features_sn_M3[i] = Model3.predict(x_train_sn[i])
    train_features_sn_M4[i] = Model4.predict(x_train_sn[i])
    train_features_sn_M5[i] = Model5.predict(x_train_sn[i])
    
    
###################

for i in range(len(train_set)):

    img_train[i] = image.load_img("./Data/micrograph/" + path_train[i], target_size=(224, 224))
    x_train[i] = image.img_to_array(img_train[i])
    x_train[i] = x_train[i][0:484,:,:] # crop the bottom subtitles
    x_train[i] = np.expand_dims(x_train[i], axis=0)
    x_train[i] = preprocess_input(x_train[i])
    
for i in range(len(train_set)):
    train_features_M1[i] = Model1.predict(x_train[i])
    train_features_M2[i] = Model2.predict(x_train[i])
    train_features_M3[i] = Model3.predict(x_train[i])
    train_features_M4[i] = Model4.predict(x_train[i])
    train_features_M5[i] = Model5.predict(x_train[i])
    
###################


# In[12]:


def reshaper(X):
    x=[]
    for i in range(len(X)):
        v=[]
        a=X[i].reshape(X.shape[3],-1)
        for c in range(len(a)):
            v.append(sum(a[c])/(X.shape[1]*X.shape[2]))
        x.append(np.array(v).T)
    return np.array(x),np.array(v)


# In[13]:


X_features_ps_M1 = [0 for i in range(len(train_ps))]
X_ps_M1 = [0 for i in range(len(train_ps))]
X_features_ps_M2 = [0 for i in range(len(train_ps))]
X_ps_M2 = [0 for i in range(len(train_ps))]
X_features_ps_M3 = [0 for i in range(len(train_ps))]
X_ps_M3 = [0 for i in range(len(train_ps))]
X_features_ps_M4 = [0 for i in range(len(train_ps))]
X_ps_M4 = [0 for i in range(len(train_ps))]
X_features_ps_M5 = [0 for i in range(len(train_ps))]
X_ps_M5 = [0 for i in range(len(train_ps))]

for i in range(len(train_ps)):
    X_ps_M1[i], X_features_ps_M1[i] = reshaper(train_features_ps_M1[i])
    X_ps_M2[i], X_features_ps_M2[i] = reshaper(train_features_ps_M2[i])
    X_ps_M3[i], X_features_ps_M3[i] = reshaper(train_features_ps_M3[i])
    X_ps_M4[i], X_features_ps_M4[i] = reshaper(train_features_ps_M4[i])
    X_ps_M5[i], X_features_ps_M5[i] = reshaper(train_features_ps_M5[i])

##########

X_features_pn_M1 = [0 for i in range(len(train_pn))]
X_pn_M1 = [0 for i in range(len(train_pn))]
X_features_pn_M2 = [0 for i in range(len(train_pn))]
X_pn_M2 = [0 for i in range(len(train_pn))]
X_features_pn_M3 = [0 for i in range(len(train_pn))]
X_pn_M3 = [0 for i in range(len(train_pn))]
X_features_pn_M4 = [0 for i in range(len(train_pn))]
X_pn_M4 = [0 for i in range(len(train_pn))]
X_features_pn_M5 = [0 for i in range(len(train_pn))]
X_pn_M5 = [0 for i in range(len(train_pn))]

for i in range(len(train_pn)):
    X_pn_M1[i], X_features_pn_M1[i] = reshaper(train_features_pn_M1[i])
    X_pn_M2[i], X_features_pn_M2[i] = reshaper(train_features_pn_M2[i])
    X_pn_M3[i], X_features_pn_M3[i] = reshaper(train_features_pn_M3[i])
    X_pn_M4[i], X_features_pn_M4[i] = reshaper(train_features_pn_M4[i])
    X_pn_M5[i], X_features_pn_M5[i] = reshaper(train_features_pn_M5[i])

##########

X_features_sn_M1 = [0 for i in range(len(train_sn))]
X_sn_M1 = [0 for i in range(len(train_sn))]
X_features_sn_M2 = [0 for i in range(len(train_sn))]
X_sn_M2 = [0 for i in range(len(train_sn))]
X_features_sn_M3 = [0 for i in range(len(train_sn))]
X_sn_M3 = [0 for i in range(len(train_sn))]
X_features_sn_M4 = [0 for i in range(len(train_sn))]
X_sn_M4 = [0 for i in range(len(train_sn))]
X_features_sn_M5 = [0 for i in range(len(train_sn))]
X_sn_M5 = [0 for i in range(len(train_sn))]

for i in range(len(train_sn)):
    X_sn_M1[i], X_features_sn_M1[i] = reshaper(train_features_sn_M1[i])
    X_sn_M2[i], X_features_sn_M2[i] = reshaper(train_features_sn_M2[i])
    X_sn_M3[i], X_features_sn_M3[i] = reshaper(train_features_sn_M3[i])
    X_sn_M4[i], X_features_sn_M4[i] = reshaper(train_features_sn_M4[i])
    X_sn_M5[i], X_features_sn_M5[i] = reshaper(train_features_sn_M5[i])

##########

X_features_M1 = [0 for i in range(len(train_set))]
X_M1 = [0 for i in range(len(train_set))]
X_features_M2 = [0 for i in range(len(train_set))]
X_M2 = [0 for i in range(len(train_set))]
X_features_M3 = [0 for i in range(len(train_set))]
X_M3 = [0 for i in range(len(train_set))]
X_features_M4 = [0 for i in range(len(train_set))]
X_M4 = [0 for i in range(len(train_set))]
X_features_M5 = [0 for i in range(len(train_set))]
X_M5 = [0 for i in range(len(train_set))]

for i in range(len(train_set)):
    X_M1[i], X_features_M1[i] = reshaper(train_features_M1[i])
    X_M2[i], X_features_M2[i] = reshaper(train_features_M2[i])
    X_M3[i], X_features_M3[i] = reshaper(train_features_M3[i])
    X_M4[i], X_features_M4[i] = reshaper(train_features_M4[i])
    X_M5[i], X_features_M5[i] = reshaper(train_features_M5[i])
    
##########    


# In[22]:


from sklearn import datasets

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier

from sklearn.model_selection import cross_val_score


clf = SVC(C=10, kernel = 'rbf', random_state = 0)
clf_multi = OneVsOneClassifier(clf)

score_M1_PS = cross_val_score(clf, X_features_ps_M1, y_train_ps, cv=10)
score_M2_PS = cross_val_score(clf, X_features_ps_M2, y_train_ps, cv=10)
score_M3_PS = cross_val_score(clf, X_features_ps_M3, y_train_ps, cv=10)
score_M4_PS = cross_val_score(clf, X_features_ps_M4, y_train_ps, cv=10)
score_M5_PS = cross_val_score(clf, X_features_ps_M5, y_train_ps, cv=10)

score_M1_PN = cross_val_score(clf, X_features_pn_M1, y_train_pn, cv=10)
score_M2_PN = cross_val_score(clf, X_features_pn_M2, y_train_pn, cv=10)
score_M3_PN = cross_val_score(clf, X_features_pn_M3, y_train_pn, cv=10)
score_M4_PN = cross_val_score(clf, X_features_pn_M4, y_train_pn, cv=10)
score_M5_PN = cross_val_score(clf, X_features_pn_M5, y_train_pn, cv=10)

score_M1_SN = cross_val_score(clf, X_features_sn_M1, y_train_sn, cv=10)
score_M2_SN = cross_val_score(clf, X_features_sn_M2, y_train_sn, cv=10)
score_M3_SN = cross_val_score(clf, X_features_sn_M3, y_train_sn, cv=10)
score_M4_SN = cross_val_score(clf, X_features_sn_M4, y_train_sn, cv=10)
score_M5_SN = cross_val_score(clf, X_features_sn_M5, y_train_sn, cv=10)

score_M1 = cross_val_score(clf_multi, X_features_M1, y_train, cv=10)
score_M2 = cross_val_score(clf_multi, X_features_M2, y_train, cv=10)
score_M3 = cross_val_score(clf_multi, X_features_M3, y_train, cv=10)
score_M4 = cross_val_score(clf_multi, X_features_M4, y_train, cv=10)
score_M5 = cross_val_score(clf_multi, X_features_M5, y_train, cv=10)



# In[92]:


PS_score = [round(1-np.average(score_M1_PS),3),round(1-np.average(score_M2_PS),3),round(1-np.average(score_M3_PS),3),round(1-np.average(score_M4_PS),3),round(1-np.average(score_M5_PS),3)]
PN_score = [round(1-np.average(score_M1_PN),3),round(1-np.average(score_M2_PN),3),round(1-np.average(score_M3_PN),3),round(1-np.average(score_M4_PN),3),round(1-np.average(score_M5_PN),3)]
SN_score = [round(1-np.average(score_M1_SN),3),round(1-np.average(score_M2_SN),3),round(1-np.average(score_M3_SN),3),round(1-np.average(score_M4_SN),3),round(1-np.average(score_M5_SN),3)]
Total_score = [round(1-np.average(score_M1),3),round(1-np.average(score_M2),3),round(1-np.average(score_M3),3),round(1-np.average(score_M4),3),round(1-np.average(score_M5),3)]

print("Errors for Max_pool Layers - 1st to 5th:\n")
print("Pearlite-Spheriodite CV Error, SVM Classifier:", PS_score)
print("Pearlite-Network CV Error, SVM Classifier:", PN_score)
print("Spheriodite-Network CV Error, SVM Classifier:", SN_score)
print("Three-Class CV Error, OVO Classifier:", Total_score)



# In[71]:


test_p = pd.DataFrame.as_matrix(test_pearlite)
test_s = pd.DataFrame.as_matrix(test_spheroidite)
test_n = pd.DataFrame.as_matrix(test_network)

path_test_p = test_p[:,1]
y_test_p = test_p[:,-1]

path_test_s = test_s[:,1]
y_test_s = test_s[:,-1]

path_test_n = test_n[:,1]
y_test_n = test_n[:,-1]

test_features_p_M5 = [0 for i in range(len(test_p))]
test_features_s_M5 = [0 for i in range(len(test_s))]
test_features_n_M5 = [0 for i in range(len(test_n))]

img_test_p = [0 for i in range(len(test_p))]
x_test_p = [0 for i in range(len(test_p))]

img_test_s = [0 for i in range(len(test_s))]
x_test_s = [0 for i in range(len(test_s))]

img_test_n = [0 for i in range(len(test_n))]
x_test_n = [0 for i in range(len(test_n))]

for i in range(len(test_p)):   

    img_test_p [i] = image.load_img("./Data/micrograph/" + path_test_p[i], target_size=(224, 224))
    x_test_p[i] = image.img_to_array(img_test_p[i])
    x_test_p[i] = x_test_p[i][0:484,:,:] # crop the bottom subtitles
    x_test_p[i] = np.expand_dims(x_test_p[i], axis=0)
    x_test_p[i] = preprocess_input(x_test_p[i])
    test_features_p_M5[i] = Model5.predict(x_test_p[i])
    
for i in range(len(test_s)):    
    img_test_s [i] = image.load_img("./Data/micrograph/" + path_test_s[i], target_size=(224, 224))
    x_test_s[i] = image.img_to_array(img_test_s[i])
    x_test_s[i] = x_test_s[i][0:484,:,:] # crop the bottom subtitles
    x_test_s[i] = np.expand_dims(x_test_s[i], axis=0)
    x_test_s[i] = preprocess_input(x_test_s[i])
    test_features_s_M5[i] = Model5.predict(x_test_s[i])
    
for i in range(len(test_n)):    
    img_test_n [i] = image.load_img("./Data/micrograph/" + path_test_n[i], target_size=(224, 224))
    x_test_n[i] = image.img_to_array(img_test_n[i])
    x_test_n[i] = x_test_n[i][0:484,:,:] # crop the bottom subtitles
    x_test_n[i] = np.expand_dims(x_test_n[i], axis=0)
    x_test_n[i] = preprocess_input(x_test_n[i])
    test_features_n_M5[i] = Model5.predict(x_test_n[i])


# In[72]:


X_features_test_p_M5 = [0 for i in range(len(test_p))]
X_p_test_M5 = [0 for i in range(len(test_p))]     
X_features_test_s_M5 = [0 for i in range(len(test_s))]
X_s_test_M5 = [0 for i in range(len(test_s))]
X_features_test_n_M5 = [0 for i in range(len(test_n))]
X_n_test_M5 = [0 for i in range(len(test_n))] 

for i in range(len(test_p)):   
    X_p_test_M5[i], X_features_test_p_M5[i] = reshaper(test_features_p_M5[i])
    
for i in range(len(test_s)):
    X_s_test_M5[i], X_features_test_s_M5[i] = reshaper(test_features_s_M5[i])
    
for i in range(len(test_n)):
    X_n_test_M5[i], X_features_test_n_M5[i] = reshaper(test_features_n_M5[i])

pred_test_PS_P = clf.fit(X_features_ps_M5, y_train_ps).predict(X_features_test_p_M5)
pred_test_PS_S = clf.fit(X_features_ps_M5, y_train_ps).predict(X_features_test_s_M5)

pred_test_PN_P = clf.fit(X_features_pn_M5, y_train_pn).predict(X_features_test_p_M5)
pred_test_PN_N = clf.fit(X_features_pn_M5, y_train_pn).predict(X_features_test_n_M5)

pred_test_SN_S = clf.fit(X_features_sn_M5, y_train_sn).predict(X_features_test_s_M5)
pred_test_SN_N = clf.fit(X_features_sn_M5, y_train_sn).predict(X_features_test_n_M5)


# In[94]:


print("Error PS on P_test:",round(1-list(pred_test_PS_P).count("pearlite")/len(pred_test_PS_P),3))
print("Error PS on S_test:",round(1-list(pred_test_PS_S).count("spheroidite")/len(pred_test_PS_S),3))

print("\nError PN on P_test:",round(1-list(pred_test_PN_P).count("pearlite")/len(pred_test_PN_P),3))
print("Error PN on N_test:",round(1-list(pred_test_PN_N).count("network")/len(pred_test_PN_N),3))

print("\nError SN on S_test:",round(1-list(pred_test_SN_S).count("spheroidite")/len(pred_test_SN_S),3))
print("Error SN on N_test:",round(1-list(pred_test_SN_N).count("network")/len(pred_test_SN_N),3))




# In[84]:


test_features_M5 = [0 for i in range(len(test_set))]


for i in range(len(test_set)):    
    img_test[i] = image.load_img("./Data/micrograph/" + path_test[i], target_size=(224, 224))
    x_test[i] = image.img_to_array(img_test[i])
    x_test[i] = x_test[i][0:484,:,:] # crop the bottom subtitles
    x_test[i] = np.expand_dims(x_test[i], axis=0)
    x_test[i] = preprocess_input(x_test[i])
    test_features_M5[i] = Model5.predict(x_test[i])


# In[95]:


X_features_test_M5 = [0 for i in range(len(test_set))]
X_test_M5 = [0 for i in range(len(test_set))] 

for i in range(len(test_set)):   
    X_test_M5[i], X_features_test_M5[i] = reshaper(test_features_M5[i])
    
pred_test_P = clf_multi.fit(X_features_M5, y_train).predict(X_features_test_p_M5)
pred_test_S = clf_multi.fit(X_features_M5, y_train).predict(X_features_test_s_M5)
pred_test_N = clf_multi.fit(X_features_M5, y_train).predict(X_features_test_n_M5)


# In[97]:


print("Error 3-class on P_test:",round(1-list(pred_test_P).count("pearlite")/len(pred_test_P),3))
print("Error 3-class on S_test:",round(1-list(pred_test_S).count("spheroidite")/len(pred_test_S),3))
print("Error 3-class on N_test:",round(1-list(pred_test_N).count("network")/len(pred_test_N),3))


# In[ ]:




