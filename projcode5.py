# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:29:01 2022

@author: MaxRo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:44:47 2022

@author: MaxRo
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tslearn


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


from tslearn.clustering import TimeSeriesKMeans

from scipy.stats import multivariate_normal

train_file = "C:/Users\MaxRo\OneDrive\Desktop\senioryear\ece480\project\Train_Arabic_Digit.txt"
test_file = "C:/Users\MaxRo\OneDrive\Desktop\senioryear\ece480\project\Test_Arabic_Digit.txt"

n_dims = 8

class Mixtures:
    cov = []
    mean = []

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


#train_data = pd.read_csv(train_file, sep=" ",header=None)
recalc= True
if(recalc):
    data = []
    labels = []
    newdata= []
    for i in range(n_dims):
        newdata.append([])
    with open(train_file) as f:
        lines = f.readlines()
        j=1
        label_label = 0
        for line in lines:
                if(line == '            \n'):
                    if(newdata[0] != []):
                        data.append(newdata)
                        newdata= []
                        for i in range(n_dims):
                            newdata.append([])
                        labels.append(label_label)
                        print(label_label)
                        if(j % 660 == 0):
                            label_label += 1
                    
                        j+=1
                        #print(j) 
                else:
                    #newdata= [[]] * 13
                    vals = line.split(" ")
                    for i in range(n_dims):
                        v = float(vals[i])
                        newdata[i].append(v)
 
                    

test_data = []
test_labels = []
newdata= []
for i in range(n_dims):
    newdata.append([])
with open(test_file) as f:
    lines = f.readlines()
    j=1
    label_label = 0
    for line in lines:
            vals = line.split(" ")
            #if(line == '            \n' or (line == '             ')):
            if(vals[0] == ''):
                if(newdata[0] != []):
                    test_data.append(newdata)
                    newdata= []
                    for i in range(n_dims):
                        newdata.append([])
                    test_labels.append(label_label)
                    print(label_label)
                    if(j % 220 == 0):
                        label_label += 1
                
                    j+=1
                    #print(j) 
            else:
                #newdata= [[]] * 13
                
                for i in range(n_dims):
                        v = float(vals[i])
                        newdata[i].append(v)
                        

zeros = data[0:660]
ones = data[660:1320]
twos = data[1320:1980]
threes = data[1980:2640]
fours =  data[2640:3300]
fives = data[3300:3960]
sixes = data[3960:4620]
sevens = data[4620:5280]
eights = data[5280:5940]
nines = data[5940:6600]

#full_digits_data = [zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines]
full_digits_data = [zeros]
# for i in range(660):
#     zeros[i] = zeros[i][:2]
  
# zeros_unpacked = []
# n=0
# for i in range(660):
#             for k in range(len(zeros[i][0])):
#                 #print(k)
#                 new_point = []
#                 for j in range(13):
#                     new_point.append(zeros[i][j][k])
#                 zeros_unpacked.append(new_point)
#                 n+=1
#print(n)
# ss = []
# for i in range(1,20):            
#     zeros_kmeans = KMeans(n_clusters=i).fit(zeros_unpacked)
#     ss.append(zeros_kmeans.inertia_)
# plt.figure(1)
# plt.plot(np.arange(1,20),ss)   

# n=0
# fig,axes = plt.subplots(2,5)
# fig.suptitle("KMeans Inertia vs. Number of Clusters")
# plt.rc('ytick',labelsize=5)
# for digit_list in full_digits_data:
#     unpacked = []
#     for i in range(660):
#           for k in range(len(digit_list[i][0])):
#               #print(k)
#               new_point = []
#               for j in range(13):
#                   new_point.append(digit_list[i][j][k])
#               unpacked.append(new_point)
#     ss = []
#     for m in range(1,10):            
#         kmeans = KMeans(n_clusters=m).fit(unpacked)
#         ss.append(kmeans.inertia_)
#     #plt.figure(n)

#     if(n<5):
#         x = 0
#     else:
#         x = 1
#     axes[x][n % 5].set_title("Digit: "+str(n))
#     axes[x][n % 5].set_xlabel("Number of Clusters")
#     axes[x][n % 5].set_ylabel("Inertia")
#     axes[x][n % 5].ticklabel_format(axis='y',style='sci')
#     axes[x][n % 5].plot(np.arange(1,10),ss)   
#     n+=1        
   
# fig.tight_layout()

mixture_component_params = []
gmm_list = []
for digit_list in full_digits_data:
    unpacked = []
    for i in range(660):
          for k in range(len(digit_list[i][0])):
              #print(k)
              new_point = []
              for j in range(n_dims):
                  new_point.append(digit_list[i][j][k])
              unpacked.append(new_point)          
    kmeans = KMeans(n_clusters=3).fit(unpacked)
    gmm = GaussianMixture(n_components=3,covariance_type='spherical').fit(unpacked)
    gmm_list.append(gmm)
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    for n in range(len(unpacked)):
        if(kmeans.labels_[n]==0):
            g1.append(unpacked[n])
        if(kmeans.labels_[n]==1):
            g2.append(unpacked[n])
        if(kmeans.labels_[n]==2):
            g3.append(unpacked[n])
        if(kmeans.labels_[n]==3):
            g4.append(unpacked[n])    
    centers = kmeans.cluster_centers_
    cov1 = np.cov(np.asarray(g1),rowvar=False)
    cov2 = np.cov(np.asarray(g2),rowvar=False)
    cov3 = np.cov(np.asarray(g3),rowvar=False)
    #cov4 = np.cov(np.asarray(g4),rowvar=False)
    
    pi1 = len(g1)/len(unpacked)
    pi2 = len(g2)/len(unpacked)
    pi3 = len(g3)/len(unpacked)
   # pi4 = len(g4)/len(unpacked)

    
    covs = [cov1,cov2,cov3]
    pis = [pi1,pi2,pi3]
    mixture_component_params.append([centers,covs,pis])

def calc_mixture_prob(x,centers,covs,pis):
    pi1 = pis[0]
    pi2 = pis[1]
    pi3 = pis[2]
    #pi4 = pis[3]
    
    p1 = pi1* multivariate_normal.pdf(x,mean=centers[0],cov=covs[0])
    p2 = pi2* multivariate_normal.pdf(x,mean=centers[1],cov=covs[1])
    p3 = pi3* multivariate_normal.pdf(x,mean=centers[2],cov=covs[2])
    #p4 = pi4* multivariate_normal.pdf(x,mean=centers[3],cov=covs[3])
    prob = p1+p2+p3
    return prob

test_data_cut = test_data[440:660]
test_zeros = test_data[0:220]
test_ones = test_data[220:440]
test_twos = test_data[440:660]
test_threes = test_data[660:880]
test_fours = test_data[880:1100]
test_fives = test_data[1100:1320]
test_sixes = test_data[1320:1540]
test_sevens = test_data[1540:1760]
test_eights = test_data[1760:1980]
test_nines = test_data[1980:2200]


full_digits_test_data = [test_zeros,test_ones,test_twos,test_threes,test_fours,test_fives,test_sixes,test_sevens,test_eights,test_nines]


to_plot = 0

datapoint = data[to_plot]
unpacked = []
for i in range(len(datapoint[0])):
    newpoint = []
    for j in range(n_dims):
        newpoint.append(datapoint[j][i])
    unpacked.append(newpoint)
preds = kmeans.predict(unpacked)    

pred = kmeans.predict(unpacked)

colors = ['r','b','y']
for i in range(n_dims):
    dat_plot = []
    for j in range(len(unpacked)):
        dat_plot.append(unpacked[j][i])
        plt.scatter(j,unpacked[j][i],label=pred[j],color=colors[pred[j]])  
    # mcc = data[to_plot][i]

    plt.plot(dat_plot,'k-',label='__nolegend__')
    k+=1
     
custom_dots = [Line2D([0], [0], marker='o', color='w', label='Cluster 0', markerfacecolor='r', markersize=12),
               Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='b', markersize=12),
               Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='y', markersize=12)
               ]


plt.legend(handles=custom_dots)
plt.xlabel("Sample Number",fontsize=13)
plt.ylabel("MFCC Amplitude",fontsize=13)
plt.title("MFCCs For Digit 0, Observation 0, k=3 Clusters",fontsize=15)



# gmm_correct = 0
# kmeans_correct =0
# n = 0
# gmm_confusion_mat = np.zeros((10,10))
# kmeans_confusion_mat = np.zeros((10,10))

# for digit_list in full_digits_test_data:
#     for datapoint in digit_list:
#         unpacked = []
#         for i in range(len(datapoint[0])):
#             newpoint = []
#             for j in range(n_dims):
#                 newpoint.append(datapoint[j][i])
#             unpacked.append(newpoint)
        
#         likelihoods = []
#         for digit in range(10):
#             centers = mixture_component_params[digit][0]
#             covs = mixture_component_params[digit][1]
#             pis = mixture_component_params[digit][2]
#             prod = 1
#             for point in unpacked:
#                 prod *= calc_mixture_prob(point,centers,covs,pis)
#             likelihoods.append(prod)
#         prediction = argmax(likelihoods)
#         kmeans_confusion_mat[n][prediction] +=1
#         print("Kmeans Predicted Digit: " + str(prediction))    
#         if(prediction == n):
#             kmeans_correct+=1
            
#         likelihoods = []
#         for digit in range(10):
#             prod = 1
#             for point in unpacked:
#                 sample = np.asarray(point)
#                 sample = sample.reshape((1,-1))
#                 prod *= np.exp(gmm_list[digit].score_samples(sample)[0])
#             likelihoods.append(prod)
#         prediction = argmax(likelihoods)
#         print("GMM Predicted Digit: " + str(prediction))
#         if(prediction == n):
#             gmm_correct+=1
#         gmm_confusion_mat[n][prediction] +=1
#     n += 1
        
# print("Total Accuracy")
# print("Kmeans: {}%".format(kmeans_correct/len(test_data)))
# print("GMM: {}%".format(gmm_correct/len(test_data)))


# df_cm = pd.DataFrame(kmeans_confusion_mat, index = [i for i in "0123456789"],
#                   columns = [i for i in "0123456789"])
# plt.figure(1,figsize = (10,7))
# # plt.xlabel("True Digit")
# # plt.ylabel("Predicted Digit")
# plt.title("KMeans Confusion Matrix")

# s = sn.heatmap(df_cm, annot=True)
# s.set(xlabel="Predicted Digit",ylabel="True Digit")
       

# df_cm = pd.DataFrame(gmm_confusion_mat, index = [i for i in "0123456789"],
#                   columns = [i for i in "0123456789"])
# plt.figure(2,figsize = (10,7))
# # plt.xlabel("True Digit")
# # plt.ylabel("Predicted Digit")
# plt.title("GMM Confusion Matrix")

# s = sn.heatmap(df_cm, annot=True)
# s.set(xlabel="Predicted Digit",ylabel="True Digit")
       
# plt.figure(1)
# k=1

#simple time series plot
# for i in range(len(data[1])):
#     mcc = data[1600][i]
#     if(i in [0,1,2,3,4,5,6]):
#         plt.plot(mcc,label=k)
#         k+=1
# plt.legend()
# plt.xlabel("Sample Number")
# plt.ylabel("MCC Amplitude")

#plot average mcc1,mcc2 for each digit over time across all samples 
# for digit in range(0,10):
#     plt.figure(digit)
#     mcc1_avg = []
#     mcc2_avg = []
#     mcc3_avg = []
#     mcc4_avg = []
#     count = 0
#     start = digit * 660
#     end = (digit+1) * 660 -1
#     for i in range(30):
#             mcc1_avg.append(0)
#             mcc2_avg.append(0)
#             mcc3_avg.append(0)
#             mcc4_avg.append(0)
#     for i in range(start,end):
#             mcc1 = data[i][0]
#             mcc2 = data[i][1]
#             mcc3 = data[i][2]
#             mcc4 = data[i][3]
#             if(len(mcc1)>=30 and len(mcc2) >=30 and len(mcc3) >=30 and len(mcc4) >=30):
#                 count +=1
#                 for j in range(0,30):
#                     mcc1_avg[j] += mcc1[j]
#                     mcc2_avg[j] += mcc2[j]
#                     mcc3_avg[j] += mcc3[j]
#                     mcc4_avg[j] += mcc4[j]
#             #plt.plot(mcc1,color='b')
#             #plt.plot(mcc2,color='g')
#     for j in range(0,30):
#         mcc1_avg[j] /= count
#         mcc2_avg[j] /= count
#         mcc3_avg[j] /= count
#         mcc4_avg[j] /= count
    
#     plt.plot(mcc1_avg,color='y',label="MCC1 Average")
#     plt.plot(mcc2_avg,color='k',label="MCC2 Average")
#     plt.plot(mcc3_avg,color='b',label="MCC3 Average")
#     plt.plot(mcc4_avg,color='g',label="MCC4 Average")
#     plt.legend()
#     plt.title("Digit: "+str(digit))
#     plt.xlabel("Sample Number")
#     plt.ylabel("MCC Amplitude")

#training features
#run through data, chop it to ~30 samples, create 3 time averages for mcc 1 and 2, that is feature vector
# features = []
# ts_features = []
# samples = 30
# spacing = 10
# n_blocks = samples//spacing
# for i in range(len(data)):

#     mcc1 = data[i][0]
#     mcc2 = data[i][1]
#     mcc3 = data[i][2]
#     mcc4 = data[i][3]
    
#     if(len(mcc1)>=30 and len(mcc2) >=30):
#         newfeatures = []
#         # ts_newfeatures = []
#         # ts_newfeatures.append(mcc1[0:29])
#         # ts_newfeatures.append(mcc2[0:29])
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc1[start:end])/len(mcc1[start:end]))
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc2[start:end])/len(mcc2[start:end]))
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc3[start:end])/len(mcc3[start:end]))
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc4[start:end])/len(mcc4[start:end]))
#         features.append(newfeatures)
#         #ts_features.append(np.asarray(ts_newfeatures))
   
  
# #testing features
# test_features = []
# test_labels_cut = []

# for i in range(len(test_data)):

#     mcc1 = test_data[i][0]
#     mcc2 = test_data[i][1]
#     mcc3 = test_data[i][2]
#     mcc4 = test_data[i][3]
    
#     if(len(mcc1)>=30 and len(mcc2) >=30):
#         newfeatures = []
#         # ts_newfeatures = []
#         # ts_newfeatures.append(mcc1[0:29])
#         # ts_newfeatures.append(mcc2[0:29])
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc1[start:end])/len(mcc1[start:end]))
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc2[start:end])/len(mcc2[start:end]))
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc3[start:end])/len(mcc3[start:end]))
#         for j in range(n_blocks):
#             start = j*spacing
#             end = (j+1) * spacing - 1
#             newfeatures.append(sum(mcc4[start:end])/len(mcc4[start:end]))
#         test_features.append(newfeatures)
#         test_labels_cut.append(test_labels[i])


     
# kmeans = KMeans(n_clusters=10).fit(features)    
# out_labels = kmeans.labels_    


# gmm = GaussianMixture(n_components=10).fit(features)
# gmm_pred = gmm.predict(test_features)

# correct = 0
# for i in range(len(test_labels_cut)):
#     if(test_labels_cut[i] == gmm_pred[i]):
#         correct+=1
# print("Accuracy: {}%".format(correct/len(test_labels_cut)*100))
        
# ts_kmeans = TimeSeriesKMeans(n_clusters=10,metric='dtw',max_iter=10).fit(ts_features)
# ts_out_labels = ts_kmeans.labels_
