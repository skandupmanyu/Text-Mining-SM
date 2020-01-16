    # -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:09:13 2018

@author: 143637
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:32:07 2018

@author: 143637
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:35:30 2018

@author: 143637
"""

###############################################################

#1. Import libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

##################################################################

#2. Import data
data = pd.read_excel("C:/Users/upman/Downloads/Case Details Orignial.xlsx")
data = data.fillna("NA")
case_list = list(data["Case History"])
################################################################



#3. Data preprocessing

###3.1 Removing special characters and numbers
pattern = '[^A-Za-z]+'
for i in range(len(case_list)):   
    if (i%100 ==0):
        print (i)
        
    tmp_string = case_list[i]
    tmp_string = re.sub(pattern, '', tmp_string)
    case_list[i] = tmp_string
#################################################################


    
###3.2 Convert to lowercase
for i in range(len(case_list)):   
    if (i%100 ==0):
        print (i)
        
    tmp_string = case_list[i]
    case_list[i] = tmp_string.lower()
#################################################################




###3.3 Removing stop words
stop_words = set(stopwords.words('english')) 

for i in range(len(case_list)):
    temp_list = []
    for word in case_list[i].split(" "):
        if word not in stop_words:
            temp_list.append(word)
    case_list[i] = " ".join(temp_list)
################################################################        



###3.4 Stemming
stemmer =SnowballStemmer("english")

for i in range(len(case_list)):
    temp_list = []
    for word in case_list[i].split(" "):
        temp_list.append(stemmer.stem(word))
        
    case_list[i] = " ".join(temp_list)
################################################################

documents = case_list



#4 Calculate TFIDF Values
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(documents)

#TFIDF Values to CSV
tfidf_dense = X.todense() #converts a sparce matrix to dense matrix
tfidf = pd.DataFrame(tfidf_dense) #converts to dataframe
features = tfidf_vectorizer.get_feature_names() #get words
tfidf.columns = features #set words as columnnames
tfidf.to_csv('tfidf.csv', sep=',') #export to csv
#################################################################



#5. Reduce variables

###5.1 No. of variables (n_components) selection
tsvd = TruncatedSVD(n_components=X.shape[1] - 1) #reduce just one variable
X_tsvd = tsvd.fit(X) #Train
var_ratio_exp = X_tsvd.explained_variance_ratio_.cumsum() #Ratios of variance explained

#plot
plt.xlabel('Number of Variables', fontsize=14, color='black')
plt.ylabel('Variance Explained(%)', fontsize=14, color='black')
plt.title ('SVD Analysis' ,fontsize=20, color='blue')
plt.plot(range(len(var_ratio_exp)), var_ratio_exp * 100) #plot
plt.show()
###################################################################



###5.2 Reduce the variables to n_components()
svd = TruncatedSVD(n_components = 700,random_state = 63) #Define our SVD
normalizer = Normalizer(copy=False) #whole row to unit norm. Every document now is a unit vector
LSA = make_pipeline(svd, normalizer) #
Xnew = LSA.fit_transform(X)

#Export to CSV
tfidf_reduced = pd.DataFrame(Xnew)
tfidf_reduced.to_csv('tfidf_reduced.csv', sep=',')
###################################################################



#6. Clustering

###6.1 Selecting no. of clusters
Nc = range(1, 40)
#k-means++ is the method of initialization - a smart way for faster convergence
kmeans = [KMeans(n_clusters=i, init='k-means++', max_iter=500, random_state =63) for i in Nc]
inertia = [kmeans[i].fit(Xnew).inertia_ for i in range(len(kmeans))]
plt.plot(Nc,inertia)
plt.xlabel('Number of Clusters', fontsize=14, color='black')
plt.ylabel('Sum of squared distances', fontsize=14, color='black')
plt.title('Elbow Curve' ,fontsize=20, color='blue')
plt.show()

#6.2 Deploy k means
model = KMeans(n_clusters=20, init='k-means++', max_iter=500, random_state =63)
model.fit(Xnew)

#6.3 Output cluster numbers to a text files
f = open('label.txt','w')
label = [str(x) for x in model.labels_.tolist()]
f.write("\n".join(label))
f.close()