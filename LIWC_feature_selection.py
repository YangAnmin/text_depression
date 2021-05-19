import numpy as np
from numpy import mean
import pandas as pd
import random
from math import isnan

import sklearn
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.metrics import classification_report

# define data path
feature_path = '/Users/anmin/Documents/AI_text_5000/textmind_features/feature_all.csv'
data_path = '/Users/anmin/Documents/AI_text_5000/data_clean_all.xlsx'

# read data
data = np.array(pd.read_excel(data_path,engine='openpyxl'))
feature_df = pd.read_csv(feature_path)
feature = np.array(feature_df)

# eliminate id with null text
cut_list = []
for i in range(data.shape[0]):
    if data[i,0] not in feature[:,0]: # this id did not write text
        cut_list.append(i)

data = np.delete(data,cut_list,axis=0)

# generate depressed and normal feature matrix
mask_depressed = (data[:,-1] == 1)
mask_normal = (data[:,-1] == 0)

depressed_data = data[mask_depressed]
normal_data = data[mask_normal]


depress_feature = np.zeros((102,))
for id in depressed_data[:,0]:
    index = np.argwhere(feature[:,0]==id)[0][0]
    depress_feature = np.vstack((depress_feature,feature[index,1:]))
depress_feature = depress_feature[1:,:]

normal_feature = np.zeros((102,))
for id in normal_data[:,0]:
    index = np.argwhere(feature[:,0]==id)[0][0]
    normal_feature = np.vstack((normal_feature,feature[index,1:]))
normal_feature = normal_feature[1:,:]

# match sampel size between depressed and normal and form one matrix
def match_size(data_prep,data_to_match):
    """
    match sample size between depressed and normal

    Parameter
    ----------
    data_prep: ndarray
        feature array whose size required to be matched
    data_to_match: ndarray
        feature array to match data_prep

    Return
    ----------
    matched_data: ndarray
        matched size randomly chosen form normal feature pool
        and concatenate as one matrix

    """
    size = data_prep.shape[0]
    np.random.shuffle(data_to_match)
    to_match = data_to_match[:size,:]
    matched_data = np.vstack((data_prep,to_match))

    return matched_data


### Univariate feature selection: Pearson Correlation ###########################
example_num = depress_feature.shape[0]
depress_array = np.ones((example_num,))
normal_array = np.zeros((example_num,))
mood_array = np.hstack((depress_array,normal_array)) # 0 denotes normal state, 1 denotes depressed state at equal size

# random pool of features for 10 times
feature_pool = {}
for i in range(10):
    feature_pool[i] = match_size(depress_feature,normal_feature)

# pearson correlation
correlation = {}
for i in range(10):
    temp_list = []
    for k in range(102): # 102 features in total
        x = feature_pool[i][:,k]
        y = mood_array
        temp_list.append(pearsonr(x,y))
    correlation[i] = temp_list

# combine 10 times of correlation
corr_poll = {}
p_poll = {}
for i in range(10):
    corr_list = []
    p_list = []
    for item in correlation[i]:
        corr_list.append(item[0])
        p_list.append(item[1])
    corr_poll[i] = np.array(corr_list)
    p_poll[i] = np.array(p_list)

feature_corr = np.zeros(corr_poll[0].shape)
for i in range(10):
    feature_corr += corr_poll[i]
feature_corr = feature_corr/10

p_corr = np.zeros(p_poll[0].shape)
for i in range(10):
    p_corr += p_poll[i]
p_corr = p_corr/10


# choose feature by p-value
p_value = 0.0001
index = np.argwhere(p_corr<p_value)

corr_value = []
for i in range(index.shape[0]):
    corr_value.append(feature_corr[index[i][0]])

## acquire header
header_list = list(feature_df)
#selected_header = []
#for i in range(index.shape[0]):
    #selected_header.append(header_list[index[i][0]+1]) # omit the 1st column id,
                                                # if the index of correlation is 2, then in header its index is 3
selected_header = ['Funct','Pronoun','PPron','Verb','AuxVerb',
                    'Adverb','Conj','Humans','Affect','NegEmo',
                    'Inclusive','Exclusive','Bio','Body','Death',
                    'Assent','We','QuanUnit','ProgM',
                    'Anx','Sad','CogMech','Insight','Discrep',
                    'Tentat','Certain','Motion','Religion','Colon']

# df of selected features
data_array = np.c_[selected_header,corr_value]
head = ['feature','correlation_value']
df = pd.DataFrame(data_array,columns=head)

# save selected features
df.to_csv('/Users/anmin/Documents/AI_text_5000/LIWC_LSTM/selected_feature_zhu.csv')

### prediction ##################################################################
# sort out selected features from data
#path_selc_features = '/Users/anmin/Documents/AI_text_5000/LIWC_LSTM/selected_feature0.001.csv'
#selected_features = pd.read_csv(path_selc_features)
#selected_features = np.array(selected_features['feature'])
selected_features = selected_header

# find the index of the features
header_list = list(feature_df)
feature_index = []
for feature in selected_features:
    feature_index.append(header_list.index(feature)-1)

LR, LR_weights  = {}, {}

for i in range(10):
    matched_data = match_size(depress_feature,normal_feature)[:,feature_index]
    X = matched_data
    y = mood_array

    ## Logistic Regression ######################################################
    clf = LogisticRegressionCV(cv=10, random_state=0).fit(X, y)
    predict_lable = clf.predict(X)
    LR[i] = classification_report(y,predict_lable)

    # weights of logistic regression
    LR_weights[i] = clf.coef_[0] # regression coefficients of LR

# calculate mean weight
weight = np.zeros(LR_weights[0].shape)
for i in range(10):
    weight += LR_weights[i]
weight_avg = weight/10

# save csv file
feature_weight = np.c_[selected_features,weight_avg]
header_fw = ['feature','weight']
df_fw = pd.DataFrame(feature_weight,columns=header_fw)
df_fw.to_csv('/Users/anmin/Documents/AI_text_5000/LIWC_LSTM/feature_weight_all.csv')
