import numpy as np
import pandas as pd
import os

# define data path
path = '/Users/anmin/documents/AI_text_5000/lelezuowen_feature'
text_path = '/Users/anmin/documents/AI_text_5000/lele_zuowen'
select_path = '/Users/anmin/Documents/AI_text_5000/LIWC_LSTM/feature_weight_all.csv'

# concatenate feature matrix
selected_features = pd.read_csv(select_path)
features = selected_features['feature']
weights = selected_features['weight']

data = np.zeros((len(features),))
for i in range(1,27):
    data_temp = pd.read_csv(os.path.join(path,str(i)+'.csv'))
    data_temp = pd.DataFrame(data_temp,columns=features)
    data = np.vstack((data,np.array(data_temp)))
data = data[1:,:]

data_combined = np.dot(data,weights)
index = data_combined.argsort() # index from the smallest to the largest, normat to depressed

id = index + 1
normal_id = id[-40000:]
depressed_id = id[:40000]

# write csv file
col_state = np.hstack((np.zeros(40000,),np.ones(40000,))) # 0 denotes normal, 1 denotes depressed

## write normal text
text_normal = []
for i in range(40000):
    name = str(normal_id[i])+'.txt'
    with open(os.path.join(text_path,name),'r') as f:
        text_temp = f.read()
        text_normal.append(text_temp)

## write depressed text
text_depressed = []
for i in range(40000):
    name = str(normal_id[i])+'.txt'
    with open(os.path.join(text_path,name),'r') as f:
        text_temp = f.read()
        text_depressed.append(text_temp)

# concatenate text array
col_text = np.hstack((np.array(text_normal),np.array(text_depressed)))

# df
data = np.c_[col_text,col_state]
df = pd.DataFrame(data)
df.columns = ['text','catagory']
df.to_csv('/Users/anmin/documents/AI_text_5000/LIWC_LSTM/lele_LIWC_all.csv',encoding='utf_8_sig')
