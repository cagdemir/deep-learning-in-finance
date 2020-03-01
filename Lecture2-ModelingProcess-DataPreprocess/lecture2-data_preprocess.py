
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import quantile_transform

import matplotlib.pyplot as plt
plt.style.use('seaborn')

#import statsmoedls.api as sm

# 0. write necessary functions - ranking function
# 1. get data paths
# 2. prepare global frame and train frame and macro data
# 3. read return, pass it to global frame, shift it by -1, prepare return mask, flatten it and attach to train frame
# 4. read var, pass it to global frame, apply return mask, rank it, multiply w/ macroeconomcis data and attach them to train frame
# 5. splitting train, validation, test data


## 0. functions

def ranking(series, low=-1, high=1):
    
    ranked_series = series.rank(method='dense').copy()
    max_rank = ranked_series.max()    
    ranks = (ranked_series-1)/(max_rank-1)
    
    return (high-low)*ranks + low

def ranking_vectorized(df, axis=1, low=-1, high=1):
    
    return pd.DataFrame(quantile_transform(df,axis=axis, copy=True) * (high-low) + low, index=df.index,columns=df.columns)

## 1. data paths
    
path_var = '.data/'
path_macro = './data/macro_vars.xlsx'


## 2. global frame & train frame, macro data

start_date = '12-1-1978' 
end_date = '11-30-2018'

idx_global = pd.date_range(start=start_date , end=end_date , freq='M')
n_period = len(idx_global)

total_year = int(n_period /12)

# first macro data
vars_macro = pd.read_excel(path_macro)
vars_macro.index =  pd.date_range(start=vars_macro.iloc[0,0] , end=vars_macro.iloc[-1,0] , freq='M')

vars_macro = vars_macro[(vars_macro.index>=start_date) & (vars_macro.index<=end_date)].iloc[:,1:]
#vars_macro.index = idx_global
#vars_macro = sm.add_constant(vars_macro)

n_macro = vars_macro.shape[1]

# 3. read return, organize it

path_ret = 'data/0.csv'
df_ret = pd.read_csv(path_ret)
df_ret.index =  pd.date_range(start=df_ret.iloc[0,0] , end=df_ret.iloc[-1,0] , freq='M')
df_ret = df_ret.shift(-1).iloc[:,1:]

df_ret = df_ret[(df_ret.index>=start_date) & (df_ret.index<=end_date)]
df_ret.index = idx_global

#applying mcap threshold to return dataframe, thereby all variables

# read mcap

path_mcap = 'data/mcap.csv'
mcap_raw = pd.read_csv(path_mcap)

mcap_raw.index = pd.date_range(start=mcap_raw.iloc[0,0] , end=mcap_raw.iloc[-1,0] , freq='M')
mcap_raw.drop(mcap_raw.columns[0], inplace=True, axis=1)
df_mcap = mcap_raw[(mcap_raw.index>=start_date) & (mcap_raw.index<=end_date)]

# reading sic, will be added after all variables are added

path_sic = '/home/research/Desktop/WRDS_Data_Preprocessing/data/GuAnomalies/Shifted/Anomalies500-600//501.csv'
df_sic = pd.read_csv(path_sic)

df_sic.index =  pd.date_range(start=df_sic.iloc[0,0] , end=df_sic.iloc[-1,0] , freq='M')
df_sic = df_sic.iloc[:,1:]

df_sic = df_sic[(df_sic.index>=start_date) & (df_sic.index<=end_date)].astype(float)
df_sic.index = idx_global

# now global frame stuff

list_var = os.listdir(path_var)
list_nvar = sorted([int(i.split('.')[0]) for i in list_var])
n_var = len(list_nvar)

list_temp = set(int(i) for i in df_ret.columns)

# yukarıda önce int ardından burada tekrar str yapmamızın sebebi şu:
# okuduğumuz variable df'lerinde kolon başlıkları str olduğundan var'ları 
# global frame'e aktarmak için str olmaları lazım. öte taraftan stock'laro 
# track etmeyi kolaylaştırmak için de sayı olarak sort etmemiz lazım.

# creating global frame
list_stocks =[str(i) for i in sorted(list(list_temp))]
n_stocks = len(list_stocks)

frame_global = pd.DataFrame(columns=list_stocks, index=idx_global)

# wrangling return, creating return mask and sic df's

frame_ret = frame_global.copy()
frame_ret.update(df_ret)
mask_return = frame_ret.notnull()

frame_mcap = frame_global.copy()
frame_mcap.update(df_mcap)
mask_mcap = frame_mcap.notnull()


mcap_threshold = .5
mask_mcap_threshold = frame_mcap.ge(frame_mcap.apply(lambda row: row.quantile(mcap_threshold), axis=1), axis=0)

mask_universal = (mask_mcap_threshold & mask_return)

frame_ret = frame_ret[mask_universal]
#frame_ret = frame_ret.apply(lambda x: ranking(x), axis=1) # this is for predicting the ranking instead of return itself
frame_ret = ranking_vectorized(frame_ret) # this is for predicting the ranking instead of return itself

frame_sic = frame_global.copy()
frame_sic.update(df_sic)

frame_sic = frame_sic[mask_universal]

sic_codes = sorted([i for i in frame_sic.stack().unique()])
n_sic = len(sic_codes)

#creating train frame
n_var_total =  n_var*(1 + n_macro) + n_sic
frame_train = np.empty((n_period*n_stocks,n_var_total + 1), np.float32) # +1 is for return series

##  and attach to train frame

frame_train[:,0] = frame_ret.values.flatten()

## 4. 

count_var = 1

for var2 in list_nvar:
    print(var2)
    var_temp2 = pd.read_csv(path_var+str(var2)+'.csv')
    var_temp2.index = pd.date_range(start=var_temp2.iloc[0,0] , end=var_temp2.iloc[-1,0] , freq='M')
    
    var_temp2.drop(var_temp2.columns[0], inplace=True, axis=1)
    var_temp2 = var_temp2[(var_temp2.index>=start_date) & (var_temp2.index<=end_date)]
    
#    var_temp2.index = idx_global
    
    var_global_temp = frame_global.copy()
    var_global_temp.update(var_temp2)
    
    var_global_temp = var_global_temp[mask_universal]
    #var_ranked = var_global_temp.apply(lambda x: ranking(x), axis=1)
    var_ranked = ranking_vectorized(var_global_temp)
    var_ranked.fillna(0, inplace=True)
    frame_train[:,count_var] = var_ranked.values.flatten()
    count_var += 1
    
    for var_macro in vars_macro.columns:
        
        print(var_macro)
        
        int_varmacro = var_ranked.values * vars_macro[var_macro].values.reshape(-1,1)
        
        frame_train[:,count_var] = int_varmacro.flatten()
        count_var +=1


# adding sic codes

for code in sic_codes:
    print(code)
    sic_temp = (frame_sic==code).astype(float)
    frame_train[:,count_var] = sic_temp.values.flatten()
    
    count_var += 1
        

# saving what we have

dict_data = {}
dict_data['start date'] = start_date
dict_data['end date'] = end_date
dict_data['total_year'] = total_year
dict_data['n of stocks'] = n_stocks
dict_data['index'] = idx_global 
dict_data['stocks'] = list_stocks
dict_data['n macrovars'] = n_macro
dict_data['n vars'] = n_var
dict_data['macro vars'] = vars_macro
dict_data['mcap threshold'] = mcap_threshold
#dict_data['data'] = frame_train

#with open('/home/research/Desktop/us_anomalies/data/data_ready.pickle', 'wb') as handle:
#    pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

project_name = 'data_ready_forRankingPred'

np.save('./data/'+project_name+'.npy',frame_train)
with open('./data/'+project_name+'.pickle', 'wb') as handle:
    pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

## 5. splitting data
        
#year_train = 16
#year_valid = 4
#year_test = 1
#year_lag = 1
#
#n_row_train = year_train * 12 * n_stocks
#n_row_valid = year_valid * 12 * n_stocks
#n_row_test = year_test * 12 * n_stocks
#n_row_lag = year_lag * 12 * n_stocks
#
#data_train = frame_train[-(n_row_train + n_row_valid + n_row_test + n_row_lag):-( n_row_valid + n_row_test + n_row_lag),:]
#data_valid =  frame_train[-(n_row_valid + n_row_test + n_row_lag):-(n_row_test + n_row_lag),:]
#data_test = frame_train[-(n_row_test + n_row_lag):-(+ n_row_lag),:]
#
#data_train = data_train[~np.isnan(data_train[:,0])]
#data_valid = data_valid[~np.isnan(data_valid[:,0])]
#data_test = data_test[~np.isnan(data_test[:,0])]

 
