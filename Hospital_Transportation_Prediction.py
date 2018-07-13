
# coding: utf-8

# # Hospital Transportation
# by Martin Siebert, 2018

# - The Goal is to predict the transport time of inter hospital transportation
# - Labeled Data is given with the following file: Krankenhaus_Transporte_working.xlsx
# - Label to be predicted: "DI"

# ## Import Dependencies and Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# display options
get_ipython().magic('matplotlib inline')
pd.options.display.max_columns = None
pd.options.display.max_rows = 200
# do not display uncritical warnings for better readability
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# In[2]:


# import data to dataframe
df = pd.read_excel("../Data/Krankenhaus_Transporte.xlsx", index=0)


# In[3]:


# display head of data for first peek at data
df.head(3)


# # 1. Create Data Understanding

# #### Clean Dataset

# In[4]:


# display possible values for Status
df.STATUS.unique()


# In[5]:


# Delete rows for Status: sto (storniert),"X" (deleted), "XA" (deleted old)
# Save in new df for preprocessed data
df_pre = df[~df['STATUS'].str.contains("sto")]
df_pre = df_pre[~df_pre['STATUS'].str.contains("X")]
df_pre = df_pre[~df_pre['STATUS'].str.contains("XA")]


# In[6]:


# check if deletion was successful 
df_pre.STATUS.unique()


# ##### Reduce DataFrame to Relevant Features

# In[7]:


# create working copy of df with only relevant features
# most features are generated as transport is happening or is already finished
# since these features are not available at time of prediction they are not included into the dataset
df_pre = df_pre[["DATUM_TAG","DATUM_MONAT","DATUM_JAHR","SO","GA",
              "PRIO","VON","NACH","TERMIN_STUNDE","TERMIN_MINUTE",
              "BS_STUNDE","BS_MINUTE","DI","DS","TM","ART"]].copy()


# In[8]:


# features remaining for training the models
df_pre.columns


# In[9]:


# Shape of DataFrame before and after row deletion
print ("Original Data: " + str(df.shape))
print ("Cleansed Data: " + str(df_pre.shape))


# In[10]:


# Get initial values for later evaluation of results

# print mean of DI and mean absolute deviation(MAD) between DI hospital estimate DS
# outputs are in minutes
print("Mean of DI: " + str(df_pre.DI.mean()))
print("MAD DI-DS: " + str((df_pre["DI"]-df_pre["DS"]).mean()))


# #### Get to know the data set

# In[11]:


# import additional dependencies for plotting
from ggplot import geom_histogram, geom_density
from ggplot import *


# In[12]:


# Distribution of target variable
ggplot(aes(x='DI',),data=df_pre) + geom_histogram(binwidth=2,alpha=0.6, fill="#008080", color= "#20b2aa")+ xlab("DI") + ggtitle("Distribution of DI")


# # 2. Outlier detection and handling

# In[13]:


# possible negative values in distribution
# check for negative values of DI
df_pre[df_pre["DI"]<0]


# In[14]:


# duration of transportion cannot be negative
# delete negative occurences of DI
df_pre = df_pre[~df_pre["DI"]<0]


# In[15]:


# check distribution after deletion
ggplot(aes(x='DI',),data=df_pre) + geom_histogram(binwidth=2,alpha=0.6,fill="#008080", color= "#20b2aa")+ xlab("Messwerte") + ggtitle("Verteilung DI")


# In[16]:


# are all occurences greater for DI>120 rightfully expected values


# In[17]:


# Which large values for Di do we expect?
df_pre[df_pre["DS"]>120].DS.unique()


# In[18]:


# How many occurences greater than 120 exist
df_pre[df_pre["DI"]>120].count()


# In[19]:


# check where these are supposed to originate from
df_pre[df_pre["DS"]>120].VON.unique()


# In[20]:


# But where do they originate from?
len(df_pre[df_pre["DI"]>120].NACH.unique())


# In[21]:


# check how many occurences of DI> 120 and not VON= D-LS Pat exist
(df_pre[(df_pre.VON!="D-LS Pat") & (df_pre.DI>120)]).count()


# In[22]:


# check occurences of DI> 120 and not VON= D-LS Pat
(df_pre[(df_pre.VON!="D-LS Pat") & (df_pre.DI>120)])


# In[23]:


# large deviations from expected duration with rare occurences 
# for combination of VON-NACH can be classified as outliers (without VON: D-LS Pat)


# In[24]:


# delete those occurences
df_pre = df_pre[~((df_pre.VON!="D-LS Pat") & (df_pre.DI>120))]


# In[25]:


# check observations DI>500
df_pre[df_pre["DI"]>520].count()
df_pre[df_pre["DI"]>520].head()


# In[26]:


# delete occurences greater DI>520
df_pre = df_pre[~(df_pre.DI>520)]


# #### Undertaken attempts to cleaning data without success

# In[27]:


# CHECK FOR PERFORMANCE
# SO =2 (extern) DI größer 60 (max DS Wert 32) löschen
#df_pre[df_pre["SO"]==2][df_pre["DI"]>60]
#df_pre = df_pre[~((df_pre.SO==2) & (df_pre.DI>60))]
#df_pre[df_pre["SO"]==2][df_pre["DI"]>60]


# In[28]:


# CHECK FOR PERFORMANCE
# 5302 Fusstransporte länger als 2h
#df_pre[df_pre["TM"]=="Fuss"][df_pre["DI"]>120].count()


# In[29]:


# EXPERIMENT: delete all rows for DI>90
#df_pre = df_pre[~(df_pre.DI>90)]
# --> Leads to Performance increase, but is not a valid approach, since model cannot model certain occurences


# In[30]:


# with an expected deviation of up to 50 min  we should observe any values between 280 and 430 
# those observations can deleted as outliers
# --> war vielleicht ein wenig zu viel mit alles zweischen 280 und 430 zu löschen. Sowie größer 500 Leichte Performance Einbuße
#df_pre = df_pre[~((df_pre.DI>300) & (df_pre.DI<400))]
# also delete 12 occurances of DI > 500
#df_pre = df_pre[~(df_pre.DI>120)]

#--> war dann doch zu radikal


# In[31]:


# all DS greater than 60 come from D_LS Pat
# all DS greater than 90 go to D-ZLab or D-LS Pat
# --> create new feature just for this? Or delete these instances^


# # 3. Handling Missing Data

# In[32]:


# check for missing data and display top 5
total = df_pre.isnull().sum().sort_values(ascending=False)
percent = (df_pre.isnull().sum()/df_pre.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[33]:


# Plot missing data
f, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation='90')
sns.barplot(x=percent.head(10).index, y=percent.head(10))
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[34]:


# fill missing values TERMIN_MINUTE and TERMIN_STUNDE with 0 (led to better results than -10 or mean)
df_pre.TERMIN_MINUTE.fillna(df_pre.TERMIN_MINUTE.median(), inplace=True)
df_pre.TERMIN_STUNDE.fillna(df_pre.TERMIN_STUNDE.median(), inplace=True)


# In[35]:


# fill missing data for VON and NACH with new location "Unknown"
df_pre.VON.fillna("Unknown", inplace=True)
df_pre.NACH.fillna("Unknown", inplace=True)


# In[36]:


# delete missing data for VON and NACH
# not proven to be the best solution
#df_pre = df_pre[~df_pre.VON.isnull()]
#df_pre = df_pre[~df_pre.NACH.isnull()]


# In[37]:


# check for remaining missing values
df_pre.isnull().values.any()


# # 4. Feature Engineering

# In[38]:


# BS is available as Minute and Hour value, the date is saved in the DATUM variables
# create timestamp, combinging all of that information
df_temp = pd.DataFrame({'year': df_pre.DATUM_JAHR,
                'month': df_pre.DATUM_MONAT,
                'day': df_pre.DATUM_TAG,
                'hour': df_pre.BS_STUNDE,
                'minute': df_pre.BS_MINUTE})
df_pre["BS"] = pd.to_datetime(df_temp)


# In[39]:


# create timestamp for TERMIN
df_temp = pd.DataFrame({'year': df_pre.DATUM_JAHR,
                'month': df_pre.DATUM_MONAT,
                'day': df_pre.DATUM_TAG,
                'hour': df_pre.TERMIN_STUNDE,
                'minute': df_pre.TERMIN_MINUTE})
df_pre["TERMIN"] = pd.to_datetime(df_temp)


# In[40]:


# create features for the weekday 0=Monday, 6=Sunday, expecting some seasonality in data
df_pre["WOCHENTAG"] = df_pre["BS"].dt.dayofweek


# In[41]:


# drop DATUM fields as information is encoded in new date fields
df_pre.drop(["DATUM_TAG","DATUM_MONAT","DATUM_JAHR"],axis=1, inplace=True)


# In[42]:


# create distance feature as the combination of VON and NACH
df_pre["DIST"] = df_pre.VON + df_pre.NACH


# In[43]:


# create feature for difference between BS and Termin
# TERMIN which were manually filled will lead to large differences (-1 day etc.) These large diff signify "no Termin"
df_pre["TERMIN-BS"] = df_pre.TERMIN-df_pre.BS


# In[44]:


df_pre["VON-PRIO"] = df_pre.VON+str(df_pre.PRIO)


# In[45]:


# check data distribution
ggplot(aes(x='BS',y="DI", color="SO"),data=df_pre) + geom_point(alpha=0.6,)


# In[46]:


# check data structure
df_pre.head(10)


# # 5. Data transformation

# In[47]:


# Label encoder for DIST
#from sklearn import preprocessing
# Create a label (category) encoder object
#le = preprocessing.LabelEncoder()
#df_simpl["DIST_LE"] = le.fit_transform(df_simpl["DIST"].astype(str))


# In[48]:


# Encode categorical features which may contain information in their ordering set
from sklearn.preprocessing import LabelEncoder
cols = ('PRIO', "DIST")
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df_pre[c].values)) 
    df_pre[c] = lbl.transform(list(df_pre[c].values))


# In[49]:


# should I stay or should I go
#df_simpl['DS'].corr(df_simpl['DIST_LE'])


# In[50]:


#ggplot(aes(x='BS',y="DI", color="SO"),data=df_simpl) + geom_point(alpha=0.6,)


# In[51]:


#ggplot(aes(x='DI'),data=df_simpl) + geom_density(alpha=0.6)


# In[52]:


#check data types of columns for num which are cat
#df_simpl.dtypes


# In[53]:


#df_simpl.describe()


# #### Prepare Data for Training 

# In[54]:


# create working copy of df
df_train = df_pre.copy()


# In[55]:


# for models to handle, parse Datetime to int
df_train["BS"] = df_train["BS"].astype(int)
df_train["TERMIN"] = df_train["TERMIN"].astype(int)
df_train["TERMIN-BS"] = df_train["TERMIN-BS"].astype(int)


# In[56]:


# create dummy variables for all categorial features
df_train = pd.get_dummies(df_train)
print(df_train.shape)


# In[57]:


# drop hospital estimation DS to be independant of their results
df_train.drop("DS",axis=1,inplace=True)


# In[58]:


# log1p transformation for skewed DI makes it easier for moedl to infer linear relationships
df_train["DI"] = np.log1p(df_train["DI"])


# In[59]:


# plot density of log1p transformed DI
ggplot(aes(x='DI',),data=df_train) + geom_histogram(binwidth=.15,alpha=0.6, fill="#008080",color="#008080")+ xlab("Values") + ggtitle("Distribution log1p transformed DI")


# # 6. Train Models

# In[60]:


# define target variable DI as y
y = df_train.DI


# In[61]:


# drop target variable from training data
df_train.drop(["DI"], axis=1, inplace = True)


# In[62]:


# didn't lead to better prediction, but much longer calc time
# transform features by scaling each feature to tange(0,1)
#from sklearn.preprocessing import MinMaxScaler
#df_train = MinMaxScaler().fit_transform(df_train)


# In[63]:


# create training- test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# #### Define and Train Models

# In[64]:


# define and train RandomForrest Regressor
from sklearn.ensemble import RandomForestRegressor
regr_rf = RandomForestRegressor(max_depth=30, random_state=42)
regr_rf.fit(X_train, y_train)

# get predictions
predictions_rf = regr_rf.predict(X_test)


# In[65]:


# inverse transformation of log1p
predictions_rf_n = np.expm1(predictions_rf)
y_test_n = np.expm1(y_test)


# In[66]:


regr_rf.score(X_test, y_test_n)
# 0.60608954771513657


# In[67]:


from sklearn.metrics import mean_squared_error
from math import sqrt
print("Baseline: " + str(sqrt(mean_squared_error(df_pre["DI"], df_pre["DS"]))))
      
print("RMSE: " +str(sqrt(mean_squared_error(y_test_n, predictions_rf_n))))

# 17.998355536788985
# 14.125812814012203 mit DI<90


# In[68]:


from sklearn.metrics import mean_absolute_error
print("Baseline: " + str(mean_absolute_error(df_pre["DI"], df_pre["DS"])))
      
print("MAE: " +str(mean_absolute_error(y_test_n, predictions_rf_n)))
# 10.8692601982 
# 10.1663389821 mit DI<90


# In[69]:


# ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor

model_extraTree = ExtraTreesRegressor(random_state=42)
model_extraTree.fit(X_train,y_train)

predictions_extraTree = model_extraTree.predict(X_test)
predictions_extraTree_n = np.expm1(predictions_extraTree)
model_extraTree.score(X_test,y_test_n)


# In[70]:


#SVR (Support Vector Regressor)
from sklearn.svm import LinearSVR

model_svr= LinearSVR(random_state=42)
model_svr.fit(X_train,y_train)

predictions_svr = model_svr.predict(X_test)
predictions_svr_n = np.expm1(predictions_svr)
model_svr.score(X_test,y_test_n)


# In[71]:


# Decision Tree (decision tree regressor)
from sklearn.tree import DecisionTreeRegressor
model_tree = DecisionTreeRegressor(random_state=42)
model_tree.fit(X_train,y_train)

predictions_tree = model_tree.predict(X_test)
predictions_tree_n = np.expm1(predictions_tree)             
model_tree.score(X_test,y_test_n)


# In[72]:


# Elastic Net (linear regression model)
from sklearn.linear_model import ElasticNet
model_enet = ElasticNet(random_state=42)
model_enet.fit(X_train,y_train)

predictions_enet = model_enet.predict(X_test)
predictions_enet_n = np.expm1(predictions_enet)
model_enet.score(X_test,y_test_n)


# In[73]:


# XGBoostRegressor
import xgboost as xgb
model_xgb = xgb.XGBRegressor(max_depth=30, random_state=42)
model_xgb.fit(X_train, y_train)
predictions_xgb = model_xgb.predict(X_test)
predictions_xgb_n = np.expm1(predictions_xgb)
model_xgb.score(X_test, y_test_n)


# In[74]:


print("RMSE ")
print("Baseline: " + str(sqrt(mean_squared_error(df_pre["DI"], df_pre["DS"]))))
print("RandomForest: " +str(sqrt(mean_squared_error(y_test_n, predictions_rf_n))))
print("ExtraTrees: " +str(sqrt(mean_squared_error(y_test_n, predictions_extraTree_n))))
print("SVR: " +str(sqrt(mean_squared_error(y_test_n, predictions_svr_n))))
print("DecisionTree: " +str(sqrt(mean_squared_error(y_test_n, predictions_tree_n))))
print("ElasticNet: " +str(sqrt(mean_squared_error(y_test_n, predictions_enet_n))))
print("XGBoost: " +str(sqrt(mean_squared_error(y_test_n, predictions_xgb_n))))


# In[75]:


print("MAE ")
print("Baseline: " + str(mean_absolute_error(df_pre["DI"], df_pre["DS"])))
print("RandomForest: " +str(mean_absolute_error(y_test_n, predictions_rf_n)))
print("ExtraTrees: " +str(mean_absolute_error(y_test_n, predictions_extraTree_n)))
print("SVR: " +str(mean_absolute_error(y_test_n, predictions_svr_n)))
print("DecisionTree: " +str(mean_absolute_error(y_test_n, predictions_tree_n)))
print("ElasticNet: " +str(mean_absolute_error(y_test_n, predictions_enet_n)))
print("XGBoost: " +str(mean_absolute_error(y_test_n, predictions_xgb_n)))


# #### Feature Importances

# In[76]:


# Get feature importances from RandomForrestRegressor and XGBoost
# see sklearn for description: The feature importances. The higher, the more important the feature. The importance of a feature is computed as the (normalized) 
# total reduction of the criterion brought by that feature. It is also known as the Gini importance [R251].

df_impo = pd.DataFrame()
df_impo["Name"] = X_train.columns
df_impo["Importances_rf"] = regr_rf.feature_importances_
df_impo["Importances_xgb"] = model_xgb.feature_importances_


# In[77]:


df_impo.sort_values(by=['Importances_xgb'], ascending=False).head(15)


# #### Export Tree Viz

# In[78]:


# Export exemplary Decision Tree
# from sklearn import tree
# export_graphviz(model_tree.estimators_[0],feature_names = df_model.columns,
#                filled=True, rounded=True, out_file='tree.dot')

