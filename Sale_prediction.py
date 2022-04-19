#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Necessary Libraries
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
import featuretools as ft
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Imorting the datasets
train =pd.read_csv("train_kOBLwZA.csv")
test=pd.read_csv("test_t02dQwI.csv")

print(train.shape,test.shape)


# In[3]:


def concat(X,Y):
    df= pd.concat([X,Y],ignore_index=True)
    return df


# In[4]:


df=concat(train,test)
print(df.shape)


# In[5]:


df.head()


# In[7]:


df.isnull().sum()  
#Checks number of null values for all the variables
#Item_Weight has 2439 null values 
#Outlet Size has 4016 null values


# In[8]:


df.apply(lambda x: len(x.unique()))
#Checks the number of unique entries correspnding to each variable


# In[9]:


#defining a function:
#frequency of unique entries in each columns with their names

def frequency_each_item(X,Y):
    for i in Y:
        print("frequency of each category for",i)
        print(X[i].value_counts())


# In[9]:


#frequency of unique entries in each columns with their names
category=['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Size','Outlet_Type']
frequency_each_item(df,category)


# In[10]:


mode_Outlet_Size=df.pivot_table(values='Outlet_Size', index='Outlet_Type',aggfunc=(lambda x: stats.mode(x)[0]))
print(mode_Outlet_Size)
bool2=df['Outlet_Size'].isnull()
df['Outlet_Size'][bool2]=df['Outlet_Type'][bool2].apply(lambda x : mode_Outlet_Size.loc[x]).values
sum(df['Outlet_Size'].isnull())


# In[11]:


# Correcting the mis-written datas
df['Item_Fat_Content'].replace(to_replace =['low fat','reg','LF'], 
                 value =['Low Fat','Regular','Low Fat'],inplace=True)
df['Item_Fat_Content'].value_counts()
df.head()


# In[12]:


avg_item_weight=df.pivot_table(values='Item_Weight', index='Item_Identifier',aggfunc=[np.mean])
print(avg_item_weight)
bool=df['Item_Weight'].isnull()
df['Item_Weight'][bool]=df['Item_Identifier'][bool].apply(lambda x :avg_item_weight.loc[x]).values
sum(df['Item_Weight'].isnull())


# In[13]:


#Reducing food category to only 3 types with the help of the first 2 alphabets of the Item_Identifier column

df['Item_Type_combined']=df['Item_Identifier'].apply(lambda x : x[0:2])
df['Item_Type_combined'].replace(to_replace =['FD','DR','NC'], 
                 value =['Food','Drinks','Non_consumable'],inplace=True)
 #dropping the redundant column
df=df.drop(columns=['Item_Type'])     
df.head()


# In[14]:


#Calculating number of Item_fat_contents that are also non_consumable

bool3=df['Item_Type_combined']=='Non_consumable'
df['Item_Fat_Content'][bool3]='Non_edible'
df['Item_Fat_Content'].value_counts()


# In[15]:


#Using feature Engineering and adding new column
df['yearsold']=2013-df['Outlet_Establishment_Year']
df=df.drop(columns=['Outlet_Establishment_Year'])
df.head()


# In[16]:


# Converting all the zero values to mean in the visibility column
Item_Visibility_mean=df.pivot_table(index='Item_Identifier',values='Item_Visibility',aggfunc=[np.mean])
print(Item_Visibility_mean)
bool4=df['Item_Visibility']==0
df['Item_Visibility'][bool4]=df['Item_Identifier'][bool4].apply(lambda x:Item_Visibility_mean.loc[x] ).values
df.head()


# In[17]:


#Checks for correation between different numerical columns
df.corr()


# # Identifying outliers and fixing them

# In[18]:


df.describe()            


# In[19]:


sns.set(style="whitegrid")
ax = sns.boxplot(x=df["Item_Outlet_Sales"])


# In[20]:


#Only Item_Outlet_Sales have outliers we can fix them but fixing them will increase our RMSE score 
#to a large extent


# # Plotting Graphs for more Analysis

# In[21]:


#value of sales increases for the increase in MRP of the item
plt.scatter(df.Item_MRP,df.Item_Outlet_Sales,c='g')
plt.show()


# In[22]:


sns.FacetGrid(df, col='Item_Type_combined', size=3, col_wrap=5)     .map(plt.hist, 'Item_Outlet_Sales')     .add_legend();
# Maximum contribution to outlet sales is from Items that are food type and least is from drinks


# In[23]:


sns.FacetGrid(df, col='Outlet_Location_Type', size=3, col_wrap=5)     .map(plt.hist, 'Item_Outlet_Sales')     .add_legend();
#Tier3 type of outlet location provides for the maximum sales and other two provides the least sales


# In[24]:


sns.FacetGrid(df, col='Outlet_Size', size=3, col_wrap=5)     .map(plt.hist, 'Item_Outlet_Sales')     .add_legend();
#Small sized Outlets are providing the maximum sales whereas large sized outlets 
# are contributing the least


# In[25]:


sns.FacetGrid(df, col='Item_Fat_Content', size=3, col_wrap=5)     .map(plt.hist, 'Item_Outlet_Sales')     .add_legend();
# people are prefering items with lowest fat content the most 


# In[26]:


sns.FacetGrid(df, col='Outlet_Type', size=3, col_wrap=2)     .map(plt.hist, 'Item_Outlet_Sales')     .add_legend();
#Maximum of the high sales margin is from Supermarket Type1
#Grocery store has the least sales


# In[27]:


#Label Encoding all the columns with text entries and dropping Item_identifier  
le=LabelEncoder()
list=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_combined',
      'Outlet_Size']
for i in list:
    le.fit(df[i])
    df[i]=le.transform(df[i])
df_new=df.drop(columns='Item_Identifier')
df_new= pd.get_dummies(df_new,columns=['Outlet_Identifier'])
df_new.head()


# In[28]:


#Separating test and train set
df_new_train=df_new.iloc[:8523,:]
df_new_test=df_new.iloc[8523:,:]
df_new_test=df_new_test.drop(columns=['Item_Outlet_Sales'])


# In[29]:


Y_train=df_new_train['Item_Outlet_Sales']
df_train_test=df_new_train.drop(columns=['Item_Outlet_Sales'])


# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import xgboost as xgb


# In[31]:


models = [('lr',LinearRegression()),('ridge',Ridge()),('rfr',RandomForestRegressor()),('etr',ExtraTreesRegressor()),
         ('br',BaggingRegressor()),('gbr',GradientBoostingRegressor()),('en',ElasticNet()),('mlp',MLPRegressor())]


# In[34]:


#Making function for making best 2 models for further hyperparameter tuning
def basic_model_selection(x,y,cross_folds,model):
    scores=[]
    names = []
    for i , j in model:
        cv_scores = cross_val_score(j, x, y, cv=cross_folds,n_jobs=5)
        scores.append(cv_scores)
        names.append(i)
    for k in range(len(scores)):
        print(names[k],scores[k].mean())


# In[35]:


basic_model_selection(df_train_test,Y_train,4,models)


# In[36]:


#Average score for XGBoost matrix
# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=df_train_test,label=Y_train)
# import XGBRegressor
xgb1 = XGBRegressor()
cv_score = cross_val_score(xgb1, df_train_test, Y_train, cv=4,n_jobs=5)
print(cv_score.mean())


# In[37]:


def model_parameter_tuning(x,y,model,parameters,cross_folds):
    model_grid = GridSearchCV(model,
                        parameters,
                        cv = cross_folds,
                        n_jobs = 5,
                        verbose=True)
    model_grid.fit(x,y)
    y_predicted = model_grid.predict(x)
    print(model_grid.score)
    print(model_grid.best_params_)
    print("The RMSE score is",np.sqrt(np.mean((y-y_predicted)**2)))

#defining function for hyper parameter tuning and using RMSE as my metric
    


# In[50]:


parameters_xgb = {'nthread':[3,4], 
              'learning_rate':[0.02,0.03], #so called `eta` value
              'max_depth': [3,2,4],
              'min_child_weight':[3,4,5],
              'silent': [1],
              'subsample': [0.5],
              'colsample_bytree': [0.7],
              'n_estimators': [300,320]
             }
parameters_gbr={'loss':['ls','lad'],
               'learning_rate':[0.3],
               'n_estimators':[300],
               'min_samples_split':[3,4],
               'max_depth':[3,4],
               'min_samples_leaf':[3,4,2],
               'max_features':['auto','log2','sqrt']
              }

# Defining the useful parameters for parameter tuning
# to get the optimum output


# In[39]:


model_parameter_tuning(df_train_test,Y_train,xgb1,parameters_xgb,4)


# In[40]:


gbr=GradientBoostingRegressor()
model_parameter_tuning(df_train_test,Y_train,gbr,parameters_gbr,4)


# In[41]:


from sklearn.neural_network import MLPRegressor
mlp=MLPRegressor()
parameters_mlp = {'hidden_layer_sizes':[300,400,500],
              'activation':['relu','tanh'],
              'learning_rate':['adaptive'],
              'learning_rate_init':[0.001,0.004],
              'solver':['adam'],
              'max_iter':[200,300]
             }


# In[42]:


model_parameter_tuning(df_train_test,Y_train,mlp,parameters_mlp,4)


# # Standardization of the model before training

# In[43]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized=scaler.fit_transform(df_train_test)
column_names = df_train_test.columns
df_standardized = pd.DataFrame(data=standardized,columns=column_names)
df_standardized.head()


# In[44]:


basic_model_selection(df_standardized,Y_train,4,models)


# In[45]:


#Average score for XGBoost matrix
# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=df_standardized,label=Y_train)
# import XGBRegressor
xgb1 = XGBRegressor()
cv_score = cross_val_score(xgb1, df_standardized, Y_train, cv=4,n_jobs=5)
print(cv_score.mean())


# In[46]:


model_parameter_tuning(df_standardized,Y_train,xgb1,parameters_xgb,4)


# In[47]:


model_parameter_tuning(df_standardized,Y_train,gbr,parameters_gbr,4)


# In[49]:


df_train_test.head()


# # Using Robust Scaler
# 
# 

# In[51]:


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

normalize = MinMaxScaler()
robust = RobustScaler(quantile_range = (0.1,0.8)) #range of inerquartile is one of the parameters
robust_stan = robust.fit_transform(df_train_test)
robust_stan_normalize = normalize.fit_transform(robust_stan)
# also normalized the dataset using MinMaxScaler i.e has bought the data set between (0,1)
df_robust_normalize = pd.DataFrame(robust_stan_normalize,columns=column_names)
df_robust_normalize.head()


# In[52]:


basic_model_selection(df_robust_normalize,Y_train,4,models)


# In[53]:


cv_score = cross_val_score(xgb1, df_robust_normalize, Y_train, cv=4,n_jobs=5)
print(cv_score.mean())


# In[54]:


model_parameter_tuning(df_robust_normalize,Y_train,xgb1,parameters_xgb,4)


# In[55]:


model_parameter_tuning(df_robust_normalize,Y_train,gbr,parameters_gbr,4)


# # Best Model 
# 

# # Gradient Boosting Method is the best method 
#  PARAMETERS AND RMSE RESPECTIVELY
#  {'learning_rate': 0.3, 'loss': 'lad', 'max_depth': 3, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}
# The RMSE score is 1049.14085875651
# 

# In[56]:


robust_test = robust.fit_transform(df_new_test)
robust_normalize_test = normalize.fit_transform(robust_test)
df_test_robust_normalize = pd.DataFrame(robust_normalize_test,columns=column_names)


# In[59]:


gbr = GradientBoostingRegressor(learning_rate= 0.3, loss= 'lad',max_depth= 3,min_samples_leaf=2,min_samples_split=3
                                ,n_estimators= 300)
# Defining my final model that I will use for prediction


# In[60]:


gbr.fit(df_robust_normalize,Y_train)


# In[61]:


final_prediction=gbr.predict(df_test_robust_normalize) #Predicting the outlet sales


# In[65]:



# Converting into Dataframe 
df_final_prediction = pd.DataFrame(final_prediction,columns=['Item_Outlet_Sales'])


# In[66]:


df_final_prediction.head()


# In[62]:


import joblib
filename = 'final_model.sav' 
joblib.dump(gbr, filename)


# In[67]:



load_model = joblib.load(filename)

