#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[19]:


# car_data = pd.read_csv('C:/Users/admin/Desktop/mudassir/New folder/OLX_Car_Data_CSV')
df = pd.read_csv("C:/Users/admin/Desktop/olx_car_data.csv",  encoding= 'unicode_escape')
print(df)


# In[21]:


df.head()


# In[22]:


print(f"The value counts of Car Brands:\n{df['Brand'].value_counts()}")


# In[23]:


df['Model'].nunique()


# In[24]:


df['Transaction Type'].unique()


# In[25]:


sns.heatmap(pd.DataFrame(df.isnull().sum()),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1)


# In[26]:


df[df.isnull().any(axis=1)].shape


# In[27]:


df = df.fillna(df.mode().iloc[0])


# In[28]:


df.info()


# In[29]:


to_drop = []
cat_features = ['Brand','Model','Registered City']
down_limit = 0.02 * len(df)
for feature in cat_features:
    unique_values = df[feature].unique()
    to_drop = [val for val in unique_values if (list(df[feature]).count(val) < down_limit)]
    print('\n', to_drop, 'are now Other')
    df[feature].mask(df[feature].isin(to_drop), 'Others', inplace=True)
    temp_df = pd.get_dummies(df[feature], prefix=feature, dtype=np.float64)
    df = pd.concat([df, temp_df], axis=1).drop([feature], axis=1)
    print('{} Categorized'.format(feature), '\n')


# In[30]:


df.columns


# In[31]:


df.head()


# In[33]:


sns.distplot(df['Price'])


# In[34]:


f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(15,3))
ax1.scatter(df['Year'], df['Price'])
ax1.set_title('Price vs Year')
ax2.scatter(df['KMs Driven'], df['Price'])
ax2.set_title('Price vs KMs Driven')


# In[35]:


data_with_dummies = pd.get_dummies(df)
data_with_dummies.head()


# In[36]:


data_with_dummies.columns


# In[37]:


len(df.loc[df["Price"] > 2000000].index)


# In[38]:


data_with_dummies.drop(data_with_dummies[data_with_dummies['Price'] > 2000000].index,inplace=True)
data_with_dummies.info()


# In[39]:


from pylab import rcParams

rcParams['figure.figsize'] = 12.7, 8.27
g = sns.kdeplot(data_with_dummies['Price'], color="red", shade = True)
g.set_xlabel("Price")
g.set_ylabel("Frequency")
plt.title('Distribution of Price',size = 20)


# In[40]:


data_with_dummies["Price Period"] = pd.cut(data_with_dummies.Price,
                                    5,
                                    labels=['Very cheap',
                                            'Cheap', 'Normal',
                                            'Expensive', 'Very expensive'])

data_with_dummies.head()


# In[41]:


data_with_dummies = data_with_dummies.drop(axis=1,columns='Price')
data_with_dummies.head()


# In[43]:


data_with_dummies['Price Period'].unique()


# In[44]:


sns.catplot(x="Price Period", y="Year", data=data_with_dummies);
plt.xticks(rotation=45)


# In[45]:


data_with_dummies = data_with_dummies.sample(frac=1).reset_index(drop=True)


# In[46]:


df_Target = data_with_dummies.iloc[:,35:36]
col_no_price = data_with_dummies.drop(columns='Price Period')


# In[75]:


X = np.array(col_no_price.values.tolist())
y = np.array(df_Target.values.tolist())
print(X)
print(y)


# In[76]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[77]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, Y_train)
print("training accuracy :", tree_model.score(X_train, Y_train))
print("testing accuracy :", tree_model.score(X_test, Y_test))


# In[78]:


print(tree_model.tree_.max_depth)


# In[80]:


from sklearn.tree import export_graphviz
import graphviz

dot_file = export_graphviz(tree_model,out_file=None,filled=True
                           ,feature_names=col_no_price.columns,rounded=True,max_depth=5)

graphviz.Source(dot_file)


# In[ ]:




