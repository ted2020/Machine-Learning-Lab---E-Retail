#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
import datetime
from sklearn import preprocessing
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.api as sm
from scipy import stats
from statsmodels.compat import lzip
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics.regressionplots import *
from yellowbrick.regressor import CooksDistance
from yellowbrick.datasets import load_concrete
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from yellowbrick.model_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from yellowbrick.datasets import load_credit
from yellowbrick.target import FeatureCorrelation
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[2]:


orders=pd.read_csv("Orders.csv")


# In[3]:


returns=pd.read_csv("Returns.csv")


# # orders dataset

# In[4]:


orders.columns[orders.isnull().any()]


# In[5]:


returns.columns[returns.isnull().any()]


# In[6]:


orders.shape


# In[7]:


orders=orders.drop(["Postal.Code"],axis=1)


# In[8]:


orders.shape


# In[9]:


orders["Order.ID"].nunique()


# In[10]:


returns["Order ID"].nunique()


# In[11]:


returns.shape


# In[12]:


print(returns.dtypes),print(orders.dtypes)


# In[13]:


orders["Customer.Name"].nunique()


# In[14]:


len(orders)


# ## check date object

# In[15]:


orders["Profit"]=orders["Profit"].str.replace("$","")
orders["Profit"]=orders["Profit"].str.replace(",","")
orders["Profit"]=orders["Profit"].str.replace(".","")
orders["Sales"]=orders["Sales"].str.replace("$","")
orders["Sales"]=orders["Sales"].str.replace(",","")
orders["Sales"]=orders["Sales"].str.replace(".","")


# In[16]:


orders["Profit"]=orders["Profit"].astype(float)
orders["Sales"]=orders["Sales"].astype(float)


# In[17]:


orders['Ship.Date'] = pd.to_datetime(orders['Ship.Date'])
orders['Order.Date'] = pd.to_datetime(orders['Order.Date'])


# In[18]:


orders['day'] = pd.DatetimeIndex(orders['Ship.Date']).day
orders['month'] = pd.DatetimeIndex(orders['Ship.Date']).month
orders['year'] = pd.DatetimeIndex(orders['Ship.Date']).year


# In[19]:


orders["diff"]=orders["Ship.Date"] - orders["Order.Date"]
orders["diff2"]=orders["diff"].dt.days
orders["diff"].sort_values(ascending=False).head()


# In[ ]:





# ## groupby category, sub, and customer name

# In[20]:


(orders.groupby("Category").agg(['count', 'sum', 'min', 'max', 'mean', 'std'])["Discount"].sort_values("mean",ascending=False)).head(10)


# In[21]:


(orders.groupby("Sub.Category").agg(['count', 'sum', 'min', 'max', 'mean', 'std'])['Discount'].sort_values("mean",ascending=False)).head(10)


# In[22]:


(orders.groupby("Sub.Category").agg(['count', 'sum', 'min', 'max', 'mean', 'std'])["Sales"].sort_values("mean",ascending=False)).head(10)


# In[23]:


(orders.groupby("Sub.Category").agg(['count', 'sum', 'min', 'max', 'mean', 'std'])["Profit"].sort_values("mean",ascending=False)).head(10)


# In[24]:


(orders.groupby("Customer.Name").agg(['count', 'sum', 'min', 'max', 'mean', 'std'])['Quantity'].sort_values("mean",ascending=False)).head(10)


# # visualization

# In[25]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="Market", y="Quantity", hue="month", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Market")


# In[26]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="Market", y="Quantity", hue="year", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Market")


# In[27]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="year", y="Quantity", hue="month", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Category")


# In[28]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="Category", y="Quantity", hue="month", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Category")


# In[29]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="Category", y="Quantity", hue="year", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Category")


# In[30]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="Sub.Category", y="Quantity", hue="month", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Sub.Category")
g.set_xticklabels(rotation=90)


# In[31]:


sns.set(style="whitegrid")
sns.set(font_scale=2)
g = sns.catplot(x="Sub.Category", y="Quantity", hue="year", data=orders,
                height=11, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Quantity")
g.set_xlabels("Sub.Category")
g.set_xticklabels(rotation=90)


# In[32]:


f, ax = plt.subplots(figsize=(15, 9))
sns.set(font_scale=2)
g=sns.lineplot(x="Ship.Date", y="Profit", data=orders)
ax.set(xlabel='ship date', ylabel='profit')


# In[33]:


sns.set(style="ticks")
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(8, 8))
ax.set_xscale("log")
my_order = orders.groupby(["Category"])["Profit"].median().sort_values(ascending=False).head(10).index
sns.boxplot(x="Profit", y="Category", data=orders,
            whis="range", palette="vlag",order=my_order)
ax.xaxis.grid(True)
ax.set(xlabel="Profit")
sns.despine(trim=True, left=True)


# In[34]:


sns.set(style="ticks")
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(8, 8))
ax.set_xscale("log")
my_order = orders.groupby(["Sub.Category"])["Profit"].median().sort_values(ascending=False).head(10).index
sns.boxplot(x="Profit", y="Sub.Category", data=orders,
            whis="range", palette="vlag",order=my_order)
ax.xaxis.grid(True)
ax.set(xlabel="Profit")
sns.despine(trim=True, left=True)


# # returns dataset

# In[35]:


returns["Order.ID"]=returns["Order ID"]


# In[36]:


returns.drop(["Order ID","Region"], axis=1, inplace=True)


# In[37]:


returns.dtypes


# # merge orders and returns

# In[38]:


df2=pd.merge(orders,returns,on="Order.ID")


# In[39]:


df2.columns[df2.isnull().any()]


# ## groupby

# In[40]:


h=(df2.groupby("Customer.Name").agg({"Profit":["mean"],"Quantity":["mean"]}))
h.columns = ["_".join(x) for x in h.columns.ravel()]
h.sort_values("Profit_mean",ascending=False).head(10)


# In[41]:


h=(df2.groupby("Customer.Name").agg({"Profit":["mean"],"Quantity":["mean"]}))
h.columns = ["_".join(x) for x in h.columns.ravel()]
h.sort_values("Quantity_mean",ascending=False).head(10)


# In[42]:


h=(df2.groupby("Customer.Name").agg({"Profit":["mean"],"Quantity":["mean"]}))
h.columns = ["_".join(x) for x in h.columns.ravel()]
h.sort_values("Quantity_mean",ascending=False).head(10)


# In[43]:


(h[h["Quantity_mean"]>1]).count()


# In[44]:


df2["Profit"].groupby(by=df2["year"]).sum()


# In[45]:


k=pd.DataFrame(df2["Returned"].groupby(by=df2["Customer.Name"]).count().sort_values(ascending=False))
k.head(10)


# In[46]:


(k[k["Returned"]>1]).count()


# In[47]:


(k[k["Returned"]>5]).count()


# In[48]:


k=pd.DataFrame(df2["Returned"].groupby(by=df2["Region"]).count().sort_values(ascending=False))
k


# In[49]:


k=pd.DataFrame(df2["Returned"].groupby(by=df2["Category"]).count().sort_values(ascending=False))
k


# In[50]:


k=pd.DataFrame(df2["Returned"].groupby(by=df2["Sub.Category"]).count().sort_values(ascending=False))
k


# # Machine Learning

# In[51]:


df3=pd.merge(orders,returns, how= "outer",on="Order.ID")


# In[52]:


df3.columns[df3.isnull().any()]


# In[53]:


print(df3.isnull().sum().sort_values(ascending=False))


# In[54]:


df3 = df3.replace(np.nan,'None')


# In[55]:


df3["Returned"].unique()


# In[56]:


df3["Returned"] = df3.Returned.eq('Yes').mul(1)


# In[57]:


k=pd.DataFrame(df3["Product.ID"].groupby(by=df3["Product.ID"]).count().sort_values(ascending=False))
k.head()
# k["Product.ID"].sum()


# In[58]:


m=pd.DataFrame(df3[df3["Returned"]==0].groupby(by=df3["Product.ID"]).count())
m["Returned"].sort_values(ascending=False).head()


# In[59]:


n=pd.DataFrame(df3[df3["Returned"]>0].groupby(by=df3["Product.ID"]).count())
n["Returned"].sort_values(ascending=False).head()


# In[60]:


# (n[n["Returned"]>0]).count()


# In[61]:


df3["Product.ID"].nunique()


# In[62]:


df3[["Product.ID","Returned"]].head()


# In[63]:


# Return_count=Orders4[Orders4[“Returned”]>0].groupby(by=Orders4[“Product.ID”]).count()[[“Returned”]]

# Orders4 = Orders4.merge(Return_count,on=‘Product.ID’,how=‘outer’)

# Orders4 = Orders4[[‘Returned_y’]].replace(np.nan,0)


# In[64]:


# df3[["Product.Name","Segment"]].head()


# # prep for machine learning

# In[65]:


df4=df3[['Ship.Mode',       'Segment', 'Country',       'Market', 'Category', 'Sub.Category',       'Sales', 'Quantity', 'Discount', 'Profit',       'Shipping.Cost', 'Order.Priority', 'day', 'month', 'year', 'Returned',       'diff2']]


# In[66]:


df4.head()


# In[67]:


df4_objects = df4.select_dtypes(include=[object])
df4_objects.head()


# In[68]:


df4_objects.columns


# In[69]:


df4_objects.shape


# In[70]:


df4_objects2=pd.get_dummies(df4_objects,drop_first=True)
df4_objects2.head()


# In[71]:


df5=pd.concat([df4,df4_objects2],axis=1)


# In[72]:


df5.drop(['Ship.Mode', 'Segment', 'Country', 'Market', 'Category', 'Sub.Category',
       'Order.Priority'],axis=1, inplace=True)


# In[73]:


df5=df5.apply(pd.to_numeric,downcast='integer')


# In[ ]:





# # randomforest

# In[74]:


y=df5.loc[:, df5.columns == 'Returned']


# In[75]:


X=df5.loc[:, df5.columns != 'Returned']


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[77]:


regressor = RandomForestRegressor(n_estimators=25, random_state=0)
regressor.fit(X_train, y_train)


# In[78]:


y_pred = regressor.predict(X_test)


# In[79]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[80]:


df5["Returned"].std()


# # logistic regression

# In[106]:


X.columns[X.isnull().any()]


# In[107]:


y.columns[y.isnull().any()]


# In[81]:


logit_model=sm.Logit(y,X)
result=logit_model.fit(method="bfgs")
print(result.summary2())


# In[82]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[83]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[84]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[85]:


print(classification_report(y_test, y_pred))


# In[86]:


logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:




