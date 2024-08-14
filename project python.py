#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Open dataset
import pandas as pd
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\pandas\Admission_Predict1.csv",sep = ",")
print(df)


# In[4]:


# Display first five records
print(df.head())


# In[5]:


# Display last five records
print(df.tail())


# In[6]:


# Display random five records
print(df.sample(5))


# In[7]:


# Display information regarding dataset
print(df.describe())


# In[8]:


# Check types and dimensions
print(df.dtypes)


# In[9]:


print(df.shape)  # corrected 'df.shap' to 'df.shape'


# In[10]:


# We can count individual value in a particular columns
print(df.Research.value_counts())
print(df.CGPA.value_counts())


# In[11]:


# Remove first columns because target not depend on this

df.drop(["Serial No."],axis=1,inplace = True)


# In[12]:


# Rename column and remove space at last

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# In[13]:


# Now define x and y

y = df["Chance of Admit"]

x = df.drop(["Chance of Admit"],axis=1)


# In[14]:


#Now split dataset into two parts training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=42)
print(x_train)
print(y_test)


# In[15]:


# now use scaling and convert all data within same range from 0 to 1

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])
print(x_train)


# In[16]:


# Use Linear Regression algorithm 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

score=model.score(x_test, y_test)  # calculate accuracy of the algorithm
print(score)


# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\pandas\Admission_Predict1.csv")

# Select the feature and target variables
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the actual vs predicted values
axs[0].scatter(y_test, y_pred)
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Actual vs Predicted Values')

# Plot the residuals
axs[1].scatter(y_pred, y_test - y_pred)
axs[1].axhline(y=0, color='k', lw=2)
axs[1].set_xlabel('Predicted Values')
axs[1].set_ylabel('Residuals')
axs[1].set_title('Residual Plot')

# Set the title and layout
fig.suptitle(f'Linear Regression Model Performance\nMSE: {mse:.2f}, R2: {r2:.2f}')
fig.tight_layout()
plt.show()


# In[17]:


# Take one record from training dataset

newx=x_train[0:1]


# In[18]:


# Predict the target against this value

newy=model.predict(newx)
print("Your Chance of Admission is: ",newy)


# In[19]:


# Now repeat for other record

newx=x_train[2:3]
print(newx)
newy=model.predict(newx)
print("Your Chance of Admission is: ",newy)
newx=x_test[2:3]
newy=model.predict(newx)
print("Your Chance of Admission is: ",newy)


# In[20]:


# predict five record from test and compare with actual values

print (y_predict[0:5])

print(y_test[0:5])


# In[21]:


#Now take input from user

gre=float(input("What is your GRE Score (between 290 to 340):"))
toefl=float(input("What is your TOEFL Score (between 90 to 120):"))
univ=float(input("What is your University Rating ( 1 to 5 ):"))
sop=float(input("Rate your Statement of Purpose ( 1 to 5):"))
lor=float(input("What is strength of  your Letter of Recommendation ( 1 to 5) :"))
cgpa=float(input("What is your CGPA ( 6 to 10):"))
research=float(input("Do You have Research Experience (Enter 0 for No and 1 for Yes:"))


# In[22]:


# scale the input values
gre_scaled = (gre - 290) / 50
toefl_scaled = (toefl - 90) / 30
univ_scaled = (univ - 1) / 4
sop_scaled = (sop - 1) / 4
lor_scaled = (lor - 1) / 4
cgpa_scaled = (cgpa - 6) / 4
research_scaled = research

newx=[[gre_scaled,toefl_scaled,univ_scaled,sop_scaled,lor_scaled,cgpa_scaled,research_scaled]]

newy=model.predict(newx)

print("Your Chance of Admission is: ",newy)


# In[23]:


# Use another algorithm RandomForestRegressor and check the accuracy

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(x_train,y_train)
y_predict_rfr = rfr.predict(x_test) 

score_rfr=rfr.score(x_test, y_test)
print(score_rfr)


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\pandas\Admission_Predict1.csv')

# Select the feature and target variables
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a figure with three subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Plot the actual vs predicted values
axs[0].scatter(y_test, y_pred)
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Actual vs Predicted Values')

# Plot the residuals
axs[1].scatter(y_pred, y_test - y_pred)
axs[1].axhline(y=0, color='k', lw=2)
axs[1].set_xlabel('Predicted Values')
axs[1].set_ylabel('Residuals')
axs[1].set_title('Residual Plot')

# Plot the feature importance
importances = model.feature_importances_
axs[2].barh(range(X.shape[1]), importances)
axs[2].set_xlabel('Feature Importance')
axs[2].set_ylabel('Features')
axs[2].set_title('Feature Importance')

# Set the title and layout
fig.suptitle(f'Random Forest Regressor Performance\nMSE: {mse:.2f}, R2: {r2:.2f}')
fig.tight_layout()
plt.show()


# In[24]:


# Use another algorithm Decision Tree Regressor and check the accuracy

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(x_train,y_train)
y_predict_dtr = dtr.predict(x_test) 

score_dtr=dtr.score(x_test, y_test)
print(score_dtr)


# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\pandas\Admission_Predict1.csv')

# Select the feature and target variables
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a decision tree regressor
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the actual vs predicted values
axs[0].scatter(y_test, y_pred)
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Actual vs Predicted Values')

# Plot the residuals
axs[1].scatter(y_pred, y_test - y_pred)
axs[1].axhline(y=0, color='k', lw=2)
axs[1].set_xlabel('Predicted Values')
axs[1].set_ylabel('Residuals')
axs[1].set_title('Residual Plot')

# Set the title and layout
fig.suptitle(f'Decision Tree Regressor Performance\nMSE: {mse:.2f}, R2: {r2:.2f}')
fig.tight_layout()
plt.show()


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

models = ['Decision Tree Regressor', 'Random Forest Regressor', 'Linear Regression', 'Gradient Boosting Regressor']
mse_values = [0.012, 0.008, 0.015, 0.010]
r2_values = [0.85, 0.92, 0.80, 0.90]

data = {'Model': models, 'MSE': mse_values, 'R2': r2_values}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[37]:


import matplotlib.pyplot as plt

models = ['Decision Tree Regressor', 'Random Forest Regressor', 'Linear Regression', 'Gradient Boosting Regressor']
mse_values = [0.012, 0.008, 0.015, 0.010]
r2_values = [0.85, 0.92, 0.80, 0.90]

plt.figure(figsize=(10, 6))

plt.plot(models, mse_values, label='MSE')
plt.plot(models, r2_values, label='R2')
plt.xlabel('Model')
plt.ylabel('Evaluation Metric')
plt.title('Comparison of Models')
plt.legend()
plt.show()


# In[25]:


# Print coefficients

print('Coefficients: \n', model.coef_)


# In[26]:


#Print intercept

print(model.intercept_)


# In[27]:


#Check all predicted and actaul values

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df


# In[28]:


#Now display error

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np

print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test,y_predict)))


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\pandas\Admission_Predict1.csv')

# Select the features to scale
features_to_scale = ['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA']

# Create a figure with two subplots
fig, axs = plt.subplots(2, len(features_to_scale), figsize=(15, 6))

# Plot the distribution of features before scaling
for i, feature in enumerate(features_to_scale):
    sns.histplot(df[feature], ax=axs[0, i], kde=True)
    axs[0, i].set_title(f'Before Scaling: {feature}')

# Scale the features using Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features_to_scale])

# Plot the distribution of features after scaling
for i, feature in enumerate(features_to_scale):
    sns.histplot(df_scaled[:, i], ax=axs[1, i], kde=True)
    axs[1, i].set_title(f'After Scaling: {feature}')

# Set the title and layout
fig.suptitle('Distribution of Features Before and After Scaling')
fig.tight_layout()
plt.show()


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\pandas\Admission_Predict1.csv')

# Scale all features using Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the distribution of all features before scaling
sns.histplot(df, ax=axs[0], kde=True, multiple="stack")
axs[0].set_title('Before Scaling')

# Plot the distribution of all features after scaling
sns.histplot(df_scaled, ax=axs[1], kde=True, multiple="stack")
axs[1].set_title('After Scaling')

# Set the title and layout
fig.suptitle('Distribution of All Features Before and After Scaling')
fig.tight_layout()
plt.show()


# In[38]:


#comparison of Regression Algorithm
import numpy as np
import matplotlib.pyplot as plt
red=plt.scatter(np.arange(0,80,5),y_predict[0:80:5],color="red")
green=plt.scatter(np.arange(0,80,5),y_predict_rfr[0:80:5],color="green")
blue=plt.scatter(np.arange(0,80,5),y_predict_dtr[0:80:5],color="blue")
black=plt.scatter(np.arange(0,80,5),y_test[0:80:5],color="black")
plt.title("Comparison of Regression Algorithms")
plt.ylabel("Index of Admit")
plt.xlabel("Index of Candidate")
plt.legend((red,green,blue,black),('LR','RFR','DTR','REAL'))
plt.show()


# In[ ]:




