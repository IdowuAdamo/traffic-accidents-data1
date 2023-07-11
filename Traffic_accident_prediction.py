#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind # importing the libraries we will be using in this analysis 


# In[5]:


df = pd.read_csv("Traffic.csv") # reading a csv file using pandas


# In[6]:


df.head() # displaying the first five rows of the dataset


# In[7]:


df.tail() # displaying the last five rows of the dataset 


# In[8]:


df.shape # using the shape attribute to display the number of rows and columns


# In[9]:


list(df.columns) # using the columns attribute to display the column names in the dataset, the list function organises it in a list format


# In[10]:


df.info() # using the info attribute to diplay the total number of non-null values, and the datatype of the columns in the dataset 


# In[11]:


df.describe() # using the describe attribute to display the statistical properties of the columns in the dataset


# # Data cleaning

# In[12]:


df.isnull().sum() # using the isnull attribute to check for the missing values in each column of the dataset  


# In[13]:


df_subset = df.drop('Unnamed: 0',1) # we will be dropping the "Unnamed:0" column since it is a repetition of the index column 
df_subset.head() # We then print the first five rows of the dataset to show that the "Unnamed: 0" column has been dropped


# In[14]:


df_subset.rename(columns={"y": "accident_count", "limit": "speed_limit"}, inplace=True) # Changing the column names to more descriptive names
df_subset.head()


# In[15]:


df_subset["speed_limit"] = df_subset["speed_limit"].astype("category")


# In[17]:


df_subset.info() # the datatype of the speed_limit column has been changed to category


# In[18]:


df_subset.duplicated().sum() # checking for duplicated values in the dataset


# In[34]:


# summary table
pd.crosstab(df_subset.speed_limit, df_subset.accident_count) # cross tabulating between the speed_limit and accident_count columns to give a contigency table


# # Exploratory data analysis

# In[19]:


width = 0.35 # specifying the width of each bar
x = (0,0.8) # specifying the position of the two bars on the x-axis

enforcement_count = df_subset["speed_limit"].value_counts()
plt.bar(x, enforcement_count, width, edgecolor="black")
plt.xticks(x,enforcement_count.index)
plt.title("Speed limit enforcement rate")
plt.xlabel("Speed limit enforcement")
plt.ylabel("Total number of days")
plt.show()


# In[19]:


df_subset["speed_limit"].value_counts()


# In[38]:


enforced = df_subset[df_subset["speed_limit"]=="yes"]
non_enforced = df_subset[df_subset["speed_limit"]=="no"]
plt.boxplot([enforced.accident_count, non_enforced.accident_count], notch =True, widths =(0.5,0.5))
plt.xlabel("Speed limit enforcement")
plt.ylabel("Number of accidents")
plt.xticks([1,2],["enforced","non_enforced"])
plt.show()  # the little circles represent outliers


# In[ ]:





# In[17]:


# Accident frequency

enforced_speed = df_subset[df_subset["speed_limit"]== "yes"]
non_enforced_speed = df_subset[df_subset["speed_limit"]== "no"]

count_enforced = enforced_speed["accident_count"].sum()
count_non_enforced = non_enforced_speed["accident_count"].sum()

print("Accident Frequency:")
print("Enforced speed limit accidents:", count_enforced)
print("Non enforced speed limit accidents:", count_non_enforced)


# In[19]:


# Average accidents

average_accidents_enforced = enforced_speed["accident_count"].mean()
average_accidents_non_enforced = non_enforced_speed["accident_count"].mean()

print("Average Accidents in a day:")
print("Average Accidents on speed limit enforced days:", average_accidents_enforced)
print("Average Accidents on non-enforced speed limit days:", average_accidents_non_enforced)


# In[20]:


# Accident rate comparison
total_days_enforced = enforced_speed.shape[0]
total_days_non_enforced = non_enforced_speed.shape[0]

accident_rate_enforced = count_enforced / total_days_enforced
accident_rate_non_enforced = count_non_enforced/ total_days_non_enforced

print("Accident rate comparison:")
print("Accident rate for enforced speed limit:", accident_rate_enforced)
print("Accident rate for non enforced speed limit:", accident_rate_non_enforced)


# In[27]:


# Time series analysis
accidents_by_day_enforced = enforced_speed.groupby("day").sum()
accidents_by_day_non_enforced = non_enforced_speed.groupby("day").sum()

plt.figure(figsize=(14,6))
plt.plot(accidents_by_day_enforced.index, accidents_by_day_enforced["accident_count"], label= "Enforced speed limit")
plt.plot(accidents_by_day_non_enforced.index, accidents_by_day_non_enforced["accident_count"], label= "Non enforced speed limit")
plt.xlabel("day")
plt.ylabel("accident_count")
plt.title("Accidents by day")
plt.legend()
plt.show()


# In[21]:


# statistical significance
t_statistic, p_value =ttest_ind(enforced_speed["accident_count"], non_enforced_speed["accident_count"])

print("\nStatistical significance:")
print("T-Statistics:", t_statistic)
print("p-Value:", p_value)


# In[ ]:




