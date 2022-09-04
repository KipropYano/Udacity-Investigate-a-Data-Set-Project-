#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset - No-show appointments 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# The dataset we are working is No-show appointments data set (. This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. 
# 
# I will analyze and communicate my findings using Python libraries (NumPy, Pandas, Matplotlib, and seaborn).
# 
# 
# #### Column names in each table
#     Name            Significance
# 
#     PatientId:       Identification of patient
#     AppointmentID:   Identification of each appointment
#     Gender:          Male (M) or Female (F)
#     ScheduledDay:    Time expected for appointment
#     AppointmentDay:  Actual day of appointment 
#     Age:             How old the patient is
#     Neighbourhood:   Where the appointment takes place
#     Scholarship:     True (1) of False (0) 
#     Hipertension:    Medical condition 
#     Diabetes:        Medical condition
#     Alcoholism:      Medical condition
#     Handcap:         Medical condition
#     SMS_received:    Messages sent to the patient
#     No-show:         Yes or No
# 
# 
# ### Question(s) for Analysis
# 1. Research Question 1: What is the rate of those who did not show up vs. those who showed up?
# 2. Research Question 2: What factors are important for us to know in order to predict if a patient will show up for their scheduled appointment?
# 

# In[4]:


# Importing statements for all of the packages to be used

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 

# In[200]:


#Loading the dataset 
df = pd.read_csv('noshowappointments.csv')

df.head()


# In[5]:


#Checkign for data types
df.info()


# From the above, it is evident that some columns have incorrect data types. The "ScheduledDay" and "AppointmentDay" should contaub string objects not object types. 
# 
# This will be converted to datetime types in the cleanup section. 

# In[7]:


#Checking for dataset shape
df.shape


# Reference
# https://pythonexamples.org/pandas-dataframe-shape/
# 
# From the results, the dataset has 110527 records and 14 columns. 

# In[14]:


#Checking for missing values
df.isnull().sum()


# There are no missing values in this dataset 
# 
# There are some typos in the dataset 'Hipertension' will be changed to 'Hypertension' and 'Handcap' will be changed to 'Handicap.'

# In[8]:


#Descriptive summary of the dataset 
df.describe()


# From the output, it is the minimum age is -1. Therefore, we will drop it because there is no negative age. 

# In[11]:


#dropping the -1 value 
df.drop(df.query("Age == -1").index,inplace=True)


# Reference
# https://www.w3schools.com/python/pandas/ref_df_drop.asp

# In[210]:


#Confirming if the minimum age is okay now 
df.describe()


# As seen above, the minimum age has been set to 0. 

# In[212]:


df.duplicated().sum()


# It is evident that there are no duplicate values in the dataset 

# In[211]:


#Returning the number of unique values for each column
print(df.nunique())


# Reference
# https://www.w3schools.com/python/pandas/ref_df_nunique.asp#:~:text=The%20nunique()%20method%20returns,unique%20values%20for%20each%20row.

# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[16]:


df.info()


# In[216]:


#Correcting typos 
df.rename(columns = {'Hipertension': 'Hypertension',
                'Handcap': 'Handicap','No-show':'No_show'}, inplace = True)


# In[217]:


df.info()


# In[218]:


#Fixing datatypes errors
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df.head()


# In[219]:


#Converting no show data to 0 and 1
df.No_show[df['No_show'] == 'Yes'] = '1'
df.No_show[df['No_show'] == 'No'] = '0'
df['No-show'] = pd.to_numeric(df['No_show'])


# Converting no to 0 and yes to 1 will help when using groupby or plot in the subsequent sections.  

# In[107]:


#Dropping columns that are not needed in the analysis

df.drop(['AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)
df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### Research Question 1: What is the rate of those who did not show up vs. those who showed up? 
# 
# This will be represented using a pie chart showing the percentage of those who showed up and not. 

# In[221]:


allP = df['No_show'].value_counts()
print(allP[1] / allP.sum() * 100)
pieChart = allP.plot.pie(figsize=(10,10), autopct='%1.1f%%', fontsize = 12);
pieChart.set_title(' (%) (Per appointment)\n', fontsize = 15);
plt.legend();


# This indicates that the percentage of those who did not show up is low as compared to those who showed up. 

# ### Research Question 2: What factors are important for us to know in order to predict if a patient will show up for their scheduled appointment?
# 
# #### The are different aspects towards understanding the different factors that can predict if a patient will show up for their schedule appointment. Therefore, we need to bring the correlation between: 
# 1. Appointment and age
# 2. Appointment and gender 
# 3. Appointment adn neighborhood 
# 4. Appointment and scholarship 
# 5. Appointment and hypertension 
# 6. Appointment and diabetes 
# 7. Appointment and alcoholism 
# 8. Appointment and handicap
# 8. Appointment and sms received

# #### Correlation between age and showing up for scheduled appointment 

# In[67]:


miss = df["No_show"] == "Yes"
show = df["No_show"] == "No"


# In[223]:


#value counts for no_show 
Age_count = df.groupby("Age")["No_show"].value_counts()
Age_count


# In[225]:


#Value counts for age 
Age_count.groupby(level=0).sum()


# In[227]:


#Bar plot for patients of all ages who missed their appointments 

round(Age_count / Age_count.groupby(level=0).sum() * 100,2).unstack().plot(kind="bar",stacked=True, figsize=(25,5));
plt.legend(["show","miss"]);
plt.title("Percentage of missed appointments across all ages");


# In[82]:


#percentage mean value for patients missing and showed appointments as per thier ages
print("The mean value of patient missing the appointment for the age:")
round(Age_count / Age_count.groupby(level=0).sum() * 100,2).unstack().mean()


# In[71]:


#Bucketing ages to have a clear look of each age group 

bins=[0,10,20,30,40,50,60,70,80,90,100,120]
df["Age_bin10"] = pd.cut(df.Age, bins=bins)
Age10_count = df.groupby("Age_bin10")["No_show"].value_counts()


# In[72]:


Age10_count


# Reference
# https://python-course.eu/numerical-programming/binning-in-python-and-pandas.php

# In[75]:


(Age10_count/Age10_count.groupby(level=0).sum()*100).unstack().plot(kind="bar",stacked=True)
plt.legend(["Show","Miss"]);
plt.title("Percentage of Missing appointment across ages")


# In[76]:


(Age10_count/Age10_count.groupby(level=0).sum()*100)


# In[81]:


#ratio to all patients 
(Age10_count/len(df)*100).unstack().plot(kind="bar",stacked=True);
plt.title("Percentage of Missing appointment across ages to total number of patients");


# In[103]:


# percentages of no show patients based on age group
No_show = df.groupby('Age').No_show.mean() * 100


# In[105]:


# plot a bar chart
plt.figure(figsize = [8, 4])
plt.bar(x = No_show.index, height = No_show)
plt.title('The Percentages of No Show Patients Based on Age Group')
plt.xlabel('Age')
plt.ylabel('No_show');


# Reference
# https://towardsdatascience.com/data-preprocessing-with-python-pandas-part-5-binning-c5bd5fd1b950
# https://www.codegrepper.com/code-examples/python/bin+age+groups+in+python+
# 

# From the above analysis, there is a link between age and patients missing appointments. 
# 
# The mean value of patient missing the appointment for the age is 19.8 and those who showed up is 80.2 indicating that most of the patients across all ages showed up for their appointments. 
# 
# Besides, it is evident that patients aged between 15 to 35 years missed most of their appointments. 

# #### Correlation between gender and showing up for scheduled appointment

# In[90]:


##value counts for no_show 
Gender_count = df.groupby("Gender")["No_show"].value_counts()
print(Gender_count)


# In[136]:


#Bar plot for patients of all genders who missed their appointments 

print("% of women missed their appointment: ",round(Gender_count[1]/(Gender_count[0]+Gender_count[1])*100,2))
print("% of males who missed their appointment  : ",round(Gender_count[3]/(Gender_count[3]+Gender_count[2])*100,2))
df.groupby("Gender")["No_show"].value_counts().unstack().plot(kind="bar",stacked=True)
plt.legend(["show","miss"]);
plt.title("% of male and females")


# The % of women who missed their appointment 20.31 whereas that of males who missed their appointment is 19.97.
# 
# This cannot be a factor for missing appointments since the percentages are approximately similar (20% when rounded off). 

# #### Correlation between neighborhood and showing up for scheduled appointment
# 

# In[109]:


##value counts for no_show 
Neighbourhood_count = df.groupby("Neighbourhood")["No_show"].value_counts()
print(Neighbourhood_count)


# In[231]:


#Bar plot for patients of all neighbourhood who missed their appointments 

round(Neighbourhood_count / len(df) * 100,2).unstack().plot(kind="bar",stacked=True, figsize=(25,5));
plt.legend(["show","miss"]);
plt.title("percentage of patients missing their appointment in each neighbourhood");


# In[138]:


print("The mean value for the percentage of patients missing appointments in all neighbourhood:",
      round((Neighbourhood_count / Neighbourhood_count.groupby(level=0).sum() * 100).unstack().mean(),2))


# JARDIM CAMBURI neighbourhood had the heighest number of missed appointments. 
# 
# The mean value for the percentage of patients missing appointments in all neighbourhood is 80 while those showing up is 20. This indicates that patients from those from the neighbourhood have many missed appointments. 

# #### Correlation between health conditions (alcoholism, diabetes, hypertension and handicap) and showing up for scheduled appointment

# In[148]:


df['number_of_cond'] = df.Hypertension + df.Diabetes + df.Alcoholism + df.Handicap


# In[149]:


# empty dictionary 
cond_dict = {}

# percentage of no show patients who suffered from hypertension only
cond_dict['Hypertension'] = (df[df.number_of_cond <= 1].groupby('Hypertension').No_show.mean() * 100)[1]
# percentage of no show patients who suffered from diabetes only
cond_dict['Diabetes'] = (df[df.number_of_cond <= 1].groupby('Diabetes').No_show.mean() * 100)[1]
# percentage of no show patients who suffered from alcoholism only
cond_dict['Alcoholism'] = (df[df.number_of_cond <= 1].groupby('Alcoholism').No_show.mean() * 100)[1]
# percentage of no show patients who were handicapped only
cond_dict['Handicapped'] = (df[df.number_of_cond <= 1].groupby('Handicap').No_show.mean() * 100)[1]

# percentage of no show patients who suffered from multiple conditions
cond_dict['Multiple Conditions'] = df[df.number_of_cond > 1].No_show.mean() * 100

cond_dict


# In[153]:


# converting the dictionary to a pandas series
cond_ser = pd.Series(cond_dict)
cond_ser


# References
# https://www.machinelearningplus.com/pandas/creating-pandas-series-from-dictionary/#:~:text=To%20make%20a%20series%20from,the%20values%20of%20the%20series.
# 
# https://www.geeksforgeeks.org/how-to-convert-a-dictionary-to-a-pandas-series/
# 
# https://www.w3resource.com/python-exercises/pandas/python-pandas-data-series-exercise-5.php
# 

# In[155]:


# ploting a bar chart
plt.figure(figsize = [10, 4])
plt.bar(x = cond_ser.index, height = cond_ser)
plt.title('Percentages of Missed Appointments For Each Health Condition')
plt.ylabel('No Show Percentage');


# Alcoholic patients have the highest percentage of missed appointments 
# 
# The health conditions (alcoholism, diabetes, hypertension and handicap) is a factor to missed appointments. 

# #### Correlation between scholarship and showing up for scheduled appointment

# In[156]:


##value counts for no_show 
Scholarship_count = df.groupby("Scholarship")["No_show"].value_counts()
print(Scholarship_count)


# In[168]:


print("The mean value for Patients with scholarship who missed their appointment:",
      round((Scholarship_count / Scholarship_count.groupby(level=0).sum() * 100).unstack().mean(),2))


# In[167]:


print("Patients with scholarship who missed their appointment: ",round(Scholarship_count[1]/(Scholarship_count[0]+Scholarship_count[1])*100,2))
df.groupby("Scholarship")["No_show"].value_counts().unstack().plot(kind="bar",stacked=True)
plt.legend(["show","miss"]);
plt.title("No. of patients showed based on scholarship")


# From the above output, a high number of patients with scholarship missed their appointments as opposed to those without scholarship.
# 
# This indicates that a scholarship is an indicator of appointments. 

# #### Correlation between sms received and missed their appointment

# In[173]:


print(df.groupby('SMS_received').mean())


# In[197]:


SMS_count = df.groupby("SMS_received")["No_show"].value_counts()
SMS_count


# In[193]:


SMS_YN = (SMS_count / SMS_count.groupby(level=0).sum() * 100).unstack()
SMS_YN.index = ["NOt Received", "Received"]
SMS_YN


# In[233]:


SMS_YN.plot(kind="bar",stacked=True)
plt.legend(["showed","missed"])
plt.title("Percentage of patients who missed or showed and Receiving SMS");


# The above output indicates that majority of the patients did not receive SMS and they had a lower percentage to missing appointments as opposed to those who received.
# 
# Therefore, it shows that receiving SMS has no effect on patients showing up for appointments. 

# <a id='conclusions'></a>
# ## Conclusions
# 
# I investigated the factors that are important for us to know in order to predict if a patient will show up for their scheduled appointment and these are the results of these analyses: 
# 
# - The percentage of those who did not show up is low as compared to those who showed up. 20% never showed up whereas 80% of the patients showed up for the appointment. 
# 
# - The factors that have a include age, neighborhood, health conditions (alcoholism, diabetes, hypertension and handicap) and scholarship. 
# 
# - The mean value of patient missing the appointment for the age is 19.8 and those who showed up is 80.2 indicating that most of the patients across all ages showed up for their appointments. Also, it was evident that patients aged between 15 to 35 years missed most of their appointments.
# 
# - The mean value for the percentage of patients missing appointments in all neighbourhood is 80 while those showing up is 20. This indicates that patients from those from the neighbourhood have many missed appointments. Also, JARDIM CAMBURI neighbourhood had the heighest number of missed appointments. 
# 
# - The health conditions (alcoholism, diabetes, hypertension and handicap) is a factor to missed appointments with alcoholic patients have the highest percentage of missed appointments. 
# 
# - A high number of patients with scholarship missed their appointments as opposed to those without scholarship.
#  
# #### Limitation 
# The SMS_received column was not clear on the dataset page since it was difficult to derive if it was a factor for missed appointments. This shows why receiving SMS has no effect on patients showing up for appointments from our analyses. 
# 
# 
# 
# 
# 

# In[232]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

