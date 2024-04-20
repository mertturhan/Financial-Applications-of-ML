#!/usr/bin/env python
# coding: utf-8

# In[195]:


get_ipython().system('pip install --upgrade gapminder')


# In[196]:


import pandas as pd
import matplotlib.pyplot as plot
import numpy as np


# In[197]:


import gapminder


# In[198]:


college = pd.read_csv("College.csv")
college.index.name='College'
college.describe()


# In[392]:


columns = ['Top10perc','Apps','Enroll']
pd.plotting.scatter_matrix(college[columns], alpha=0.6)


# In[200]:


college.boxplot(column='Outstate', by='Private')


# In[201]:


college['Elite'] = pd.cut(college['Top10perc'], bins = [0,50,100], labels=['Not Elite', 'Elite'], right=False)


# In[202]:


college.value_counts("Elite")


# In[203]:


college.boxplot(column='Outstate', by='Elite')


# In[233]:


fig, axs= plot.subplots(2,2, figsize=(12,6))
axs[0,0].hist(college['Private'], bins=3)
axs[0,0].set_xlabel('University Type')
axs[0,0].set_ylabel('Number of Universities')
axs[0,0].set_title('Is it a private university?')

axs[0,1].hist(college['Top10perc'])
axs[0,1].set_xlabel('% of applicants in Top 25')
axs[0,1].set_ylabel('Number of Universities')

axs[1,0].hist(college['Enroll'])
axs[1,0].set_xlabel('# of enrollment')
axs[1,0].set_ylabel('Number of Universities')

axs[1,1].hist(college['PhD'])
axs[1,1].set_xlabel('# PhD students')
axs[1,1].set_ylabel('Number of Universities')



plot.show()


# In[286]:


auto = pd.read_csv("Auto.csv")
auto.index.name='Auto'
auto.dropna(inplace=True)


# In[287]:


auto['horsepower'] = pd.to_numeric(auto['horsepower'], errors='coerce')
auto.dropna(subset=['horsepower'], inplace=True)


# In[288]:


quantitative_predictors = auto.select_dtypes(include='number')
quantitative_predictors.drop(columns=['origin'], inplace=True)
quantitative_predictors



# In[289]:


for i in quantitative_predictors.columns:
   print(f"Column '{i}': Min = {quantitative_predictors[i].min()}, Max = {quantitative_predictors[i].max()}")




# In[292]:


for i in quantitative_predictors.columns:
    mean = quantitative_predictors[i].mean()
    std_dev = quantitative_predictors[i].std()
    print(f" {i}, Mean: {mean}, Standard Deviation: {std_dev}")


# In[301]:


auto_subset = quantitative_predictors.drop(quantitative_predictors.index[9:85])
subset_stats = auto_subset.describe()
range_values = subset_stats.loc[['min', 'max']]
mean_values = subset_stats.loc['mean']
std_dev_values = subset_stats.loc['std']
print("Range:")
print(range_values)
print("\nMean:")
print(mean_values)
print("\nStandard deviation:")
print(std_dev_values)


# In[393]:


pd.plotting.scatter_matrix(quantitative_predictors, alpha=0.6, figsize=(12, 12))


# In[41]:


gdpcapita = pd.read_csv("gdp_pcap.csv")
lif_x = pd.read_csv("lex.csv")
merged= pd.merge(gdpcapita, lif_x, on="country")
cleaned= merged.dropna()


# In[55]:


gdp_per_capita_2023 = cleaned['2023_x']
life_expectancy_2023 = cleaned['2023_y']


# In[59]:


plot.figure(figsize=(20,6))
plot.scatter(gdp_per_capita_2023, life_expectancy_2023)
plot.title('GDP per Capita vs. Life Expectancy in 2023')
plot.xlabel('GDP per Capita ($)')
plot.xscale('log')
plot.ylabel('Life Expectancy (Years)')
plot.grid(True)
plot.show()


# In[81]:


import seaborn as sns
titanic = sns.load_dataset('titanic')


# In[250]:


female_alive = titanic[(titanic["sex"]=="female") & (titanic["alive"]=="yes")]
male_alive = titanic[(titanic["sex"]=="male") & (titanic["alive"]=="yes")]
female_dead = titanic[(titanic["sex"]=="female") & (titanic["alive"]=="no")]
male_dead = titanic[(titanic["sex"]=="male") & (titanic["alive"]=="no")]


genders = ['Male','Female']
survivor = [len(male_alive), len(female_alive)]
casualty = [len(male_dead), len(female_dead)]


# In[116]:


fig, axs= plot.subplots(1,2, figsize=(12,6))
axs[0].bar(genders, survivor)
axs[0].set_title('Number of Survivors by Gender')
axs[1].bar(genders,casualty)
axs[1].set_title('Number of Casualties by Gender')
max_value = max(max(survivor), max(casualty))
axs[0].set_ylim(0, max_value)
axs[1].set_ylim(0, max_value)
plot.tight_layout()
plot.show()


# In[388]:


covid = pd.read_csv("WHO-COVID-19-global-data.csv")
germany_data = covid[covid["Country"] == 'Germany'].copy()
germany_data["Date_reported"] = pd.to_datetime(germany_data["Date_reported"])
seven_day_avg = []

for i in range(len(germany_data) - 1):
    change = germany_data.iloc[i+1]['Cumulative_cases'] - germany_data.iloc[i]['Cumulative_cases'] 
    seven_day_avg.append(change)

seven_day_avg.append(np.nan)
germany_data['7day_avg'] = seven_day_avg    
    

plot.figure(figsize=(10, 6))
plot.plot(germany_data["Date_reported"], germany_data["Cumulative_cases"], label='Cumulative Cases')
plot.plot(germany_data["Date_reported"], germany_data["7day_avg"], label='7-Day Moving Average of New Cases')
plot.title('Covid Data in Germany')
plot.xlabel('Date')
plot.ylabel('Cases')
plot.legend()
plot.show()

