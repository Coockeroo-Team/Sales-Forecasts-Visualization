#!/usr/bin/env python
# coding: utf-8

# ## Coockeroo 

# ## Sanjoy Biswas

# ## Time Series / Forecasts: From the basic solution to the complex – daily and monthly- by store and by product
# 
# 

# 
# * This notebook is one of several notebooks for a project to improve store forecasts
# 1.	EDA – Exploratory Data Analysis – includes working with annual forecasts
# 2.	Main Modelling
# 3.	XG Boost modelling by Month
# 4.	Weighted average
# 5.	ARIMA – Month and Other Modelling
# 6.	Deep Learning


# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns



df = pd.read_csv(r"E:\Coockeroo\Sales Data Visualization\Data\train.csv")


#df = pd.read_csv(r'C:/Users/alexd/Alex Folder 1/Capstone - Store Forecast/train.csv')  
df.head()
df.shape


# # EDA

df.shape
df.nunique()
df['sales'].sum()


# # Other Data Files - That were not used

df.max()


# # Manipulation

df['date'] =  pd.to_datetime(df['date'])

df = df.set_index('date')

df.head()

df_sales_only = df.drop(['store','item'], axis = 1)


df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek
print(df)


sns.boxplot(x="dayofweek", y="sales", data=df)


# # Working With individual products and individual stores
df.groupby(['store','item']).size()
df[(df.store==1) & (df.item==1)]
df[(df.store==1) & (df.item==1)]['sales'].plot()
df[(df.store==1) & (df.item==1)]['sales']
df_1_1 = df[(df.store==1) & (df.item==1)]['sales']
split = "2017-01-01"


df_i1_s1 = df[(df.store==1) & (df.item==1)]['sales']
df_i1_s2 = df[(df.store==2) & (df.item==1)]['sales']
df_i2_s1 = df[(df.store==1) & (df.item==2)]['sales']
df_i2_s2 = df[(df.store==2) & (df.item==2)]['sales']

df['sales'].resample('W').sum().plot()
df['sales'].resample('M').sum().plot()


df_i1_s1.to_frame()
df_i1_s2.to_frame()
df_i2_s1.to_frame()
df_i2_s1.to_frame()

split1 = "2013-12-31"

df_i1_s1_a = df_i1_s1[:split1] 
df_i1_s2_a = df_i1_s2[:split1] 
df_i2_s1_a = df_i2_s1[:split1] 
df_i2_s2_a = df_i2_s2[:split1] 


df_i1_s1_a.resample('W').sum().plot()
df_i1_s2_a.head(10)
df_i1_s2_a.resample('D').sum().plot()
df_i2_s1_a.head()
df_i2_s2_a.resample('W').sum().plot()
df_i2_s1_a.resample('W').sum().plot()
df_1_1.to_frame()


df_1_1 = df_1_1.to_frame()
df_xg_1_1= df_1_1.copy()  # it was here
df_1_1.head()


# # Exploring Shifting
df_1_1['sales-1'] = df_1_1['sales'].shift(1)
df_1_1['sales+2'] = df_1_1['sales'].shift(-2)
df_1_1.head()
df_1_1['sales-2'] = df_1_1['sales'].shift(2)
df_1_1.head()
df_1_1 = df_1_1.dropna()
df_1_1.head()
df_1_1.tail()
df_1_1['sales+1'] = df_1_1['sales'].shift(-1)
df_1_1.head()
df_1_1.tail()


# # Looking at the Data
agg_month_item = pd.pivot_table(df, index='month', columns='item', values='sales', aggfunc=np.sum).values

print(agg_month_item)
df["2017-01-03" : "2017-01-20"].sales.sum()
df.dtypes
df["2017-01"]


# # Plotting 1
df.sales.resample('M').sum().plot()
df["2017-01-01" : "2017-12-31"].sales.resample('W').sum().plot()
df["2016-01-01" : "2016-12-31"].sales.resample('W').sum().plot()
df["2017-01-03" : "2017-12-31"].sales.resample('W').sum().plot(kind="hist")
df.sales.resample('B').sum().plot()
df["2017-01-01" : "2017-01-31"].sales.resample('d').sum().plot()  


# # ACF FROM HERE

from statsmodels.graphics.tsaplots import plot_acf
Jan2017 = df["2017-01-01" : "2017-01-31"].sales.resample('d').sum()  
plot_acf(Jan2017)
plt.plot(Jan2017)
T2017 = df["2017-01-01" : "2017-12-31"].sales.resample('m').sum() 


#plot_acf(T2017)
df_1_1_Jan_17 = df_1_1["2017-01-01" : "2017-01-31"]
df_1_1_Jan_17 = df_1_1_Jan_17['sales']
Byyear = df["2013-01-01" : "2017-12-31"].sales.resample('y').sum() 
plot_acf(Byyear)


# # Graph 2

sns.pointplot(x=df['year'], y=df['sales'])


# # More Working With Data


df['Year'] = df.index.year
df['Month'] = df.index.month
#df['Weekday Name'] = df.weekday_name
df.sample(5, random_state=0)

ax = df.loc['2017', 'sales'].plot()
sns.boxplot(data=df, x='Year', y='sales');
sns.boxplot(data=df, x='Month', y='sales');
sns.boxplot(data=df, x='store', y='sales');
sns.boxplot(data=df, x='item', y='sales');


# # Working with data

daybyweek = df.groupby(['dayofweek']).agg({'sales':'sum'})
print(daybyweek)
day = df.groupby(['day']).agg({'sales':'sum'})
print(day)
day = df.groupby(['item','store' ]).agg({'sales':'sum'})
print(day)
SalesByDay = df.groupby(['date']).agg({'sales':'sum'})
print (SalesByDay)
SalesByDay.plot()


salesstore = df.groupby(['date','store']).agg({'sales' : 'sum'})

sales_by_year = pd.pivot_table(df, index='year', values='sales', aggfunc=np.sum)
print(sales_by_year)


# # By ITEM
df[df.item == 1]['sales'].plot()
df[df.item == 2]['sales'].plot()
df[df.item == 20]['sales'].plot()
storetotal = df.groupby(['store']).agg({'sales':'sum'})
print(sum(storetotal['sales']))
print(storetotal)
itemtotal = df.groupby(['item']).agg({'sales':'sum'})
print(sum(itemtotal['sales']))
store1 = df[df.store == 1]['sales']
store1.plot()



store5 = df[df.store == 5]['sales']
store5.plot(color='green')
df['sales'].plot(linewidth=0.5);


# # Working with Totals and Averages
grand_avg = df.sales.mean()
store_item_table = pd.pivot_table(df, index='store', columns='item', values='sales', aggfunc=np.mean)
display(store_item_table)
month_table = pd.pivot_table(df, index='month', values='sales', aggfunc=np.mean)
month_table.sales /= grand_avg
print(month_table)



dow_table = pd.pivot_table(df, index='dayofweek', values='sales', aggfunc=np.mean)
dow_table.sales /= grand_avg
print(dow_table)
year_table = pd.pivot_table(df, index='year', values='sales', aggfunc=np.mean)
year_table /= grand_avg


print(year_table)
year_table.info()


# # Working with year
years = np.arange(2013, 2019)
annual_sales_avg = year_table.values.squeeze()
print(annual_sales_avg)
print(year_table)


p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
p3 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 3))



plt.figure(figsize=(8,6))
plt.plot(years[:-1], annual_sales_avg, 'ko')
plt.plot(years, p1(years), 'C0-', color ='red')
plt.plot(years, p2(years), 'C1-',color ='blue')
plt.plot(years, p3(years), 'C2-',color ='green')
plt.xlim(2012.5, 2018.5)
plt.title("Relative Sales by Year")
plt.ylabel("Relative Sales")
plt.xlabel("Year")
plt.show()



print(f"2017 Relative Sales by Degree-1 (Linear) Fit = {p1(2017):.4f}")
print(f"2017 Relative Sales by Degree-2 (Quadratic) Fit = {p2(2017):.4f}")
print(f"2017 Relative Sales by Degree-3 (3 degrees) Fit = {p3(2017):.4f}")

