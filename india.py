
# coding: utf-8

# # Gathering tweets from India
# for gathering twitter feed i used **tweepy** python module

# ### importing necessary libraries

# In[1]:

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


# In[2]:

import json
import pandas
import csv


# In[3]:

consumer_key="0ZfraBKgXcIFVeuEEtYsrdX5i"
consumer_secret="CNNRk4Pg0ShsfocYrSDNd35xiDyAREB107P6q32a0y9ezN3KPu"
access_token="3677025552-OhjxtEl74XNgoFx3axJUOA4mwR12oQEdhZhnbiw"
access_secret="FjQtZy6dKeUW1XsiKk7VKG0iGkqNtmOh4MMMUj8XxeH0w"


# In[56]:

import tweepy


# In[57]:

auth=OAuthHandler(consumer_key,consumer_secret)


# In[76]:

api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)


# ## following tags were used to get tweets

# In[79]:

searchQuery = '#gayrights OR #LGBT OR #lgbtq OR'               '#isthelgbt OR #loeislove OR #gaylife OR'               '#transrights OR #gayhunkor OR #section377 OR'               '"mortauxgay" OR "lgbt" OR "supportlgbt"'


# In[84]:

t_count=10000
t=0
list=[]
for tweets in tweepy.Cursor(api.search,q=searchQuery,lang="en",geocode="29.0218535,79.4017232,4000km").items(t_count):
    t+=1
    print(t)
    list.append(tweets)
    


# columns="created_At,text\n"

# In[101]:

columns=["created_at","text"]


# In[106]:

data=[]


# In[107]:

data.append(columns)


# In[108]:

for i in list:
    date_data=i.created_at
    text_data=i.text
    data.append([date_data,text_data])


# ## Storing data in a csv file

# In[114]:

csvfile = open('india.csv', 'w')
csvwriter = csv.writer(csvfile)
for item in data:
    csvwriter.writerow(item)
csvfile.close()

