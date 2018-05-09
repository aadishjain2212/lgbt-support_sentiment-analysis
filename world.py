
# coding: utf-8

# # Gathering Tweets from world
# For getting twitter feed I used **tweepy** python module.

# #importing necessary libraries

# In[1]:

import pandas


# In[51]:

import json
import csv


# In[9]:

from tweepy.streaming import StreamListener  #using 
from tweepy import OAuthHandler
from tweepy import Stream


# In[10]:

import json


# In[127]:

class StdOutListener(StreamListener):
    def __init__(self):
        self.count=0
        self.data=[]
        self.data.append(["created_at","text"])
    def on_data(self,data):
        if self.count<=4000:
            print(self.count)
            self.count+=1
            json_data=json.loads(data)
            self.data.append([json_data['created_at'],json_data['text']])
        else:
            return
    def on_error(self,status):
        print("ERROR")
        print(status)


# In[128]:

consumer_key="0ZfraBKgXcIFVeuEEtYsrdX5i"
consumer_secret="CNNRk4Pg0ShsfocYrSDNd35xiDyAREB107P6q32a0y9ezN3KPu"
access_token="3677025552-OhjxtEl74XNgoFx3axJUOA4mwR12oQEdhZhnbiw"
access_secret="FjQtZy6dKeUW1XsiKk7VKG0iGkqNtmOh4MMMUj8XxeH0w"


# In[129]:

auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)


# In[130]:

obj=StdOutListener()


# In[131]:

twitterStream=Stream(auth,obj)


# ### I have used the following tags to get the data from twitter feed 

# In[132]:

tracks=["gayrights","LGBT","lgbtq","pride","is thelgbt","loveislove","gaylife",'transrights',
                           'gayhunk','equality','support','section377','mortauxgay']


# In[133]:

twitterStream.filter(track=tracks)


# In[151]:

data=obj.data


# ### After retriving the data , it is stored in a csv file with date and text

# In[153]:

csvfile = open('world.csv', 'w') 
csvwriter = csv.writer(csvfile)
for item in data:
    csvwriter.writerow(item)
csvfile.close()

