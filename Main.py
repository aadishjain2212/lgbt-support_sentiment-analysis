
# coding: utf-8

# In[1]:

import nltk
from nltk import *
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
import csv
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier # A wrapper to include the scikit learn algorithms within the nltk classifier
import matplotlib.pyplot as plt
import random


# In[34]:

from sklearn.metrics import confusion_matrix 
import numpy as np
import itertools


# **Tokenize the data**

# In[2]:

def remove_punctuation(word):
    no_punct = ""
    for char in word:
        if char not in string.punctuation:
            no_punct = no_punct + char
    if no_punct.isnumeric():
        return ''
    return no_punct


# In[3]:

''.isnumeric()


# In[4]:

remove_punctuation(".")


# In[5]:

ps = PorterStemmer()
def stem_the_word(word):
    word.encode('utf-8')
    return ps.stem(word)


# In[6]:

stem_the_word("running")


# In[7]:

def tokenize_the_training_data(data):
    tokenized_training_data = []
    for each in data:
        sentiment = each.strip().split(',')[0]
        text = str(each.strip().split(',')[1:])
        tokenized_training_data.append((word_tokenize(text), sentiment))
    return tokenized_training_data


# In[8]:

stop_words = stopwords.words('english')
def filter_words(tokenized_data):
    filtered_data = []
    for (tokenized_words, sentiment) in tokenized_data:
        filtered_data.append(([stem_the_word(remove_punctuation(word.lower())) for word in tokenized_words if remove_punctuation(word.lower()) not in stop_words and remove_punctuation(word.lower()) != ''], sentiment))
    return filtered_data


# In[ ]:




# In[9]:

# reading data from reviews.txt
with open("Reviews.txt") as file:
    data = file.readlines()
#print(data)

tokenized_training_data = tokenize_the_training_data(data)
#print(tokenized_training_data)
filtered_training_data = filter_words(tokenized_training_data)
print(filtered_training_data)


# In[10]:

def find_words_in_data(filtered_training_data):
    all_words = []
    for (words, sentiment) in filtered_training_data:
        if words != "":
          all_words += words
    return all_words


# In[11]:

len(find_words_in_data(filtered_training_data))


# In[12]:

def find_word_count(words):
    words = FreqDist(words)
    print(words)
    word_features = [w for (w, c) in words.most_common(1000)]
    return word_features


# In[13]:

word_features = find_word_count(find_words_in_data(filtered_training_data))


# In[14]:

#word_features_set = set(word_features)
word_features_set = find_words_in_data(filtered_training_data)


# In[15]:

len(word_features_set)


# In[16]:

def find_features(each_review_words):
    global all_words_in_data
    features = {}
    words = set(each_review_words)
    for each_word in word_features_set:
        #features[each_word] = (each_word in words)
        if each_word in words:
            features[each_word] = 1
        else:
            features[each_word] = 0
    return features


# In[17]:

final_filtered_data = [(find_features(words), sentiment) for (words,sentiment) in filtered_training_data]


# In[18]:

final_filtered_data[0]


# In[19]:

len(final_filtered_data)


# In[20]:

print(word_features_set)


# In[21]:

'''def storing_dataset(training_set):
    writer=csv.writer(open("final_dataset.csv",'w'))
    header = list(word_features_set)
    header.append("CLASS_SENTIMENT")   #  1 - for positive   and  0 - for negative
    writer.writerow(header)
    for each_row in training_set:
        listt = []
        dict = each_row[0]
        lable = each_row[1]
        for key in dict:
            listt.append(dict[key])
        listt.append(lable)
        writer.writerow(listt)
        '''


# In[22]:

#storing_dataset(final_filtered_data)


# In[23]:

def tokenize_the_test_data(data):
    tokenized_test_data = []
    for each in data:
        sentiment = 'null'
        text = str(each.strip().split(',')[0])
        #print(text)
        tokenized_test_data.append((word_tokenize(text), sentiment))
    #print(tokenized_test_data)
    return tokenized_test_data
def predict(data, model):
    result = []
    tokenized_test_data = tokenize_the_test_data(data)
    #print(tokenized_training_data)
    filtered_test_data = filter_words(tokenized_test_data)
    #print(filtered_test_data)
    filtered_test_data = [(find_features(words), sentiment) for (words,sentiment) in filtered_test_data]
    #print(filtered_test_data)
    #print(len(filtered_test_data))
    for each in filtered_test_data:
        #print(each[0])
        result.append(model.classify(each[0]))
    return result
    
    
    


# ### Splitting the final filtered data into training and test Split

# In[25]:

random.shuffle(final_filtered_data)
train_set = final_filtered_data[:3500]
test_set = final_filtered_data[3501:]


# # Fitting a LinearSVC Classifier

# In[26]:

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)


# In[27]:

print("LinearSVC_Classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)


# In[28]:

y_pred = []
y_actual = []
for each_set in test_set:
    y_actual.append(each_set[1])
    y_pred.append(LinearSVC_classifier.classify(each_set[0]))
print(y_pred)
print(y_actual)


# In[105]:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_actual, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(12, 8))
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix, without normalization')
plt.savefig('./Output/Confusion_matrix.png')
plt.show()


# ## Predicting Sentiment for Tweets on LGBT

# In[37]:

india_df = pd.read_csv('./TwitterTweets/test.csv')
india_df.head()


# In[38]:

rs = predict(list(india_df.text) , LinearSVC_classifier)
rs


# In[39]:

india_df['Sentiment'] = rs


# In[40]:

n_tweets = len(india_df[india_df.Sentiment == '0'])


# In[41]:

p_tweets = len(india_df[india_df.Sentiment == '1'])


# In[103]:

plt.figure(figsize=(12, 8))
plt.bar(['negative', 'positive'], [n_tweets, p_tweets] , width=0.2, align = 'center' , ec=['k','k'] , linewidth = 2)
plt.title('Count of Positive and Negative Tweets  (location : India)')
plt.ylabel('Frequency')
plt.xlabel('Sentiment')
plt.savefig('./Output/India_LGBT_sentiment.png')
plt.show()


# ### Similarly For the world Data

# In[44]:

world_df = pd.read_csv('./TwitterTweets/world.csv')
world_df.head()


# In[46]:

w_rs = predict(list(world_df.text) , LinearSVC_classifier)
w_rs


# In[47]:

world_df['Sentiment'] = w_rs


# In[56]:

world_df.head()


# In[57]:

neg_tweets = len(world_df[world_df.Sentiment == '0'])


# In[74]:

pos_tweets = len(world_df[world_df.Sentiment == '1'])


# In[102]:

plt.figure(figsize=(12, 8))
plt.bar(['negative', 'positive'], [neg_tweets, pos_tweets] , width=0.2, align = 'center' , ec=['k','k'] , linewidth = 2)
plt.title('Count of Positive and Negative Tweets (World)')
plt.ylabel('Frequency')
plt.xlabel('Sentiment')
plt.savefig('./Output/World_LGBT_sentiment.png')
plt.show()


# In[104]:

ind = np.arange(2)  # the x locations for the groups
width = 0.35       # the width of the bars
india  = (n_tweets, p_tweets)
plt.figure(figsize = (12, 8))
fig, ax = plt.subplots(figsize = (12, 8))
rects1 = ax.bar(ind, india, width, color='#FF5252')

world = (neg_tweets, pos_tweets)
rects2 = ax.bar(ind + width, world, width, color='#303F9F', linewidth=6)
ax.set_ylabel('Frequency')
ax.set_title('Comaprative Sentiment Graph of Support For LGBT Community')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Negative', 'Positive'))
ax.set_xlabel('Sentiment')

ax.legend((rects1[0], rects2[0]), ('India', 'World'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.savefig('./Output/Comaparative_LGBT_sentiment.png')
plt.show()


# In[94]:




# In[ ]:



