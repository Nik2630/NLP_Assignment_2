import pandas as pd
import re
import nltk
from nltk import word_tokenize
import math

def vocab() :
    test_set=pd.read_csv('test_set_preprocessed.csv')
    comm=test_set['Comment']
    ## creating a list of words 
    all_words=[]
    for comment in comm :
        all_words+=(str(comment).split())
    ##converting them to lowercase
    all_words=[word.lower() if word.isalpha() else word for word in all_words]
    ## creating a vocab which contains all the unique words
    vocabulary=list(set(all_words))

    train_set=pd.read_csv('train_set_preprocessed.csv')
    comm=train_set['Comment']
    ## creating a list of words 
    all_words=[]
    for comment in comm :
        all_words+=(str(comment).split())
    ##converting them to lowercase
    all_words=[word.lower() if word.isalpha() else word for word in all_words]
    vocabulary.extend(list(set(all_words)))
    vocabulary = list(set(vocabulary))
    return all_words, vocabulary

def create_quadgram_list() :
    quadgram_list=[]
    train_set=pd.read_csv('train_set_preprocessed.csv')
    comm=train_set['Comment']
    #words = []
    for comment in comm :
        words=(str(comment).split())
        if(len(words) < 3): continue
        for i in range(len(words)) :
            if i == 0:
                ## adding start of the sentence
                quadgram_list.append(('<s>','<s>','<s>',words[i]))
            if i == 1:
                quadgram_list.append(('<s>','<s>',words[i - 1],words[i]))
            if i == 2:
                quadgram_list.append(('<s>',words[i - 2],words[i - 1],words[i]))
            if i == len(words) - 3:
                ## adding the end of the sentence
                quadgram_list.append((words[i],words[i+1],words[i+2],'</s>'))
            if i == len(words) - 2:
                quadgram_list.append((words[i],words[i+1],'</s>','</s>'))
            if i == len(words) - 1:
                quadgram_list.append((words[i],'</s>','</s>','</s>'))
            if (i != (len(words) - 1) and i != (len(words) - 2) and i != (len(words) - 3)):
                quadgram_list.append((words[i], words[i + 1],words[i + 2],words[i + 3]))
    size=len(quadgram_list)
    return quadgram_list, size

def count_quadgrams(quadgrams_list) :
    quadgrams={}
    for quadgram in quadgrams_list :
        if quadgram not in quadgrams :
            quadgrams[quadgram] = 1
        else :
            quadgrams[quadgram] += 1
    return quadgrams

### trigram model , creating trigram list, trigram count
def create_trigram_list() :
    trigram_list=[]
    train_set=pd.read_csv('train_set_preprocessed.csv')
    comm=train_set['Comment']
    for comment in comm :
        words=(str(comment).split())
        for i in range(len(words)) :
            if i==0 :
                ## adding start of the sentence
                trigram_list.append(('<s>','<s>',words[i]))
            if i==1 :
                trigram_list.append(('<s>',words[0],words[1]))
            if i==len(words)-2 :
                trigram_list.append((words[i],words[i+1],'</s>'))
            if i==len(words)-1 :
                ## adding the end of the sentence
                trigram_list.append((words[i],'</s>','</s>'))
            if (i!=(len(words)-1) and i!=(len(words)-2)) :
                trigram_list.append((words[i], words[i+1], words[i+2]))
    size=len(trigram_list)
    return trigram_list, size

def count_trigrams(trigrams_list) :
    trigrams={}
    for trigram in trigrams_list :
        if trigram not in trigrams :
            trigrams[trigram] = 1
        else :
            trigrams[trigram] += 1
    return trigrams

def calculate_perplexity(quadgrams_count,trigrams_count) :
    test_set=pd.read_csv('test_set_preprocessed.csv')
    sentences=test_set['Comment']
    total_sentences=len(sentences) # number of <s>
    perplexity=[]
    not_perplexable=[]
    #perplexable_cnt = 0
    #perplexity=0
    #not_perplexable=0
    for sentence in sentences :
        words=(str(sentence).split())
        probability=1
        words_in_sent=len(words)
        if(words_in_sent < 1):
            not_perplexable.append(sentence)
            continue
        for i in range(len(words)-2) :
            chk = 0
            if ((i==0) and (('<s>','<s>','<s>',words[i]) in quadgrams_count)):
                probability *= (quadgrams_count[('<s>','<s>','<s>',words[i])])/total_sentences
                chk = 1
            if ((i==1) and (('<s>','<s>',words[i - 1],words[i]) in quadgrams_count) and (('<s>','<s>',words[i - 1]) in trigrams_count)):
                probability *= (quadgrams_count[('<s>','<s>',words[i-1],words[i])])/trigrams_count[('<s>','<s>',words[i - 1])]
                chk = 1
            if ((i==2) and (('<s>',words[i-2],words[i - 1],words[i]) in quadgrams_count) and (('<s>',words[i-2],words[i - 1]) in trigrams_count)):
                probability *= (quadgrams_count[('<s>',words[i-2],words[i-1],words[i])])/trigrams_count[('<s>',words[i-2],words[i - 1])]
                chk = 1
            if (i==len(words)-3) and ((words[i],words[i+1],words[i+2],'</s>') in quadgrams_count and ((words[i+1],words[i+2],'</s>','</s>') in quadgrams_count) and ((words[i+2],'</s>','</s>','</s>') in quadgrams_count)):
                if ((words[i],words[i+1],words[i+2]) in trigrams_count):
                    probability *= (quadgrams_count[(words[i],words[i+1],words[i+2],'</s>')])/trigrams_count[(words[i],words[i+1],words[i+2])]
                    chk = 1
                if ((words[i+1],words[i+2],'</s>') in trigrams_count):
                    probability *= (quadgrams_count[(words[i+1],words[i+2],'</s>','</s>')])/trigrams_count[(words[i+1],words[i+2],'</s>')]
                    chk = 1
                if ((words[i+2],'</s>','</s>') in trigrams_count):
                    probability *= (quadgrams_count[(words[i+2],'</s>','</s>','</s>')])/trigrams_count[(words[i+2],'</s>','</s>')]
                    chk = 1
            if ((i != len(words)-3) and ((words[i],words[i+1],words[i+2]) in trigrams_count) and ((words[i],words[i+1],words[i+2],words[i+3]) in quadgrams_count)) :
                probability *=(quadgrams_count[(words[i],words[i+1],words[i+2],words[i+3])])/trigrams_count[(words[i],words[i+1],words[i+2])]
                chk = 1
            if chk == 0:
                probability=0
        
        if probability != 0 :
            perplex=(1/(probability))**(1/(words_in_sent))
            #perplexity.append(perplex)
            if perplex >= 1000000:
                not_perplexable.append(sentence)
                #not_perplexable += 1
            else :
                perplexity.append(perplex)
                #perplexity += perplex
                #perplexable_cnt += 1
        else :
            not_perplexable.append(sentence)
            #not_perplexable += 1
    average_perplexity=sum(perplexity)/len(perplexity)
    #average_perplexity = perplexity/perplexable_cnt
    return average_perplexity, not_perplexable, perplexity

all_words, vocabulary=vocab()
quadgrams_list, quadgrams_size  = create_quadgram_list()
quadgrams = count_quadgrams(quadgrams_list)
trigrams_list, trigram_size= create_trigram_list()
trigrams = count_trigrams(trigrams_list)
average_perplexity, not_perplexable, perplexity = calculate_perplexity(quadgrams,trigrams)
print("the average perplexity over all the sentences that are perplexable is ", average_perplexity)
print("the total sentences in test set are ", len(perplexity) + len(not_perplexable))
print("the total no of not_perplexable sentences in the validation set are :", len(not_perplexable))

##OUTPUT
#the average perplexity over all the sentences that are perplexable is  5.475069992222899
#the total sentences in test set are  67870
#the total no of not_perplexable sentences in the validation set are : 44635
