import pandas as pd
import re
import nltk
from nltk import word_tokenize
import math

def vocab() :
    train_set=pd.read_csv("C:\\Users\\Darshi Doshi\\Desktop\\complete_dataset.csv")
    comm=train_set['Comment']
    # print(comm)
    ## creating a list of words
    all_words=[]
    for comment in comm :
        if isinstance(comment, str):
            all_words+=comment.split()

    ## creating a vocab which contains all the unique words
    vocabulary=list(set(all_words))
    return all_words, vocabulary
# all_words, vocabulary = vocab()

def create_quadgram_list() :
    quadgram_list=[]
    train_set=pd.read_csv("C:\\Users\\Darshi Doshi\\Desktop\\train_set_preprocessed11.csv")
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
    train_set=pd.read_csv("C:\\Users\\Darshi Doshi\\Desktop\\train_set_preprocessed11.csv")
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

all_words, vocabulary=vocab()
quadgrams_list, quadgrams_size  = create_quadgram_list()
quadgrams = count_quadgrams(quadgrams_list)
trigrams_list, trigram_size= create_trigram_list()
trigrams = count_trigrams(trigrams_list)

#WITH SMOOTHING
def calculate_perplexity_quadgram_smoothing(quadgrams_count,trigrams_count, quadgram_size, vocab_size) :
    test_set=pd.read_csv("C:\\Users\\Darshi Doshi\\Desktop\\test_set_preprocessed11.csv")
    sentences=test_set['Comment']
    total_sentences=len(sentences)
    perplexity=[]
    not_perplexable=[]
    not_perplex=[]
    for sentence in sentences :
        words=(str(sentence).split())
        probability=0 #it will be the log of the actual probability
        perplex=1
        words_in_sent=len(words)
        if(words_in_sent < 1):
            not_perplexable.append(sentence)
            continue
        for i in range(len(words)-2) :
            chk = 0
            if (i == 0):
                if (('<s>','<s>','<s>',words[i]) not in quadgrams_count):
                    quadgrams_count[('<s>','<s>','<s>',words[i])] = 0
                probability += math.log((quadgrams_count[('<s>','<s>','<s>',words[i])] + 1)/(total_sentences + vocab_size))
                chk = 1
            if ((i==1) and (('<s>','<s>',words[i - 1]) in trigrams_count)):
                if (('<s>','<s>',words[i - 1],words[i]) not in quadgrams_count):
                    quadgrams_count[('<s>','<s>',words[i - 1],words[i])] = 0
                probability += math.log((quadgrams_count[('<s>','<s>',words[i-1],words[i])] + 1)/(trigrams_count[('<s>','<s>',words[i - 1])] + vocab_size))
                chk = 1
            if ((i==2) and (('<s>',words[i-2],words[i - 1]) in trigrams_count)):
                if (('<s>',words[i-2],words[i - 1],words[i]) not in quadgrams_count):
                    quadgrams_count[('<s>',words[i-2],words[i - 1],words[i])] = 0
                probability += math.log((quadgrams_count[('<s>',words[i-2],words[i-1],words[i])] + 1)/(trigrams_count[('<s>',words[i-2],words[i - 1])] + vocab_size))
                chk = 1
            if (i==len(words)-3):
                if ((words[i],words[i+1],words[i+2],'</s>') not in quadgrams_count):
                    quadgrams_count[(words[i],words[i+1],words[i+2],'</s>')] = 0
                if ((words[i+1],words[i+2],'</s>','</s>') not in quadgrams_count):
                    quadgrams_count[(words[i+1],words[i+2],'</s>','</s>')] = 0
                if ((words[i+2],'</s>','</s>','</s>') not in quadgrams_count):
                    quadgrams_count[(words[i+2],'</s>','</s>','</s>')] = 0
                if ((words[i],words[i+1],words[i+2]) in trigrams_count):
                    probability += math.log((quadgrams_count[(words[i],words[i+1],words[i+2],'</s>')] + 1)/(trigrams_count[(words[i],words[i+1],words[i+2])] + vocab_size))
                    chk = 1
                if ((words[i+1],words[i+2],'</s>') in trigrams_count):
                    probability += math.log((quadgrams_count[(words[i+1],words[i+2],'</s>','</s>')] + 1)/(trigrams_count[(words[i+1],words[i+2],'</s>')] + vocab_size))
                    chk = 1
                if ((words[i+2],'</s>','</s>') in trigrams_count):
                    probability += math.log((quadgrams_count[(words[i+2],'</s>','</s>','</s>')] + 1)/(trigrams_count[(words[i+2],'</s>','</s>')] + vocab_size))
                    chk = 1
            if ((i != len(words)-3) and ((words[i],words[i+1],words[i+2]) in trigrams_count)):
                if ((words[i],words[i+1],words[i+2],words[i+3]) not in quadgrams_count):
                    quadgrams_count[(words[i],words[i+1],words[i+2],words[i+3])] = 0
                probability += math.log((quadgrams_count[(words[i],words[i+1],words[i+2],words[i+3])] + 1)/(trigrams_count[(words[i],words[i+1],words[i+2])] + vocab_size))
                chk = 1
            if chk == 0:
                probability=0

        if probability != 0:
            #perplex=(1/(probability))**(1/(words_in_sent))
            perplex = math.exp(-probability/words_in_sent)
            #perplexity.append(perplex)
            if perplex >= 10000000000:
                not_perplexable.append(sentence)
                not_perplex.append((perplex,probability))
            else :
                perplexity.append(perplex)
        else :
            not_perplexable.append(sentence)
            not_perplex.append((perplex,probability))
            
    average_perplexity=sum(perplexity)/len(perplexity)
    return average_perplexity, not_perplexable, perplexity, not_perplex

average_perplexity, not_perplexable, perplexity, not_perplex_quadgram=calculate_perplexity_quadgram_smoothing(quadgrams,trigrams,quadgrams_size,len(vocabulary))

print("the average perplexity over all the sentences that are perplexable is ", average_perplexity)
print("the total sentences in test set are ", len(perplexity)+ len(not_perplexable))
print("the total no of not_perplexable sentences in the validation set are :", len(not_perplexable))

##OUTPUT
#the average perplexity over all the sentences that are perplexable is  140442.97563929934
#the total sentences in test set are  67870
#the total no of not_perplexable sentences in the validation set are : 7594
