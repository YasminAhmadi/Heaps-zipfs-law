import operator
import re
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import special
import numpy as np
import nltk
from nltk import word_tokenize
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
#x is a dictionary containing each distinct word in the text as keys and their frequencies as corresponding values
x = {}
#x2 is a dictionary containing number of distinct words in the text at each point of time as keys and number of words until encountering the accoring new word as values
x2={}
tokens=[]
numberofvocab=0
numberoftokens=0
for w in open('filename.txt').read().split():
    #lower and uppercase don't matter if letters of the words are same, it must not count it as a new word
    w = w.lower()
    tokens.append(w)
    numberoftokens += 1
    if w in x:
        x[w] += 1
    else:
        numberofvocab += 1
        x[w] = 1
        #here in else, numberofvocab is added by 1, so we add an item storing the according number of tokens with the updated value of numberofvocab
        x2[numberofvocab] = numberoftokens
x3=x2

#sorting the dictionary by values in descending order so the most frequent word will be on top and the least at the bottem
#Therefore each word's rank is its index in the values list of x dict
counts = dict( sorted(x.items(), key=operator.itemgetter(1),reverse=True))

#here we print 10 most used distinct words in our text next to their frequency in descending order
i=0
print("Zipf's law:")
print("word: | frequency:")
for distinctWord, WordOccurrences in counts.items():
    print (distinctWord,"      ", WordOccurrences)
    i+=1
    #eliminate this condition to see a list of all the words in the vocab
    if(i==25):
        break
print()
i=0
#here we show the number of words(tokens) used in a text until encountering 25 distinct words step by step
print("Heap's law:")
print("VocabSize: | CollectionSize:")
for distinctWord, WordOccurrences in x2.items():
    print (distinctWord,"      ", WordOccurrences)
    i+=1
    #eliminate this condition to see a list of all the words in the vocab
    if(i==50):
        break

totalnum = 0
vocab=[]
#counting the total number of words in the text
for word, times in x.items():
    vocab.append(word)
    totalnum+=times
print()
print ("total number of words(tokens) in the text: ",totalnum)
print()
#the number of distinct words is the same as the size of the dict
vocabnum = len(x)
print ("total number of distinct words(vocab) in the text: ",vocabnum)
print()

dataframe = pd.DataFrame(index = counts.keys(), columns = ['Count','Rank'], dtype = 'float64')
for a in dataframe.index:
    dataframe.loc[a,'Count'] = counts[a]
dataframe['Rank'] = dataframe['Count'].rank(method = 'min', ascending = False)

x = np.log(dataframe['Rank'])
y = np.log(dataframe['Count'])
fit = np.polyfit(x, y, deg=1)

#finding the line equation y=mx+b   => m=? & b=?
polyy=np.poly1d(fit)
y1=(polyy(2))
y2=(polyy(4))
#slope m=(y2-y1 / x2-x1)
m=(y2-y1)/(4-2)
#what's y-intercept?
b=(polyy(0))
print("Zipf's Law --> The line equation is:","y=",m,"x+",b)
print()
fitted = fit[0] * x + fit[1]
#Log(f) = Log(c) – s Log(r)
print("Zipf's Law --> Estimation of parameter s is:",fit[0])

#first graph

fig = plt.Figure(figsize = (4,4), facecolor = 'w', edgecolor = 'w')
aa = plt.subplot(111)
aa.plot(x, y, 'bo', alpha = 0.5)
aa.plot(x,fitted,'r')
aa.set_title(' Zipf’s Law, \nLog(Rank) vs. Log(Count)')
aa.set_xlabel('Log(Rank)')
aa.set_ylabel('Log(Count)')
aa.set_xlim(left = max([min(np.log(dataframe['Rank'])) * 0.95,0]))
plt.tight_layout()
plt.show()

#second graph

#x3.values() --> collection size
#x3.keys() --> vocab size
xpoints = list(x3.values())
ypoints = list(x3.keys())
xs = np.array(np.log(xpoints), dtype=np.float64)
ys = np.array(np.log(ypoints), dtype=np.float64)

#finding m and b of the line equation
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b
m, b = best_fit_slope_and_intercept(xs, ys)

print()
print("Heap's Law --> The line equation is: y= ",m,"x+", b)

#Now we just need to create a line for the data
regressionline = []
for xsel in xs:
    regressionline.append((m*xsel)+b)

style.use('ggplot')
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regressionline)
plt.xlabel('Log(Token)')
plt.ylabel("Log(Vocab)")
plt.title("Heap's Law\n Log(Token) vs. Log(Vocab)")
plt.show()