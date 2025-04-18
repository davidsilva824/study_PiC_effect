from nltk import word_tokenize, FreqDist
from nltk.util import bigrams

tokens = word_tokenize("rat catcher mice catcher rats catcher")
bgs = bigrams(tokens)
fdist = FreqDist(bgs)

print(fdist[('rat', 'catcher')])