import xml.dom.minidom as xmldom
import os

import nltk

s= "Nice to meet you,Tony Stark"

tokens = nltk.word_tokenize(s)
print(tokens)