
import numpy as np
import nltk
import torch
class Lang:
	def __init__(self):
		self.word2idx = { "<SOS>" :0 ,  "<EOS>":1 , "<UNK>":2   }
		self.idx2word = { 0: "<SOS>", 1: "<EOS>" , 2:"<UNK>"  }
		self.word2count = { }
		self.n_words = 3

	def addSentence(self, sentence):
		for word in nltk.word_tokenize(sentence):
			self.addWord(word)

	def addWord(self, word):

		if word not in self.word2idx:
			self.word2idx[word] = self.n_words
			self.word2count[word] = 1
			self.idx2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	def string2idx(self,sentence , max_length=None):
		res = []
		for word in sentence.split(' '):

			if word in self.word2idx:
				res.append( self.word2idx[word] )
			else:
				res.append( self.word2idx["<UNK>"] )
		# res = np.array(res)
		if max_length is not None:
			l = len(res)
			for i in range(l, max_length ):
				res.append( self.word2idx["<EOS>"] )
		return res

	def idx2string(self,idxs):
		res = ''
		for idx in idxs:
			if idx == 1 :
				break
			if idx in self.idx2word:
				res	+= self.idx2word[idx] 
			else:
				res+=  "<UNK>" 
			res += ' '
		return res




# test_dialog = [
# 	("hello , my name is Tony Stark" , "Nice to meet you , Tony Stark"),
# 	("Who is Tony Stark ?","Tony Stark is a scientist"),
# 	("What's your name ?" , "My name is Tony Stark") ,
# 	("What are you doing ?" , "I am reading book"),
# 	("hello , my name is Bruce Lee" , "Nice to meet you , Bruce Lee "),
# 	("I was born in ShangHai" , "I also come from ShangHai  " ),
# 	("How much is this book ?" , "This book is 10 dollars "),
# 	("Who is your favourite celebrity?" , "Tony Stark"),
# 	("What are you planning to do tomorrow?" , "I am going to study in the library")
# ]







def dialog2tensor(lang,dialog , max_length = 10):

	for (i,pairs) in enumerate(dialog):
		lang.addSentence(pairs[0])
		lang.addSentence(pairs[1]) 
	source_list = []
	target_list = []
	for (i , pairs) in enumerate(dialog):
		source_list.append( lang.string2idx(pairs[0] ,max_length)  )
		target_list.append( lang.string2idx(pairs[1] ,max_length)  )
	source_tensor = torch.tensor(source_list,dtype=torch.long)
	target_tensor = torch.tensor(target_list,dtype=torch.long)
	return source_tensor , target_tensor

def tensor2string(lang,tensors):
	arr = tensors.numpy()
	for a in arr:
		print( lang.idx2string(a) )

# test_dialog = [
#     ("hello , my name is Tony Stark" , "Nice to meet you , Tony Stark"),
#     ("Who is Tony Stark ?","Tony Stark is a scientist"),
#     ("What's your name ?" , "My name is Tony Stark") ,
#     # ("What are you doing ?" , "I am reading book"),
#     # ("hello , my name is Bruce Lee" , "Nice to meet you , Bruce Lee "),
#     # ("I was born in ShangHai" , "I also come from ShangHai  " ),
#     # ("How much is this book ?" , "This book is 10 dollars "),
#     # ("who is your favourite celebrity?" , "Tony Stark"),
#     # ("What are you planning to do tomorrow?" , "I am going to study in the library")
# ]
# lang = Lang()
# input_tensor,target_tensor = dialog2tensor(lang, test_dialog ,max_length=7)
# print(input_tensor)
# print(target_tensor)



