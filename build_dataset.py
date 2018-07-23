
import numpy as np
class Lang:
	def __init__(self):
		self.word2idx = {}
		self.idx2word = { 0: "<SOS>", 1: "<EOS>" ,2:"<UNK>" }
		self.word2count = { }
		self.n_words = 3
		self.vocab_size = 3

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2idx:
			self.word2idx[word] = self.n_words
			self.word2count[word] = 1
			self.idx2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	def string2idx(self,sentence):
		res = []
		for word in sentence.split(' '):
			if word in self.word2idx:
				res.append( self.word2idx[word] )
			else:
				res.append( self.word2idx["<UNK>"] )
		# res = np.array(res)
		return res

	def idx2string(self,idxs):
		res = []
		for idx in idxs:
			if idx in self.idx2word:
				res.append( self.idx2word[word] )
			else:
				res.append( "<UNK>" )
		return res




test_dialog = [
	("hello , my name is Tony Stark . " , "Nice to meet you , Tony Stark "),
	("Who is Tony Stark ? ","Tony Stark is a scientist"),
	("hello , my name is Bruce Lee . " , " Hi , Bruce Li "),
]


lang = Lang()
for (i,pairs) in enumerate(test_dialog):
	lang.addSentence(pairs[0])
	lang.addSentence(pairs[1])



idx_dialog = []
for (i,pairs) in enumerate(test_dialog):
	t = []
	t.append(lang.string2idx(pairs[0]))
	t.append(lang.string2idx(pairs[1]))
	print(t)
	idx_dialog.append( t  )
	print('-'*8)
 
import torch
idx_arr = np.array(idx_dialog)

print(idx_arr)
print(idx_arr.shape)
