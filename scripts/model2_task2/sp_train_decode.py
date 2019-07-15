import os
import sys
import sentencepiece as spm
import fnmatch
import glob

sp = spm.SentencePieceProcessor()
sp.Load('/truba/home/ebudur/tse-s2v/data/bulk_sentences/tr/TR/sentencepiece/spm.model')

sentence ='merhaba arabalar evler eller gitmek gider yapabilir miyim ister misin'
encoded_pieces = sp.EncodeAsPieces(sentence)
encoded_ids = sp.EncodeAsIds(sentence)
decoded_sentence = sp.DecodeIds(encoded_ids)

print(sentence)
print(encoded_pieces)
print(encoded_ids)
print(decoded_sentence)
