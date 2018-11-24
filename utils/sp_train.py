import os
import sys
import sentencepiece as spm

input_params = '--input='+sys.argv[1]+' --model_prefix=spm --model_type=bpe --vocab_size=8000 --character_coverage=1.0'
params = (input_params)

# Train model on the input text file
spm.SentencePieceTrainer.Train(params)

'''
sp = spm.SentencePieceProcessor()
sp.Load('spm.model')

print(sp.EncodeAsPieces('Hello world.'))
print(sp.EncodeAsIds('Hello world.'))
print(sp.DecodeIds([151, 88, 21, 887, 6]))
'''