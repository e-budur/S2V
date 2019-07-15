import os
import sys
import sentencepiece as spm
import fnmatch
import glob

print(sys.argv)
input_dirs = sys.argv[1]
input_files_pattern = sys.argv[2]
output_model_filename = sys.argv[3]
vocab_size = int(sys.argv[4])
input_sentence_size = sys.argv[5]
matches = []

for input_dir in input_dirs.split('|'):
  for root, dirnames, filenames in os.walk(input_dir):
    for filename in fnmatch.filter(filenames, input_files_pattern):
        matches.append(os.path.join(root, filename))

filenames = ','.join(matches)
print(filenames)
input_params = '--input='+filenames+' --model_type=bpe --model_prefix='+output_model_filename+' --model_type=bpe --vocab_size='+str(vocab_size)+' --hard_vocab_limit=false --character_coverage=1.0 '+input_sentence_size
params = (input_params)

# Train model on the input text file
spm.SentencePieceTrainer.Train(params)

