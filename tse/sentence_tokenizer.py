import spacy
import os
import sys
from joblib import Parallel, delayed
#%%
def split_paragraphs_to_sentences(read_filename,read_folder,out_file_path):
    # Original files are organized paragraph by paragraph. Split each of them
    with open(read_folder + read_filename,'r',encoding='utf8') as f:
        raw_data = f.read()
        paragraphs = raw_data.split('\n\n')
    
    my_sentences = []
    iteration_count = 1
    # Iterate through paragraphs and append sentences.
    for paragraph in paragraphs:
        sentences = nlp(paragraph)
        for sentence in sentences.sents:
            my_sentences.append(sentence.text)
        my_sentences.append('\n')
        if iteration_count % 10000 == 0:
            print(iteration_count, read_filename)
        iteration_count = iteration_count + 1
    # Write sentences to a file paragraph by paragraph as in the input    
    with open(out_file_path + 'detected_sentences_' + read_filename + '.txt','w',encoding='utf8') as writer:
        writer.write('\n'.join(my_sentences))
#%%
# This script expects 4 terminal parameter that language, read and write folders and n_jobs. An example run from terminal is as follows:
# python ./sentence_tokenizer.py en ./UMBC-SMALL ./UMBC-SMALL_SENTENCES 6
if len(sys.argv) != 5:
    print('Invalid number of params')
    sys.exit(0)

# Just read the params and init spacy module.
language = sys.argv[1]
read_folder = sys.argv[2]
out_folder = sys.argv[3]
n_jobs = int(sys.argv[4])
nlp = spacy.load(language)

# Just a simple correcter for folder names
if not read_folder.endswith('/'):
    read_folder = read_folder + '/'

if not out_folder.endswith('/'):
    out_folder = out_folder + '/'

# Read all txts in the specified folder
files = os.listdir(read_folder)
files = [file for file in files if file.endswith('.txt')]

# Run the code in parallel for each file
Parallel(n_jobs=n_jobs,prefer='threads')(delayed(split_paragraphs_to_sentences)(filename, read_folder,out_folder) for filename in files)

