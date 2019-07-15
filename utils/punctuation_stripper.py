import codecs
import string
from unicode_tr import unicode_tr
import regex as re

file_names = ['/truba/home/ebudur/tse-s2v/data/bulk_sentences/tr/TR/trwiki/trwiki-extracted_raw_sentences_by_lines.txt',
              '/truba/home/ebudur/tse-s2v/data/bulk_sentences/tr/TR/milliyet/milliyet_news_sentences_utf8.txt',
              '/truba/home/ebudur/tse-s2v/data/bulk_sentences/tr/TR/hurriyet/hurriyet-all-nz.txt'
             ]

pattern = re.compile(r'\W+', re.UNICODE)
for filename in file_names:
    output_file = codecs.open(filename+'.no_punct.txt', mode='w', encoding='utf-8')
    for line in codecs.open(filename, mode='r', encoding='utf-8'):
        line = unicode_tr(line)
        line = line.lower()
        
        if len(line.strip())==0:
           continue
        line = ' '.join(pattern.split(line))+'\n'
        output_file.write(line)
    output_file.close()

