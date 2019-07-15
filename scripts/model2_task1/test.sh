#!/bin/bash
TOKENIZED_FILES="/truba/home/ebudur/tse-s2v/data/bulk_sentences/en/UMBC/txt_tokenized/mbta.com_mtu.pages*"
OUTPUT_FILE="/truba/home/ebudur/tse-s2v/data/bulk_sentences/en/UMBC/all_files.sp"


find $1 -type f -name '$TOKENIZED_FILES' -exec cat {} + > '$OUTPUT_FILES'
