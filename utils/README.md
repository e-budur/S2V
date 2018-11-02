# Utilities

This folder contains helper scripts for subsetting and preprocessing text files. Explanation and usage of each script can be seen below, new scripts will be added along the project.


### Contents
* subset_text.py       ->        Subsets text files into smaller chunks


### subset_text.py

This file has a single function that splits the .txt file located at 'filepath' into sub-files of length 'lines_per_file'. The split_file function doesn't shuffle the document and preserves the context.

To run the file, you can simply enter the filepath and the number of lines of the split. Alternatively, you can use the defined num_lines (total number of lines) variable and the desired split percentage to subset the file as below.

```
# A %5 split
fraction = 5 / 100  
lines_per_file = math.ceil(num_lines /fraction)
```


