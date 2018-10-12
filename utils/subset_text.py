import os 

# Print the number of lines in your document
num_lines = sum(1 for line in open('filepath'))
print(num_lines)

# Splits the file located at 'filepath' into sub-files of length 'lines_per_file'
def split_file(filepath, lines_per_file=1000):
    lpf = lines_per_file
    path, filename = os.path.split(filepath)
    with open(filepath, 'r') as r:
        name, ext = os.path.splitext(filename)
        try:
            w = open(os.path.join(path, '{}_{}{}'.format(name, 0, ext)), 'w')
            for i, line in enumerate(r):
                if not i % lpf:

                    w.close()
                    filename = os.path.join(path,
                                            '{}_{}{}'.format(name, i, ext))
                    w = open(filename, 'w')
                w.write(line)
        finally:
            w.close()

# Example usage
split_file('trwiki/trwikiRaw.txt')