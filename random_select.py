import numpy as np
import argparse

def read_array_1(line, a):
    """READ INT ARRAY FROM STRING ARRAY
    
    first argument: array of strings;
    second argument: empty array for write.
    """

    if (len(line) == 0):
        return a
    a = np.append(a, int(line[0]))
    return read_array_1(line[1:], a)

def read_array(line):
    """READ INT ARRAY FROM STRING ARRAY
    
    first argument: array of strings.
    """

    return read_array_1(line, np.array([]))

parser = argparse.ArgumentParser()
parser.add_argument('f1', type = str)
parser.add_argument('f2', type = str)
parser.add_argument('p', type = float)

args = parser.parse_args()

p = args.p

file = args.f1
file = open(file)
line1 = file.readline().split(' ')

file = args.f2
file = open(file)
line2 = file.readline().split(' ')
file.close()

if (len(line1) != len(line2)):
    print("DIFFERENT SIZES")
    exit()

mas1 = read_array(line1)
mas2 = read_array(line2)

uber = list(map(lambda a, b: a if np.random.random() > p else b, mas1, mas2))
print(uber)
