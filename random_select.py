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

def sint_maker_1(p, a, mas1, mas2):
    """WITH PROBABILITY '1 - P' SELECT DATA FROM FIRST ARRAY AND 'P' SECOND ARRAY
    but all make in order

    first argument: probability of entering element from mas2 to mas1 in order;
    second argument: emprt array for write;
    third argument: first array;
    fourth argument: second array.
    """

    if (len(mas1) == 0):
        return a
    if (np.random.random() > p):
        a = np.append(a, mas1[0])
    else:
        a = np.append(a, mas2[0])
    return sint_maker_1(p, a, mas1[1:], mas2[1:])

def sint_maker(p, mas1, mas2):
    """WITH PROBABILITY '1 - P' SELECT DATA FROM FIRST ARRAY AND 'P' SECOND ARRAY
    but all make in order

    first argument: probability of entering element from mas2 to mas1 in order;
    second argument: first array;
    third argument: second array.
    """

    return sint_maker_1(p, np.array([]), mas1, mas2)

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

mas3 = sint_maker(p, mas1, mas2)

print(mas3)