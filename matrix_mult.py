import argparse

def read_matr(file, matr):
    """FUNCTION FOR READ MATRICE FROM THE OPENED FILE

    first argument: opened file;
    second argument: matrice for saving info.

    return 1 if bad matrice.
    """
    
    stroka = file.readline()
    
    while (stroka != "\n" and stroka != ""):
        if (stroka.find('\n') != -1):
            stroka = stroka[:len(stroka) - 1]   #remove \n in the end of line
            
        buf = stroka.split(" ")
        
        for i in range(len(buf)):
            try:
                buf[i] = int(buf[i])
            except ValueError:
                print("THIS MATRICE HAVE FORBIDDEN SIMBOLS")
                return 1
        
        matr.append(buf)
        stroka = file.readline()

def row_check(matr):
    """CHECK MATRICE FOR NON-EQUAL NUMVER OF ROWS

    first argument: A matrice.

    return 1 if bad matrice.
    """
    
    lenght = len(matr[0])
    for i in matr[1:]:
        if (len(i) != lenght):
            return 1

def check_matrice(matr1, matr2):
    """CHECK MATRICE FOR NON-EQUAL NUMVER OF ROWS AND POSSIBILITY TO MULTIPLY

    first argument: A matrice;
    second argument: B matrice.

    return 1 if bad matrices.
    """
    
    if (len(matr1[0]) != len(matr2) or
        row_check(matr1) or
        row_check(matr2)):
        return 1

def mult_matrices(matr1, matr2):
    """MULTIPLY TWO MATRICES

    first argument: A matrice;
    second argument: B matrice.

    return mlt of matrices
    """
    
    matr3 = [[0 for i in range(len(matr2[0]))] for j in range(len(matr1))]
    for i in range(len(matr1)):
        for j in range(len(matr2[0])):
            for k in range(len(matr2)):
                matr3[i][j] += matr1[i][k] * matr2[k][j]
    return matr3

parser = argparse.ArgumentParser()
parser.add_argument('frm', type = str)
parser.add_argument('to', type = str)

args = parser.parse_args()
from_file = args.frm
to_file = args.to

file = open(from_file)

matr1 = []
matr2 = []

if (read_matr(file, matr1) or
    read_matr(file, matr2)):
    print("CAN'T DO OPERATION")
    
else:
    if (check_matrice(matr1, matr2)):
        print("WRONG SIZES OF MATRICES")
    
    else:
        matr3 = mult_matrices(matr1, matr2)
        
        file = open(to_file, 'w')
        for i in range(len(matr3)):
            for j in range(len(matr3[0])):
                file.write(str(matr3[i][j]) + " ")
            file.write("\n")

file.close()
