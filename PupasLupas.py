class Accountant:
    """MAN, WHO GIVE YOU SALARY"""
    def __init__(self, salary):
        self._salary = salary
        
    def give_salary(self, worker):
        """METHOD FOR GIVING SALARY
        first argument - worker
        """
        if(isinstance(worker, Worker)):
            worker.take_salary(self._salary)
            return 0
        return 1

class Worker:
    def __init__(self, money = 0, **kwargs):
        super().__init__(**kwargs)
        self._money = money

    @property
    def cash(self):
        return int(self._money)

    def read_matr(self, file, matr):
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

    def row_check(self, matr):
        """CHECK MATRICE FOR NON-EQUAL NUMVER OF ROWS

        first argument: A matrice.

        return 1 if bad matrice.
        """
        
        lenght = len(matr[0])
        for i in matr[1:]:
            if (len(i) != lenght):
                return 1

    def check_matrice(self, matr1, matr2):
        """CHECK MATRICE FOR NON-EQUAL NUMVER OF ROWS AND POSSIBILITY TO MULTIPLY

        first argument: A matrice;
        second argument: B matrice.

        return 1 if bad matrices.
        """
        
        if (len(matr1[0]) != len(matr2) or
            self.row_check(matr1) or
            self.row_check(matr2)):
            return 1

    def mini_work(self, matr1, matr2, simbol):
        """PLUS OR MINUS MATRICES

        first argument: A matrice;
        second argument: B matrice;
        third argument: simbol '+' or '-'.

        return None if bad simbol.
        """
    
        matr3 = [[0 for i in range(len(matr1[0]))] for j in range(len(matr1))]
        
        for i in range(0, len(matr1)):
            for j in range(0, len(matr1[0])):
                if simbol == '+':
                    matr3[i][j] = matr1[i][j] + matr2[i][j]
                else:
                    if simbol == '-':
                        matr3[i][j] = matr1[i][j] - matr2[i][j]
                    else:
                        print("FORBIDDEN SIMBOL")
                        return None
        return matr3

    def big_work(self, filename1, filename2, simbol):
        """PLUS OR MINUS MATRICES FROM FILE1 AND FILE2

        first argument: first file;
        second argument: second file;
        third argument: simbol '+' or '-'.

        return None if bad simbol.
        """
        
        file1 = open(filename1)
        file2 = open(filename2)
        matr1 = []
        matr2 = []
        if (self.read_matr(file1, matr1) or
            self.read_matr(file2, matr2)):
            print("CAN'T DO OPERATION")
            file1.close()
            file2.close()
            return 1
        file1.close()
        file2.close()
        if (self.check_matrice(matr1, matr2)):
            print("WRONG SIZES OF MATRICES")
            return 1
        matr3 = self.mini_work(matr1, matr2, simbol)

        for i in range(len(matr3)):
            for j in range(len(matr3[0])):
                print(str(matr3[i][j]) + " ", end = "")
            print("")
    
    def take_salary(self, salary):
        """PLUS OR MINUS MATRICES

        first argument: how many money.
        """
        self._money += salary

    
class Pupa(Worker):
    def do_work(self, filename1, filename2):
        """PLUS MATRICES FROM FILE1 AND FILE2

        first argument: first file;
        second argument: second file;

        return 1 if bad matrices.
        """
        
        if(self.big_work(filename1, filename2, '+')):
            print("ERROR")
            return 1
        return 0


class Lupa(Worker):
    def do_work(self, filename1, filename2):
        """MINUS MATRICES FROM FILE1 AND FILE2

        first argument: first file;
        second argument: second file;

        return 1 if bad matrices.
        """
        
        if(self.big_work(filename1, filename2, '-')):
            print("ERROR")
            return 1
        return 0

man = Accountant(salary = 6)

pupich = Pupa()
lupich = Lupa()

if(man.give_salary(pupich)):
    print("Error")

print("Pupas: ", pupich.cash)
print("Lupas: ", lupich.cash)

pupich.do_work("1.txt", "2.txt")