import os

def rec_down(mini_path):
    for i in os.listdir(path = mini_path):
        if (os.path.isdir(i)):
            rec_down(i)
        else: print(os.path.abspath(i), sep = '\n')


my_path = str(input())

os.path.abspath(my_path)

rec_down(my_path)
    

a = input()
