import string

name = input()
file = open(name)

str_count = 0
word_count = 0
alpha_count = 0

stroka = file.readline()

while stroka != "":
    str_count += 1

    lst = stroka.split(" ")
    while(1):
        try:
            lst.remove("")
        except ValueError:
            break

    try:
        lst.remove("\n")
    except ValueError:
        pass
        
    word_count += len(lst)

    for i in stroka:
        if(i.isalpha()):
            alpha_count += 1

    stroka = file.readline()

print("Str conut: ", str_count, "\nWord count: ", word_count, "\nAlpha count: ", alpha_count)

file.close()
