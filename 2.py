stroka = input();

words = stroka.split(' ')
buf = ""
lenght = 0;

for word in words:
    if lenght < len(word):
        lenght = len(word)
        buf = word

print("Longest word: {0}, contains {1} simbols.".format(buf, lenght))
