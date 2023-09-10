dct = {"one" : "1", "two" : "2", "couple" : "tuple"}

stroka = input()

for i in dct.items():
    if(stroka.find(i[0]) != -1):
        partit = stroka.split(i[0])
        stroka = ""
        for j in range(0, len(partit) - 1):
            stroka += partit[j] + i[1]
        stroka += partit[len(partit) - 1]

print(stroka)
