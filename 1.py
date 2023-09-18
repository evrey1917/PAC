class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = 16
        
    def update_count(self, val):
        if val <= self._max_count and val >= 0:
            self._count = val
            return True
        else:
            print("YOU CAN'T DO THIS, MORTAL BEING")
            return False
        
    # Свойство объекта. Не принимает параметров кроме self, вызывается без круглых скобок
    # Определяется с помощью декоратора property
    @property
    def count(self):
        return int(self._count)
    
    
    # Ещё один способ изменить атрибут класса
    @count.setter
    def counts(self, val):
        self._count = val
        if val <= self._max_count:
            self._counts = val
        else:
            pass
    
    @staticmethod
    def static():
        print('I am function')
    
    @classmethod
    def my_name(cls):
        return cls.__name__

    def __add__(self, num):
        """ Сложение с числом """
        self.count += num
        return self
    
    def __mul__(self, num):
        """ Умножение на число """
        self.count *= num
        return self
    
    def __lt__(self, num):
        """ Сравнение меньше """
        return self.count < num

    def __bt__(self, num):
        """ Сравнение больше """
        return self.count > num

    def __le__(self, num):
        """ Сравнение меньше или равно """
        return self.count <= num

    def __be__(self, num):
        """ Сравнение больше или равно """
        return self.count >= num

    def __eq__(self, num):
        """ Сравнение равно """
        return self.count == num

    def __iadd__(self, num):
        """ += """
        self.update_count(self.count + num)
        return self

    def __isub__(self, num):
        """ -= """
        self.update_count(self.count - num)
        return self

    def __imul__(self, num):
        """ *= """
        self.update_count(self.count * num)
        return self
    
    def __len__(self):
        """ Получение длины объекта """
        return self.count


class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe


class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation
        
    @property
    def eatable(self):
        return self._saturation > 0


class Banana(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='yellow', saturation=10):
        
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color


class Pineapple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='yellow-brown', saturation=10):
        
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color


class Pizza(Food):
    def __init__(self, count=1, max_count=32, color='multi-color', saturation=10):
        
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color


class Mozarella(Food):
    def __init__(self, count=1, max_count=32, color='white-yellow', saturation=10):
        
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

class Inventory():
    def __init__(self, lenght):
        self._spisok = [None] * lenght

    def __getitem__(self, key):
        return self._spisok[key]

    def __setitem__(self, key, item):
        if(isinstance(item, Food) and item.eatable):
            self._spisok[key] = item
        return self._spisok[key]
    
    def sub(self, num, key):
        """ -= """
        if(self._spisok[key] != None):
            self._spisok[key] -= num
            if(self._spisok[key] == 0):
                self._spisok[key] = None
        return self

asas = Item()


piz = Pizza()

sps = Inventory(4)

print(sps[2])
sps[2] = piz

print(sps[2].count)
sps.sub(1, 2)
print(sps[2])
