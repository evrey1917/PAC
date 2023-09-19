class Stack(list):
    """STACK. USE RULE OF LIFO - LAST IN FIRST OUT"""

    def __init__(self):
        self._list = []     #   from list we have 'append' and 'pop' already, so, don't do anything

class Queue(list):
    """QUEUE. USE RULE OF FIFO - FIRST IN FIRST OUT"""

    def __init__(self):
        self._list = []

    append = property(doc='(!) Disallowed inherited')   #   disable these methods, because
    pop = property(doc='(!) Disallowed inherited')      #   they made Stack from Queue
    
    def queue(self, elem):
        self._list.append(elem)
    
    def enqueue(self):
        if (len(self._list) != 0):
            elem = self._list[0]
            self._list = self._list[1:]
            return elem
        else:
            print("NO ELEMENTS HERE, MORTAL")
            return -1
    
    def copy(self):
        return self._list.copy()

class Zamn():
    """HERE WE HAVE QUEUE. THAT'S ALL"""

    def __init__(self):
        self.__queue
    
    def queue(self, elem):
        self.__queue.queue(elem)
    
    def enqueue(self):
        return self.__queue.enqueue()
    
    def copy(self):
        return self.__queue.copy()

    __queue = Queue()