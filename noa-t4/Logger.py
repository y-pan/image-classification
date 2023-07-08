from collections import deque
from datetime import datetime
from pprint import pprint

class Logger:
    def __init__(self, logger_path="logger.log"):
        self.logger_path = logger_path
        self.buffer = deque()
    
    def __del__(self):
        if (self.buffer):
            print(f"Logger - Warning: leftover logger in memory not flushed to file: {self.buffer}")
        pass

    def add(self, text):
        self.buffer.append(text)
    
    def addline(self, text):
        self.buffer.append(text)
        self.buffer.append("\n")

        if len(self.buffer) > 100:
            self.flush()

    def newline(self):
        self.buffer.append("\n")

    def hr(self):
        self.buffer.append("__________________________________________________________________________________________________\n")
        
    def add_(self, text):
        print(text)
        self.add(text)

    def addline_(self, text):
        print(text)
        self.addline(text)

    def addlines(self, textlist):
        for text in textlist:
            self.addline(str(text))
    
    def addlines_(self, textlist):
        pprint(textlist)
        self.addlines(textlist)
    
    def newline_(self):
        print("")
        self.newline()

    def addtimeline(self):
        t = datetime.now()
        self.buffer.append(f"Time: {t}")
        self.buffer.append("\n")
    
    def addtimeline_(self):
        t = datetime.now()
        print(t)
        self.buffer.append(f"Time: {t}")
        self.buffer.append("\n")

    def flush(self):
        if not self.buffer:
            return
        
        with open(self.logger_path, "a") as file:
            while self.buffer:
                file.write(self.buffer.popleft())
            
    def describe(self):
        print(f"Logger path: {self.logger_path}")
    

if __name__ == '__main__':
    logger = Logger()
    logger.addline("2")
    logger.addlines(["hello", "words"])
    logger.flush()
