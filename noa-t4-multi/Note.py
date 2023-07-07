from collections import deque
from datetime import datetime
from pprint import pprint

class Note:
    def __init__(self, note_path="__note.txt"):
        self.note_path = note_path
        self.buffer = deque()
    
    def __del__(self):
        if (self.buffer):
            print(f"Note - Warning: leftover note in memory not flushed to file: {self.buffer}")
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
        
        with open(self.note_path, "a") as file:
            while self.buffer:
                file.write(self.buffer.popleft())
            
    def describe(self):
        print(f"Note path: {self.note_path}")
    

if __name__ == '__main__':
    # note = Note("__note-test.txt")
    # note.addline("2")
    # note.addlines(["hello", "words"])
    # note.flush()
    import os
    if not os.path.exists("./on_train_done.sh.off.__ignore__"):
        print("WWW")
    else:
        print("OK see it")