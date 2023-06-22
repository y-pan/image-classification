from collections import deque
from datetime import datetime


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
        return self
    
    def addline(self, text):
        self.buffer.append(text)
        self.buffer.append("\n")

        if len(self.buffer) > 100:
            self.flush()
        return self

    def newline(self):
        self.buffer.append("\n")
        return self

    def add_(self, text):
        print(text)
        self.add(text)
        return self

    def addline_(self, text):
        print(text)
        self.addline(text)
        return self

    def newline_(self):
        print("")
        self.newline()
        return self

    def addtimeline(self):
        t = datetime.now()
        self.buffer.append(f"Time: {t}")
        self.buffer.append("\n")
        return self
    
    def addtimeline_(self):
        t = datetime.now()
        print(t)
        self.buffer.append(f"Time: {t}")
        self.buffer.append("\n")
        return self

    def flush(self):
        if not self.buffer:
            return
        
        with open(self.note_path, "a") as file:
            while self.buffer:
                file.write(self.buffer.popleft())
        return self
            
    def describe(self):
        print(f"Note path: {self.note_path}")
        return self
    

if __name__ == '__main__':
   note = Note("__note-test.txt").add("1")\
    .addline("2").add("3").flush()