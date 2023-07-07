import os
import datetime

def timestamp():
    """like 20230613_001228"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def self_name():
    return ".".join(os.path.basename(__file__).split(".")[0:-1])

if __name__ == '__main__':
    print(timestamp())
    print(self_name())
    