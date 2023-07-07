import datetime

def timestamp():
    """like 20230613_001228"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == '__main__':
    print(timestamp())
    