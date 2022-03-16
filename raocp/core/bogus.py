def add(i):
    return i + 1


def main():
    x = 0
    while True:
        x = add(x)
        if x == 1000000:
            x = 0
