import core.dl_framework.data as Data


def other_func(a):
    print(a)


def main():
    x = 1
    y = 2
    y = Data.Dataset(x, y)
    other_func(x)

