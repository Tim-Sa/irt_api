from irt import irt
from utils import open_xlsx



def main():
    df = open_xlsx('test.xlsx')
    print(irt(df))


if __name__ == '__main__':
    main()