from irt import irt
from utils import open_xlsx



def main():
    df = open_xlsx('test.xlsx')
    result = irt(df)
    print(result.abilities)
    print(result.difficult)
    print(result.rejected_subjects)
    print(result.rejected_tasks)
    print(result.err)


if __name__ == '__main__':
    main()