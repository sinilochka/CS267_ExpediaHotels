import pandas as pd


def multi_class(row):
    click = row['click_bool']
    book = row['booking_bool']
    if int(book) == 1:
        return 2
    elif int(book) == 0 and int(click) == 1:
        return 1
    elif int(book) == 0 and int(click) == 0:
        return 0

def main():
    data = pd.read_csv('data/train.csv')

    part_1 = data[data['click_bool'] == 1]

    select_rows = int(200000)
    part_2 = data[data['click_bool'] == 0].iloc[:select_rows, :]

    result = pd.concat([part_1, part_2])
    result['class'] = result.apply(multi_class, axis=1)
    result.to_csv('data/sample_1.csv', index=False)

if __name__ == '__main__':
    main()