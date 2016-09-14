import pandas as pd
from datetime import datetime


def main():
    start = datetime.now()
    filepath = 'data/train.csv'
    train = pd.read_csv(filepath)
    rows = len(train)

    # data for sub_training step (10% of train)
    sub_train_len = int(0.1 * rows)
    sub_train = train.loc[0:sub_train_len - 1,]

    sub_train.to_csv('data/sub_train.csv', index=False)

    # data for validating step (5% of test)
    validate_len = int(0.05 * rows)
    validate = train.loc[sub_train_len:validate_len+sub_train_len - 1,]
    validate.to_csv('data/validate.csv', index=False)

    # data for feature engineering step (the rest 85% of of train)
    statistics = train.loc[validate_len+sub_train_len:,]

    # add number of purchases for hotel id

    # TODO: add features

    time = datetime.now() - start
    print 'total processing time: %d' % int(time.seconds)

if __name__ == '__main__':
    main()



