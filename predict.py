import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
import numpy as np



def main():
    start = datetime.now()
    print start

    sub_train = pd.read_csv(sys.argv[1])
    validate = pd.read_csv(sys.argv[2])

    # target is 'booking_bool'
    # train features
    X_train = sub_train.iloc[:,:50].fillna(-100)
    X_train.drop(['position', 'date_time'], axis=1, inplace=True)
    X_train = X_train.values
    # train labels
    y_train = sub_train[['booking_bool']]
    y_train = np.ravel(y_train.values)

    # test features
    X_test = validate.iloc[:,:50].fillna(-100)
    X_test.drop(['position', 'date_time'], axis=1, inplace=True)
    X_test = X_test.values
    # test labels
    y_test = validate[['booking_bool']]
    y_test = np.ravel(y_test.values)

    print 'Learning Stage:'

    print 'Random Forest'

    clf0 = RandomForestClassifier(n_estimators=50, random_state=4321)
    clf0.fit(X_train, y_train)
    prediction0_on_train = clf0.predict_proba(X_train)
    prediction0_on_train = pd.DataFrame().append(sub_train[['srch_id', 'prop_id']],
                                                 ignore_index=True).join(pd.DataFrame(prediction0_on_train[:,1],
                                                                                      columns=['prediction']))
    prediction0_on_train.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True,False])
    prediction0_on_train.to_csv(sys.argv[3], index=False)


    prediction0_on_test = clf0.predict_proba(X_test)
    prediction0_on_test = pd.DataFrame().append(validate[['srch_id', 'prop_id']],
                                                ignore_index=True).join(pd.DataFrame(prediction0_on_test[:,1],
                                                                                     columns=['prediction']))
    prediction0_on_test.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True, False])
    prediction0_on_test.to_csv(sys.argv[4], index=False)

    print 'Logistic Regression'
    clf1 = LogisticRegression(solver='sag', random_state=4321)
    clf1.fit(X_train, y_train)
    prediction1_on_train = clf1.predict_proba(X_train)
    prediction1_on_train = pd.DataFrame().append(sub_train[['srch_id', 'prop_id']],
                                                 ignore_index=True).join(pd.DataFrame(prediction1_on_train[:, 1],
                                                                                      columns=['prediction']))
    prediction1_on_train.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True, False])
    prediction1_on_train.to_csv(sys.argv[5], index=False)

    prediction1_on_test = clf1.predict_proba(X_test)
    prediction1_on_test = pd.DataFrame().append(validate[['srch_id', 'prop_id']],
                                                ignore_index=True).join(pd.DataFrame(prediction1_on_test[:, 1],
                                                                                     columns=['prediction']))
    prediction1_on_test.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True, False])
    prediction1_on_test.to_csv(sys.argv[6], index=False)

    # print 'Naive Bayes'
    # clf2 = GaussianNB()
    # clf2.fit(X_train, y_train)
    # prediction2_on_train = clf2.predict_proba(X_train)
    # prediction2_on_train = pd.DataFrame().append(sub_train[['srch_id', 'prop_id']],
    #                                              ignore_index=True).join(pd.DataFrame(prediction2_on_train[:, 1],
    #                                                                                   columns=['prediction']))
    # prediction2_on_train.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True, False])
    # prediction2_on_train.to_csv(sys.argv[7], index=False)
    #
    # prediction2_on_test = clf2.predict_proba(X_test)
    # prediction2_on_test = pd.DataFrame().append(validate[['srch_id', 'prop_id']],
    #                                             ignore_index=True).join(pd.DataFrame(prediction2_on_test[:, 1],
    #                                                                                  columns=['prediction']))
    # prediction2_on_test.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True, False])
    # prediction2_on_test.to_csv(sys.argv[8], index=False)

    time = datetime.now() - start
    print 'total processing time: %d seconds' % int(time.seconds)


if __name__ == '__main__':
    main()


