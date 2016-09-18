import pandas as pd
import numpy as np
import random
import sys

random.seed(4321)


def main():

    data = pd.read_csv(sys.argv[1])
    data['prediction'] = np.random.uniform(size=len(data.index))
    data.sort_values(by=['srch_id', 'prediction'], inplace=True, ascending=[True, False])
    data.to_csv(sys.argv[2], index=False)


if __name__ == '__main__':
    main()
