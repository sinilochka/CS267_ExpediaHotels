import sys
import pandas as pd


def main():

    filepath = pd.read_csv(sys.argv[1])
    target = filepath[['srch_id', 'prop_id', 'click_bool', 'gross_bookings_usd', 'booking_bool']]
    target.to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()