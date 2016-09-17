import sys
import math
import pandas as pd
from datetime import datetime
import numpy as np


def calc_dcg(results):
    dcg = 0.0
    for index, r in enumerate(results):
        dcg += (2 ** r['relevance'] - 1) / math.log(index + 2, 2)
    return dcg


def calc_ndcg(search_results):
    # search_results - dict prop_id => {'prediction' => float, 'relevance' => float}
    ordered_results = sorted(search_results.values(), key=lambda x: -x['prediction'])
    ordered_results = ordered_results[:38]
    ideal_results = sorted(search_results.values(), key=lambda x: -x['relevance'])[0:38]
    idcg = calc_dcg(ideal_results)
    if idcg == 0:
        return None
    return calc_dcg(ordered_results) / idcg


def main():
    start = datetime.now()
    print start

    path_real = sys.argv[1]
    path_predicted = sys.argv[2]

    samples = dict()
    for line in open(path_predicted):
        try:
            s = line.strip().split(',')
            srch_id_pred = int(s[0])
            prop_id_pred = int(s[1])
            prediction = float(s[2])
            if srch_id_pred not in samples:
                samples[srch_id_pred] = dict()

            samples[srch_id_pred][prop_id_pred] = {
                'prediction': prediction,
            }
        except:
            pass

    for line in open(path_real):
        try:
            s = line.strip().split(',')
            srch_id = int(s[0])
            prop_id = int(s[1])
            click_bool = bool(int(s[2]))
            booking_bool = bool(int(s[4]))
            if srch_id in samples and prop_id in samples[srch_id]:
                samples[srch_id][prop_id]['relevance'] = 0.0
                if click_bool:
                    samples[srch_id][prop_id]['relevance'] = 1.0
                if booking_bool:
                    samples[srch_id][prop_id]['relevance'] = 5.0
        except:
            pass

    ndcgs = []
    for search_id, search_results in samples.iteritems():
        ndcg = calc_ndcg(search_results)
        if ndcg is not None:
            ndcgs.append(ndcg)
    final_ndcg = 1.0 * sum(ndcgs) / len(ndcgs)

    print "Evaluation Stage"
    print "NDCG score:", final_ndcg

    time = datetime.now() - start
    print 'total processing time: %d seconds' % int(time.seconds)


if __name__ == '__main__':
    main()