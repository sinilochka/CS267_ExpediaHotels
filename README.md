CS267_ExpediaHotels

ML Pipeline:

Step 1. Sub-sample data from train.csv.

1 experiment: 10% subtrain, 5% validate

2 experiment: 20% subtrain, 10% validate

available here:
https://github.com/sinilochka/CS267_ExpediaHotels/blob/master/generate_features.py

Step 2. Extract target from files obtained in Step 1.

Allows for easy computation in Step 4.

available here:
https://github.com/sinilochka/CS267_ExpediaHotels/blob/master/target_extract.py

Step 3. Learning stage.

Filling missing values with -100.

Produces predictions for subtrain and validate.

3 types of models:

Logistic Regression:
    1) solver = ’sag'
    2) solver = ‘liblinear'
    
Naive Bayes

Random Forest:
    1) n = 50
    2) n = 100
    
available here:
https://github.com/sinilochka/CS267_ExpediaHotels/blob/master/predict.py

Step 4. Evaluation stage.

Computes NDCG score between target file and prediction file.

available here:
https://github.com/sinilochka/CS267_ExpediaHotels/blob/master/compute_metrics.py
