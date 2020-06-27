import argparse
import pandas as pd
import numpy as np
import os
# from os.path import join
import sys
import logging

# import joblib
from sklearn.externals import joblib

from   sklearn.tree import DecisionTreeRegressor
from   sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from   sklearn import metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def make_predictions(model, validate):
    validate_dropped = validate.drop('unit_sales', axis=1).fillna(-1)
    validate_preds   = model.predict(validate_dropped)
    return validate_preds


def eval_nwrmsle(predictions, targets, weights):
    if type(predictions) == list:
        predictions = np.array([np.nan if x < 0 else x for x in predictions])
    elif type(predictions) == pd.Series:
        predictions[predictions < 0] = np.nan
    targetsf = targets.astype(float)
    targetsf[targets < 0] = np.nan
    weights = 1 + 0.25 * weights
    log_square_errors = (np.log(predictions + 1) - np.log(targetsf + 1)) ** 2
    return(np.sqrt(np.sum(weights * log_square_errors) / np.sum(weights)))


if 'SAGEMAKER_METRICS_DIRECTORY' in os.environ:
    log_file_handler = logging.FileHandler(os.path.join(os.environ['SAGEMAKER_METRICS_DIRECTORY'],
                                                "metrics.json"))
    log_file_handler.setFormatter(
    "{'time':'%(asctime)s', 'name': '%(name)s', \
    'level': '%(levelname)s', 'message': '%(message)s'}"
    )
    logger.addHandler(log_file_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--model_name',   type=str,   default='decision_tree')
    parser.add_argument('--n_estimators', type=int,   default=10)
    parser.add_argument('--max_features', type=float, default=0.5)
    parser.add_argument('--max_depth',    type=int,   default=4)
    parser.add_argument('--criterion',    type=str,   default='mse')

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()

    logger.info("Get train data loader")
    
    train_data      = pd.read_csv('{}/final_train.csv'.format(args.train), engine="python")

    logger.info("Get valdation data loader")

    validation_data = pd.read_csv('{}/final_validate.csv'.format(args.validation), engine="python")

    train_x = train_data.drop('unit_sales', axis=1)
    train_y = train_data['unit_sales']

    model_name   = args.model_name
    n_estimators = args.n_estimators
    max_features = args.max_features
    max_depth    = args.max_depth
    criterion    = args.criterion

    if (model_name == 'random_forest'):
        clf = RandomForestRegressor(random_state=None, n_estimators=n_estimators, max_features=max_features)
    elif (model_name == 'adaboost'):
        clf = AdaBoostRegressor(random_state=None, n_estimators=n_estimators)
    elif (model_name == 'gradient_boosting'):
        clf = GradientBoostingRegressor(random_state=None, n_estimators=n_estimators, max_depth=max_depth)
    elif (model_name == 'decision_tree'):
        clf = DecisionTreeRegressor(random_state=None, criterion=criterion)
    else:
        logger.debug("Invalid model name")
        
    logger.debug("Training starts")
    
    clf = clf.fit(train_x, train_y)

    logger.debug("Training done")

    logger.debug("Making prediction on validation data")

    validation_predictions = make_predictions(clf, validation_data)

    logger.info('nwrmsle: {:.4f};\n'.format(eval_nwrmsle(validation_predictions,
                                                         validation_data['unit_sales'].values, 
                                                         validation_data['perishable'].values)))
    logger.info('r2_score: {:.4f};\n'.format(metrics.r2_score(y_true=validation_data['unit_sales'].values, 
                                                              y_pred=validation_predictions)))

    
def model_fn(model_dir):
    # Deserialized and return fitted model
    # Note that this should have the same name as the serialized model in the main method
    
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return clf
