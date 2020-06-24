import argparse
import pandas as pd
import os

from   sklearn.externals import joblib
from   sklearn.tree import DecisionTreeRegressor
from   sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from   sklearn import metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--criterion', type=str, default='mse')

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    train_data = pd.read_csv('{}/final_train.csv'.format(args.train), engine="python")

    train_x = train_data.drop('unit_sales', axis=1)
    train_y = train_data['unit_sales']

    # Here we support a single hyperparameter. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    criterion = args.criterion
    
    # Now use scikit-learn's decision tree Regressor to train the model.

    clf = DecisionTreeRegressor(random_state=None, criterion=criterion)

    clf = clf.fit(train_x, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    # Deserialized and return fitted model
    # Note that this should have the same name as the serialized model in the main method
    
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return clf
