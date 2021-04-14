

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import logging
import pandas as pd
from collections import OrderedDict, Counter
import json
import os




CLASSIFICATION_RM = ['accuracy', 'macro avg', 'weighted avg']

COLUMN_MAP = OrderedDict([('support','NT'), ('precision','P'), ('recall','R'), ('f1-score','F1')])


'''
https://stackoverflow.com/questions/38151615/specific-cross-validation-with-random-forest
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

'''

class ModelARDS:


    def __init__(self, \
        model_type,
        hyperparams,

        # Sklearn parameters
        ):

        random_forest_defaults = OrderedDict()
        random_forest_defaults['n_estimators'] = 200
        random_forest_defaults['max_depth'] = None
        random_forest_defaults['min_samples_split'] = 2
        random_forest_defaults['min_samples_leaf'] = 1
        random_forest_defaults['min_weight_fraction_leaf'] = 0.0
        random_forest_defaults['max_features'] = 'auto'
        random_forest_defaults['max_leaf_nodes'] = None
        random_forest_defaults['random_state'] = None
        random_forest_defaults['ccp_alpha'] = 0.0
        random_forest_defaults['n_jobs'] = 1

        svm_defaults = OrderedDict()
        svm_defaults['C'] = 1.0
        svm_defaults['kernel'] = 'rbf'
        svm_defaults['degree'] = 3
        svm_defaults['gamma'] = 'scale'
        svm_defaults['coef0'] = 0.0
        svm_defaults['shrinking'] = True
        svm_defaults['probability'] = False
        svm_defaults['tol'] = 0.001
        svm_defaults['cache_size'] = 200
        svm_defaults['class_weight'] = None
        svm_defaults['verbose'] = False
        svm_defaults['max_iter'] = - 1,
        svm_defaults['decision_function_shape'] = 'ovr'
        svm_defaults['break_ties'] = False
        svm_defaults['random_state'] = None



        # Sklearn parameters
        self.hyperparams = hyperparams

        #self.hyperparams['class_weight'] = class_weight

        if model_type == 'random_forest':
            self.classifier_class = RandomForestClassifier
        elif model_type == "svm":
            self.classifier_class = SVC
        else:
            raise ValueError(f"Invalid model type: {model_type}")


        self.estimator = None

    def fit_cv(self, X, y, tuned_parameters, path=None, n_splits=3, scoring='f1_micro'):


        estimator = self.classifier_class(**self.hyperparams)

        classifier = GridSearchCV( \
                        estimator = estimator,
                        param_grid = tuned_parameters,
                        scoring = ['precision_micro', 'recall_micro', 'f1_micro'],
                        refit = scoring,
                        cv = n_splits,
                        verbose = 2)


        #print(tuned_parameters)
        #print(classifier)
        logging.info(f"Tuned parameters: {tuned_parameters}")

        classifier.fit(X, y)



        Ps = classifier.cv_results_['mean_test_precision_micro']
        Rs = classifier.cv_results_['mean_test_recall_micro']
        F1s = classifier.cv_results_['mean_test_f1_micro']


        D = []
        for P, R, F1, params in zip(Ps, Rs, F1s, classifier.cv_results_['params']):
            d = OrderedDict()
            d['P'] = P
            d['R'] = R
            d['F1'] = F1

            for k, v in params.items():
                d[k] = v

            D.append(d)

        df = pd.DataFrame(D)
        logging.info(f"CV results:\n{df}")
        logging.info("")

        best_params = classifier.best_params_
        logging.info(f"Best parameters: {best_params}")
        logging.info("")

        if path is not None:
            f = os.path.join(path, "scores_cv.csv")
            df.to_csv(f)

            f = os.path.join(path, "best_params.json")
            json.dump(best_params, open(f, 'w'))

        return (best_params, df)

    def fit(self, X, y, path=None):
        self.estimator = self.classifier_class(**self.hyperparams)
        self.estimator.fit(X, y)


    def predict(self, X, path=None):
        assert self.estimator is not None
        y_pred = self.estimator.predict(X)
        return y_pred


    def score(self, X, y, path=None, col_name='subtype'):
        y_pred = self.predict(X)

        report = classification_report(y, y_pred, output_dict=True)



        df = pd.DataFrame(report)
        for c in CLASSIFICATION_RM:
            if c in df:
                del df[c]

        df = df.transpose()
        column_map = COLUMN_MAP
        for orig, new in column_map.items():
            if orig in df:
                df[new] = df[orig]
                del df[orig]

        df = df.rename_axis(col_name).reset_index()
        df['TP'] = df['R']*df['NT']
        df['NP'] = df['TP']/df['P']

        df = df[['subtype', 'NT', 'NP', 'TP', 'P', 'R', 'F1']]

        #df['subtype'] = df.index



        if path is not None:
            f = os.path.join(path, "scores_pred.csv")
            df.to_csv(f)

        return (y_pred, df)
