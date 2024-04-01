import sys

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


class MachineLearningModel(BaseEstimator):
    def __init__(self, model_type, balance=False, verbose=True):
        super(MachineLearningModel, self).__init__()
        self.model_type = model_type
        self.balance = balance
        self.output_buffer = sys.stdout if verbose else None

    def fit(self, train_x, train_y):
        self.train(train_x, train_y)

    def train(self, train_x, train_y):
        # import warnings
        # warnings.filterwarnings('ignore')

        if self.model_type == 'SVM':
            self.model = SVC()
        elif self.model_type == 'LR':
            self.model = LogisticRegression()
            # MLPClassifier(hidden_layer_sizes=(256, 128, 256), max_iter=10)
        elif self.model_type == 'RF':
            self.model = RandomForestClassifier()
        else:
            raise ValueError('`model_type` must be SVM, LR (Logistic Regression) or RF (Random Forest)!')

        if self.balance:
            full_x, full_y = self.rebalance(train_x, train_y)
        else:
            full_x, full_y = train_x, train_y

        if self.output_buffer is not None:
            print('Fitting ' + self.model_type + ' model...', file=self.output_buffer)
        self.model.fit(full_x, full_y)
        if self.output_buffer is not None:
            print('Training Complete!', file=self.output_buffer)

    def predict(self, test_x):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        return self.model.predict(test_x)

    def predict_proba(self, test_x):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        return self.model.predict_proba(test_x)

    def evaluate(self, text_x, test_y):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        predictions = self.predict(text_x)
        return {
            'accuracy': accuracy_score(test_y, predictions) * 100,
            'precision': precision_score(test_y, predictions) * 100,
            'recall': recall_score(test_y, predictions) * 100,
            'f1': f1_score(test_y, predictions) * 100,
        }

    def score(self, text_x, test_y):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        scores = self.evaluate(text_x, test_y)
        return scores['f1']

    def rebalance(self, x, y):
        smote = SMOTE(random_state=1000)
        return smote.fit_resample(x, y)
