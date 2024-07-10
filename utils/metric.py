import numpy as np
import sklearn.metrics as metrics


class Metric:
    def __init__(self, probs: 'np.ndarray', labels: 'np.ndarray'):
        self.probs = probs.squeeze()
        self.labels = labels.squeeze()

        if probs.ndim > 2:
            self.probs = probs.squeeze()
        self.preds = (self.probs.argmax(dim=1)).astype(np.float32)

    def __str__(self):
        confusion = metrics.confusion_matrix(y_true=self.labels, y_pred=self.preds)
        tn, fp, fn, tp = confusion.ravel()
        string = "\nConfusion matrix: \n"
        string += f"{confusion}\n"
        string += f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        string += '\n'.join([name + ": " + str(metric)
                            for name, metric in self().items()])
        return string

    def __call__(self):
        _metrics = {"Accuracy": metrics.accuracy_score(y_true=self.labels, y_pred=self.preds),
                    "Precision": metrics.precision_score(y_true=self.labels, y_pred=self.preds),
                    "Recall": metrics.recall_score(y_true=self.labels, y_pred=self.preds),
                    "F1-score": metrics.f1_score(y_true=self.labels, y_pred=self.preds),
                    "PR-AUC": metrics.average_precision_score(y_true=self.labels, y_score=self.probs),
                    "ROC-AUC": metrics.roc_auc_score(y_true=self.labels, y_score=self.probs),
                    "MCC": metrics.matthews_corrcoef(y_true=self.labels, y_pred=self.preds),
                    "Error": self.error()}
        return _metrics

    def error(self):
        errors = np.where(self.probs > 0,
                          100 * (abs(self.probs - self.preds) / self.probs),
                          1)
        return errors.sum() / errors.shape[0]
