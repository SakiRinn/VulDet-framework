import numpy as np
import sklearn.metrics as metrics


class Metrics:
    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        self.scores = scores.squeeze()
        self.labels = labels.squeeze()

        if scores.ndim > 1:
            self.scores = scores[:, 0]
        self.preds = (self.scores > 0.5).astype(np.float32)

    def __str__(self):
        confusion = metrics.confusion_matrix(y_true=self.labels, y_pred=self.preds)
        tn, fp, fn, tp = confusion.ravel()
        string = f"\nConfusion matrix: \n"
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
                    "PR-AUC": metrics.average_precision_score(y_true=self.labels, y_score=self.scores),
                    "ROC-AUC": metrics.roc_auc_score(y_true=self.labels, y_score=self.scores),
                    "MCC": metrics.matthews_corrcoef(y_true=self.labels, y_pred=self.preds),
                    "Error": self.error()}
        return _metrics

    def error(self):
        errors = np.where(self.scores > 0, 100 * (abs(self.scores - self.preds) / self.scores), 1)
        return errors.sum() / errors.shape[0]
