import csv
import json
import math
import os
import pandas as pd
import numpy as np
import sys, traceback
import tensorflow as tf

from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_fscore_support

from keras.callbacks import Callback

from sabotage.model.json_encoder import NpEncoder
# from b2w.black_magic.minos.helpers import os.path.join
from sabotage.model.prediction import Predictor


class ValidateCallback(Callback):

    def __init__(self, model, labels, x_test, y_true, train_path):
        self.model = model
        self.best_loss = float("inf")
        self.labels = labels
        self.inverse_labels = (labels['labels1_inverse'], labels['labels2_inverse'],
                               labels['labels3_inverse'], labels['labels4_inverse'],
                               labels['labels5_inverse'])
        self.predictor = Predictor(labels)
        self.x_test = x_test
        self.y_true = y_true
        self.train_path = train_path
        self.__save_model_structure__()

    def __save_model_structure__(self):
        with open(os.path.join(self.train_path, "model.json"), "w+") as f:
            f.write(self.model.to_json())

    def on_train_begin(self, logs={}):

        try:
            with open(os.path.join(self.train_path, "metrics.json"), "r") as f:
                metrics_data = json.load(f)

            self.loss = metrics_data['loss']
            self.val_loss = metrics_data['val_loss']
            self.reports = {}
            self.report_metrics = {}
            self.fscore = metrics_data['fscore']
            self.thresholds = [None for _ in range(len(self.inverse_labels))]

        except:

            self.loss = []
            self.val_loss = []
            self.reports = {}
            self.report_metrics = {}
            self.fscore = [[] for _ in range(len(self.inverse_labels))]
            self.thresholds = [None for _ in range(len(self.inverse_labels))]

    @staticmethod
    def validate_level(results, y_true):
        y_pred = results.argmax(1)

        return f1_score(y_true, y_pred, average='weighted')

    def create_logs(self, logs):
        self.loss.append(logs.get('loss'))

        self.val_loss.append(logs.get('val_loss'))

        with open(os.path.join(self.train_path, "metrics.json"), "w+") as f:
            f.write(json.dumps({
                'loss': self.loss,
                'val_loss': self.val_loss,
                'fscore': self.fscore,
            }, cls=NpEncoder))

        print("Create Logs")

    def save_last_model(self):
        self.model.save_weights("/tmp/last-weights.h5")

        with open(os.path.join(self.train_path, "last-weights.h5"), "w+") as f:
            with open("/tmp/last-weights.h5", "rb") as ff:
                f.write(ff.read())
        print(os.path.join(self.train_path, "last-weights.h5"))

        for i, df in self.reports.items():
            with open(os.path.join(self.train_path, "last-level-{}-results.csv".format(i)), "w+") as f:
                df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
            print(os.path.join(self.train_path, "last-level-{}-results.csv".format(i)))

        try:
            for i, df in self.report_metrics.items():
                with open(os.path.join(self.train_path, "last-level-{}-report_metrics.csv".format(i)), "w+") as f:
                    df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                print(os.path.join(self.train_path, "last-level-{}-report_metrics.csv".format(i)))
        except:
            print("last-level-{}-report_metrics.csv")
            traceback.print_exc(file=sys.stdout)

        print("Save Last Model {}".format(os.path.join(self.train_path, "last-weights.h5")))

    def save_best_model(self, logs):
        if logs.get('val_loss') <= self.best_loss:
            self.best_loss = logs.get('val_loss')
            self.model.save_weights("/tmp/best-weights.h5")

            with open(os.path.join(self.train_path, "best-weights.h5"), "w+") as f:
                with open("/tmp/best-weights.h5", "rb") as ff:
                    f.write(ff.read())
            print(os.path.join(self.train_path, "best-weights.h5"))

            for i, df in self.reports.items():
                with open(os.path.join(self.train_path, "best-level-{}-results.csv".format(i)), "w+") as f:
                    df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                print(os.path.join(self.train_path, "best-level-{}-results.csv".format(i)))

            try:
                for i, df in self.report_metrics.items():
                    with open(os.path.join(self.train_path, "best-level-{}-report_metrics.csv".format(i)), "w+") as f:
                        df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                    print(os.path.join(self.train_path, "best-level-{}-report_metrics.csv".format(i)))
            except:
                print("best-level-{}-report_metrics.csv")
                traceback.print_exc(file=sys.stdout)

            print("Save Best Model {}".format(os.path.join(self.train_path, "best-weights.h5")))
        else:
            print("skip save best model")

    @staticmethod
    def create_report(labels, results, y_true):
        df = pd.DataFrame([])
        df['y_true'] = [labels[i] for i in y_true]
        df['y_pred'] = [labels[i] for i in results.argmax(1)]
        df['proba'] = np.amax(results, 1)
        return df

    @staticmethod
    def create_report_metrics(df):
        try:
            labels = list(set(df.y_true))
            labels.sort()

            precision, recall, f1score, support = precision_recall_fscore_support(
                y_true=df.y_true,
                y_pred=df.y_pred,
                labels=labels,
                zero_division=0
            )

            metric_df = pd.DataFrame([])
            metric_df['cat'] = labels
            metric_df['precision'] = precision
            metric_df['recall'] = recall
            metric_df['f1_score'] = f1score
            metric_df['support'] = support

            return metric_df
        except:
            print("create_report_metrics")
            traceback.print_exc(file=sys.stdout)

    def create_reports(self, results):
        for i, labels in enumerate(self.inverse_labels):
            self.fscore[i].append(self.validate_level(results[i], self.y_true[i]))

            self.reports[i] = self.create_report(labels, results[i], self.y_true[i])

            self.report_metrics[i] = self.create_report_metrics(self.reports[i])

    @staticmethod
    def create_one_hot_encode(n_classes, y_true):
        zeros = np.zeros((len(y_true), n_classes))

        for i, j in enumerate(y_true):
            zeros[i][j] = 1

        return zeros

    def calculate_threshold(self, result, y_true, labels):
        n_classes = len(labels)
        y_test_true = self.create_one_hot_encode(n_classes, y_true)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        best_thresholds = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_true[:, i], result[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i in range(n_classes):
            tf = tpr[i] - (1 - fpr[i])
            best_thresholds[labels[i]] = 1 - thresholds[i][np.absolute(tf).argsort()[0]]

        return best_thresholds

    def create_best_thresholds(self, results):
        try:
            for i, labels in enumerate(self.inverse_labels):
                self.thresholds[i] = self.calculate_threshold(results[i], self.y_true[i], labels)

            with open(os.path.join(self.train_path, "thresholds.json"), "w+") as f:
                f.write(json.dumps(self.thresholds, cls=NpEncoder))

            print(os.path.join(self.train_path, "thresholds.json"))
        except:
            print("create_best_thresholds")
            traceback.print_exc(file=sys.stdout)

    def on_epoch_end(self, epoch, logs={}):
        results = self.model.predict(self.x_test, batch_size=1024)
        results = self.predictor.normalize(results)

        self.create_reports(results)
        self.create_best_thresholds(results)
        self.save_last_model()
        self.save_best_model(logs)
        self.create_logs(logs)


class BackupAndRestoreCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model, path, best_loss=None):
        self.path = path
        self.current_model = model
        self.best_loss = best_loss or math.inf
        self.best_path = os.path.join(self.path, "best", "weights")
        self.last_path = os.path.join(self.path, "last", "weights")
        self.metadata_path = os.path.join(self.path, "metadata.json")
        super(BackupAndRestoreCheckpoint, self).__init__()

    def build_history_path(self, epoch):
        return os.path.join(self.path, "history", str(epoch), "weights")

    def restore(self):
        print("BackupAndRestoreCheckpoint", "checking metadata_path", self.metadata_path)
        if tf.io.gfile.exists(self.metadata_path):
            print("BackupAndRestoreCheckpoint", "restoring model")
            with tf.io.gfile.GFile(self.metadata_path, "r") as f:
                metadata = json.loads(f.read())
                print("reading", self.metadata_path, metadata)

            print("loading weights", self.build_history_path(metadata['epoch']))
            self.current_model.load_weights(self.build_history_path(metadata['epoch']))

            return metadata['epoch'] + 1
        else:
            return 0

    def on_epoch_end(self, epoch, logs=None):
        print("saving last weights", self.last_path)
        self.model.save_weights(self.last_path, save_format='tf')

        history_weights_path = self.build_history_path(epoch)
        print("saving history weights", history_weights_path)
        self.model.save_weights(history_weights_path, save_format='tf')

        if logs is not None:
            if logs['val_loss'] <= self.best_loss:
                print("saving best weights", self.best_path)
                self.best_loss = logs['val_loss']
                self.model.save_weights(self.best_path, save_format='tf')

        metadata = {'epoch': int(epoch)}

        with tf.io.gfile.GFile(self.metadata_path, "w+") as f:
            f.write(json.dumps(metadata))

        print("BackupMetadataCallback", self.metadata_path, metadata)
