import numpy as np
import pandas as pd


def translate_proba(results):
    return [np.amax(x, 1) for x in results]


class Predictor:
    def __init__(self, labels):
        self.labels = labels
        self.inverse_labels = [self.labels['labels1_inverse'], self.labels['labels2_inverse'],
                               self.labels['labels3_inverse'], self.labels['labels4_inverse'],
                               self.labels['labels5_inverse']]

        self.labels1_index = {}
        self.labels2_index = {}
        self.labels3_index = {}
        self.labels4_index = {}
        self.labels5_index = {}

        self.create_path(labels)

    def create_path(self, labels):
        self.labels1_index = np.array(list(labels['labels1'].values()))
        self.labels2_index = {}
        self.labels3_index = {}
        self.labels4_index = {}
        self.labels5_index = {}

        for key, value in labels['labels2'].items():
            l1, l2 = key.split("-")
            try:
                self.labels2_index[labels['labels1'][l1]].append(value)
            except:
                self.labels2_index[labels['labels1'][l1]] = [value]

        for key, value in labels['labels3'].items():
            l1, l2, l3 = key.split("-")
            try:
                self.labels3_index[labels['labels2']["-".join([l1, l2])]].append(value)
            except:
                self.labels3_index[labels['labels2']["-".join([l1, l2])]] = [value]

        for key, value in labels['labels4'].items():
            l1, l2, l3, l4 = key.split("-")
            try:
                self.labels4_index[labels['labels3']["-".join([l1, l2, l3])]].append(value)
            except:
                self.labels4_index[labels['labels3']["-".join([l1, l2, l3])]] = [value]

        for key, value in labels['labels5'].items():
            l1, l2, l3, l4, l5 = key.split("-")
            try:
                self.labels5_index[labels['labels4']["-".join([l1, l2, l3, l4])]].append(value)
            except:
                self.labels5_index[labels['labels4']["-".join([l1, l2, l3, l4])]] = [value]

        for k, value in self.labels2_index.items():
            self.labels2_index[k] = np.array(value)

        for k, value in self.labels3_index.items():
            self.labels3_index[k] = np.array(value)

        for k, value in self.labels4_index.items():
            self.labels4_index[k] = np.array(value)

        for k, value in self.labels5_index.items():
            self.labels5_index[k] = np.array(value)

    def create_full_name(self, df):
        full_name = []

        for cat5 in df.classe5:
            values = cat5.split("-")

            cat1 = values[0]
            cat2 = "-".join(values[:2])
            cat3 = "-".join(values[:3])
            cat4 = "-".join(values[:4])

            name = " -> ".join([
                self.labels['label1_name'][cat1],
                self.labels['label2_name'][cat2],
                self.labels['label3_name'][cat3],
                self.labels['label4_name'][cat4],
                self.labels['label5_name'][cat5]
            ])

            full_name.append(name)

        return full_name

    def normalize(self, predictions):
        r1, r2, r3, r4, r5 = [np.array(x, copy=True) for x in predictions]
        rr1 = r1.argmax(axis=1)

        for i, rr in enumerate(rr1):
            filtro = np.ones(r2[i].shape, dtype=bool)
            filtro[self.labels2_index[rr]] = False
            r2[i][filtro] = 0

        rr2 = r2.argmax(1)

        for i, rr in enumerate(rr2):
            filtro = np.ones(r3[i].shape, dtype=bool)
            filtro[self.labels3_index[rr]] = False
            r3[i][filtro] = 0

        rr3 = r3.argmax(1)

        for i, rr in enumerate(rr3):
            filtro = np.ones(r4[i].shape, dtype=bool)
            filtro[self.labels4_index[rr]] = False
            r4[i][filtro] = 0

        rr4 = r4.argmax(1)

        for i, rr in enumerate(rr4):
            filtro = np.ones(r5[i].shape, dtype=bool)
            filtro[self.labels5_index[rr]] = False
            r5[i][filtro] = 0


        return r1, r2, r3, r4, r5

    def translate_predictions(self, predictions):
        results = []

        for index, y_pred in enumerate(predictions):
            results.append([self.inverse_labels[index][label_index] for label_index in y_pred.argmax(1)])

        return results

    def predict_as_df(self, predictions):
        results = self.normalize(predictions)
        cat1, cat2, cat3, cat4, cat5 = self.translate_predictions(results)
        proba1, proba2, proba3, proba4, proba5 = translate_proba(results)

        df = pd.DataFrame([])
        df['classe1'] = cat1
        df['classe2'] = cat2
        df['classe3'] = cat3
        df['classe4'] = cat4
        df['classe5'] = cat5
        df['proba1'] = proba1
        df['proba2'] = proba2
        df['proba3'] = proba3
        df['proba4'] = proba4
        df['proba5'] = proba5
        df['full_name'] = self.create_full_name(df)

        return df