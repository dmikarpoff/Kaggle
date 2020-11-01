import csv
import random

class Model:
    def train(self, data, label):
        pass

    def predict(self, data):
        pass


class RandomModel(Model):
    def __init__(self):
        super().__init__()

    def train(self, data, label):
        pass

    def predict(self, data):
        return [{'Survived': random.randint(0, 1),
                 'PassengerId': p['PassengerId']}
                for p in data]

class PerceptronModel(Model):
    def __init__(self):
        super().__init__()

    def train(self, data, label):
        pass


def ignore(s):
    return None


data_transform = {
    'Pclass': int,
    'PassengerId': int,
    'Name': str,
    'Sex': str,
    'Age': int,
    'SibSp': int,
    'Parch': int,
    'Ticket': str,
    'Fare': float,
    'Cabin': str,
    'Embarked': str,
}


def load_data(filename: str):
    ret_train_data = []
    ret_train_label = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        header = []
        for row in reader:
            if row[0].startswith('Pass'):
                header = row
                continue
            data_dict = {}
            for name, value in zip(header, row):
                if name == 'Survived':
                    ret_train_label.append(int(value))
                    continue
                try:
                    parsed_val = data_transform[name](value)
                except ValueError:
                    continue
                if parsed_val is not None:
                    data_dict[name] = parsed_val
            ret_train_data.append(data_dict)
    return ret_train_data, ret_train_label


model = RandomModel()
train_data, train_label = load_data('train.csv')
model.train(train_data, train_label)
test_data, _ = load_data('test.csv')
test_label = model.predict(test_data)

with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    header = ['PassengerId', 'Survived']
    writer.writerow(header)
    for label in test_label:
        row = [label[h] for h in header]
        writer.writerow(row)


