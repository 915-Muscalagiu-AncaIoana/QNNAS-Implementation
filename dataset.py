from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = MinMaxScaler().fit_transform(dataset.data)
        self.labels = dataset.target

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def get_num_features(self):
        return self.features.shape[1]

    def split_dataset(self):
        return train_test_split(self.features, self.labels, train_size=0.8)
