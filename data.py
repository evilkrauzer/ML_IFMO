from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, data, classes, size, train_size, attributes_count = 561):
        size = int(data.shape[0] * size)
        self.data = data[0:size, :attributes_count]
        self.classes = classes[0:size]
        self.train_data, self.test_data, self.train_classes, self.test_classes = train_test_split(self.data, self.classes,
                                                                                                  train_size=train_size,
                                                                                                  test_size=1 - train_size)
