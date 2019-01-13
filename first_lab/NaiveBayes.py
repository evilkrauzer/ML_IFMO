import math


class NaiveBayes:
    summaries = {}

    # обучение классификатора
    def fit(self, train_data, train_classes):
        # получаем словарь с данными, разделенный по классам
        classes_dict = self.separate_by_class(train_data, train_classes)

        for class_name, items in classes_dict.items():
            # считаем среднее значение и среднеквадратичное отклонение атрибутов для каждого класса входных данных
            self.summaries[class_name] = self.summarize(items)

    def separate_by_class(self, train_data, train_classes):
        classes_dict = {}
        for i in range(len(train_data)):
            classes_dict.setdefault(train_classes[i], []).append(train_data[i])

        return classes_dict

    # обобщаем данные
    def summarize(self, class_data):
        summaries = [(self.mean(attributes), self.stand_dev(attributes)) for attributes in
                     zip(*class_data)]
        return summaries

    # вычисление среднего значения
    def mean(self, values):
        return sum(values) / float(len(values))

    # вычисление дисперсии
    def stand_dev(self, values):
        var = sum([pow(x - self.mean(values), 2) for x in values]) / float(len(values) - 1)
        return math.sqrt(var)

    # классификация тестовой выборки
    def predict(self, data_x):
        predictions = []
        for x in data_x:
            predictions.append(self.predict_one(self.summaries, x))
        return predictions

    # классификация одного объекта
    def predict_one(self, summaries, x):
        probabilities = self.calc_class_probabilities(summaries, x)
        best_class = None
        max_prob = -1
        for class_name, probability in probabilities.items():
            if best_class is None or probability > max_prob:
                max_prob = probability
                best_class = class_name
        return best_class

    # вычисление вероятности принадлежности объекта к каждому из классов
    def calc_class_probabilities(self, summaries, instance_attr):
        probabilities = {}
        for class_name, class_summaries in summaries.items():
            probabilities[class_name] = 1.0
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                x = float(instance_attr[i])
                probabilities[class_name] *= self.calc_probability(x, mean, stdev)
        return probabilities

    # вычисление апостериорной вероятности принадлежности объекта к определенному классу
    def calc_probability(self, x, mean, stdev):
        if stdev == 0:
            stdev += 0.000001
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
