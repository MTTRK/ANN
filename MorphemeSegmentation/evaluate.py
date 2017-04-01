"""
Expected format for the 2 files being evaluated:
...
START
B
E
B
M
E
B
E
B
M
E
S
...
STOP
"""
import sys
import morph_io as mio


POS = [mio.BEGIN, mio.SINGLE]
NEG = [mio.MIDDLE, mio.END]


class Metric:
    """
    Represents the metrics related to an id (index)
    """

    def __init__(self, tp=0, tn=0, fp=0, fn=0, index=0):
        self.true_negative = tn
        self.true_positive = tp
        self.false_negative = fn
        self.false_positive = fp
        self.index = index

    def get_precision(self):
        return self.true_positive / (self.true_positive + self.false_positive + 0.00001)

    def get_recall(self):
        return self.true_positive / (self.true_positive + self.false_negative + 0.00001)

    def get_fscore(self):
        return 2 * (self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall() + 0.00001)


def read_symbols(filepath: str):
    """
    :param filepath: input file
    :return: list of symbol blocks (word segmentations) [['B', ...], [...]]
    """

    blocks = []
    current = []
    for symbol in mio.read_file(filepath):
        sym = symbol.strip(' \n\t')
        if sym == mio.START:
            if current:
                blocks.append(current)
                current = []
        elif sym != mio.STOP:
            current.append(sym)

    # flush the last block
    if current:
        blocks.append(current)

    return blocks


def ensure_validity(expectations: list, predictions: list):
    """
    :param expectations: [['B', ...], [...], ...]
    :param predictions: [['B', ...], [...], ...]
    """

    if len(expectations) > len(predictions):
        raise Exception('List of expected words is longer')

    if len(predictions) > len(expectations):
        raise Exception('List of predictions is longer')

    for i in range(0, len(expectations)):
        if len(expectations[i]) != len(predictions[i]):
            raise Exception(str(i) + 'th word does not match in length')


def generate_metrics(expectations: list, predictions: list):
    """
    :param expectations: [['B', ...], [...], ...]
    :param predictions: [['B', ...], [...], ...]
    :return: the list of relevant metrics
    """

    metrics = []

    for word_index in range(0, len(expectations)):
        tn, tp, fn, fp = (0, 0, 0, 0)

        for sym_index in range(0, len(expectations[word_index])):
            current_exp = expectations[word_index][sym_index]
            current_pred = predictions[word_index][sym_index]

            if current_exp in POS and current_pred in POS:
                tp += 1
            elif current_exp in POS and current_pred in NEG:
                fn += 1
            elif current_exp in NEG and current_pred in POS:
                fp += 1
            else:
                tn += 1

        metrics.append(Metric(tp, tn, fp, fn, word_index))

    return metrics


def get_average_metrics(metrics):
    """
    :param metrics: [Metric, Metric, ...]
    :return: average f-score, precision, recall
    """

    avg_fscore = 0.0
    avg_precision = 0.0
    avg_recall = 0.0

    for metric in metrics:
        avg_fscore += metric.get_fscore()
        avg_precision += metric.get_precision()
        avg_recall += metric.get_recall()

        avg_fscore /= len(metrics)
        avg_precision /= len(metrics)
        avg_recall /= len(metrics)

    return avg_fscore, avg_precision, avg_recall


def get_aggregated_metric(metrics):
    """
    :param metrics: [Metric, Metric, ...]
    :return: aggregated metric
    """

    aggregated_metric = Metric()
    for metric in metrics:
        aggregated_metric.false_negative += metric.false_negative
        aggregated_metric.false_positive += metric.false_positive
        aggregated_metric.true_positive += metric.true_positive
        aggregated_metric.true_negative += metric.true_negative
    return aggregated_metric


def main():
    if len(sys.argv[1:]) != 2:
        raise Exception('Script needs 2 input-parameters (expected, predictions)')

    expectations = read_symbols(sys.argv[1:][0])
    predictions = read_symbols(sys.argv[1:][1])

    ensure_validity(expectations, predictions)
    metrics = generate_metrics(expectations, predictions)

    aggregated_metric = get_aggregated_metric(metrics)
    avg_fscore, avg_precision, avg_recall = get_average_metrics(metrics)

    print('Average F-Score=' + str(avg_fscore) +
            ' Precision=' + str(avg_precision) +
            ' Recall=' + str(avg_recall))

    print('Aggregated F-Score=' + str(aggregated_metric.get_fscore()) +
            ' Precision=' + str(aggregated_metric.get_precision()) +
            ' Recall=' + str(aggregated_metric.get_recall()))

    print('\nPer word:')
    for metric in metrics:
        print(str(metric.index) + 'th word -->' +
              ' F-Score=' + str(metric.get_fscore()) +
              ' Precision=' + str(metric.get_precision()) +
              ' Recall=' + str(metric.get_recall()))


if __name__ == "__main__":
    main()
