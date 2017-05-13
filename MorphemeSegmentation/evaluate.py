import morph_io as mio
from context import EvaluationContext
from option import EvaluateOption


class Metric:
    """
    Container for holding values necessary to calculate metrics
    """
    def __init__(self, tp=0, tn=0, fp=0, fn=0, index=1):
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


def ensure_validity(ctx: EvaluationContext):
    """
    Checks the two lists of [['B', 'M', ...], [...], ...]
    :param ctx: contains expected-actual lists
    """
    if len(ctx.expected) > len(ctx.actual):
        raise Exception('List of expected words is longer')

    if len(ctx.expected) < len(ctx.actual):
        raise Exception('List of predictions is longer')

    for i in range(0, len(ctx.expected)):
        if len(ctx.expected[i]) != len(ctx.actual[i]):
            raise Exception(str(i + 1) + 'th words do not match in length')


def generate_metrics(ctx: EvaluationContext):
    """
    Generates Metric object by comparing the expected segmentation
    with the the actual (works with lists of [['B', 'M', ...], [...], ...])
    :param ctx: contains expected-actual lists
    :return: the list of relevant Metric-s
    """
    metrics = []
    for word_index in range(0, len(ctx.expected)):
        tn, tp, fn, fp = (0, 0, 0, 0)
        # if length is 1, it really isn't interesting to us
        if len(ctx.expected[word_index]) == 1:
            continue

        for sym_index in range(1, len(ctx.expected[word_index])):
            current_exp = ctx.expected[word_index][sym_index]
            current_pred = ctx.actual[word_index][sym_index]

            if current_exp in ctx.POS and current_pred in ctx.POS:
                tp += 1
            elif current_exp in ctx.POS and current_pred in ctx.NEG:
                fn += 1
            elif current_exp in ctx.NEG and current_pred in ctx.POS:
                fp += 1
            else:
                tn += 1

        metrics.append(Metric(tp, tn, fp, fn, word_index + 1))

    return metrics


def get_average_metrics(metrics: list):
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


def get_aggregated_metric(metrics: list):
    """
    :param metrics: [Metric, Metric, ...]
    :return: aggregated Metric
    """
    aggregated_metric = Metric()
    for metric in metrics:
        aggregated_metric.false_negative += metric.false_negative
        aggregated_metric.false_positive += metric.false_positive
        aggregated_metric.true_positive += metric.true_positive
        aggregated_metric.true_negative += metric.true_negative
    return aggregated_metric


def main():
    option = EvaluateOption()
    context = EvaluationContext()

    context.expected = [segment.seg_mapping
                        for segment in mio.read_goldstd(option.expected_file, context)]
    context.actual = [segment.seg_mapping
                      for segment in mio.read_goldstd(option.actual_file, context)]

    ensure_validity(context)
    metrics = generate_metrics(context)

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
        print(str(metric.index) + '. word -->' +
              ' F-Score=' + str(metric.get_fscore()) +
              ' Precision=' + str(metric.get_precision()) +
              ' Recall=' + str(metric.get_recall()))


if __name__ == "__main__":
    main()
