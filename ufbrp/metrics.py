import torch
from torch.utils.tensorboard import SummaryWriter

# from skimage.metrics import structural_similarity
from torchmetrics import Accuracy, F1Score, Precision, Recall


class MetricPerformance:
    idx_col = 0

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.__accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.__precision = Precision(task="multiclass", num_classes=self.num_classes)
        self.__recall = Recall(task="multiclass", num_classes=self.num_classes)
        self.__f1score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.reset()

    def reset(self):
        self.targs = []
        self.preds = []

    # only compute on image score
    def update(self, preds, targs):
        self.preds.extend([t.item() for t in preds])
        self.targs.extend([t.item() for t in targs])

    def _compute(self, func):
        def get_column(x):
            return torch.tensor(x)

        return func(get_column(self.targs), get_column(self.preds))

    @property
    def accuracy(self):
        return self._compute(self.__accuracy)

    @property
    def precision(self):
        return self._compute(self.__precision)

    @property
    def recall(self):
        return self._compute(self.__recall)

    @property
    def f1score(self):
        return self._compute(self.__f1score)


def dump_scalar_metrics(
    metrics: dict, writer: SummaryWriter, phase: str, global_step: int = 0, dataset: str = "", **kwargs
):
    prefix = phase + (f"_{dataset}" if dataset else "")
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f"{metric_name}/{prefix}",
            metric_value,
            global_step=global_step,
        )


def add_metrics_dict(metrics: dict, metrics_new: dict) -> dict:
    for key, value in metrics_new.items():
        metrics[key] = metrics.get(key, 0) + value
    return metrics


def divide_metrics(metrics: dict, n: int) -> dict:
    return {key: (value / n) for key, value in metrics.items()}
