from torch.utils.tensorboard import SummaryWriter

# from skimage.metrics import structural_similarity
from torchmetrics import Accuracy, F1Score, Precision, Recall


class MetricPerformance:
    def __init__(self, num_classes, device="cuda", *args, **kwargs):

        self.metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
            "precision": Precision(task="multiclass", num_classes=num_classes).to(device),
            "recall": Recall(task="multiclass", num_classes=num_classes).to(device),
            "f1score": F1Score(task="multiclass", num_classes=num_classes).to(device),
        }

    def update(self, preds, targs):
        for m in self.metrics.values():
            m.update(preds, targs)

    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for m in self.metrics.values():
            m.reset()


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
