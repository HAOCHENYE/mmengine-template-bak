from mmeval import BaseMetric

from mmengine_template.registry import METRICS


@METRICS.register_module()
class CustomMetric(BaseMetric):

    def add(self, gt, preds):
        ...

    # NOTE for evaluator
    def compute_metric(self, size):
        ...
