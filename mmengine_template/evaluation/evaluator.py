from mmengine.evaluator import Evaluator as MMEngineEvaluator
from mmengine.structures import BaseDataElement

from mmengine_template.registry import EVALUATOR


@EVALUATOR.register_module()
class Evaluator(MMEngineEvaluator):

    def process(self, data_samples, data_batch=None):
        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.add(data_batch, _data_samples)

    def evaluate(self, size):
        metrics = {}
        for metric in self.metrics:
            _results = metric.compute(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics
