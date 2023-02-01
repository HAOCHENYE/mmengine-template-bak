import os.path as osp

import numpy as np
import torch
from mmcv import imread
from mmengine.device import get_device
from mmengine.infer import BaseInferencer
from mmengine.visualization import Visualizer


class CustomInferencer(BaseInferencer):

    def _init_visualizer(self, cfg):
        """Return custom visualizer."""
        return Visualizer(name='mmengine_template')

    def _init_pipeline(self, cfg):
        """Return a pipeline to process input data.

        This default implementation will read the image and convert it to a
        Tensor with shape (C, H, W) and dtype torch.float32.
        """
        device = get_device()

        def naive_pipeline(image):
            image = np.float32(imread(image))
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).to(device)
            return dict(inputs=image)

        return naive_pipeline

    def visualize(self, inputs, preds, show=False):
        """Visualize the predictions on the original inputs."""
        visualization = []
        for image_path, pred in zip(inputs, preds):
            image = imread(image_path)
            self.visualizer.set_image(image)
            # NOTE The implementation of visualization is left to the user.
            ...
            if show:
                self.visualizer.show()
            vis_result = self.visualizer.get_image()
            # Return the visualization for post process.
            visualization.append(
                dict(image=vis_result, filename=osp.basename(image_path)))
        return visualization

    def postprocess(self, preds, visualization, return_datasample=False):
        """Apply post process to the predictions and visualization.

        For example, you can save the predictions or visualization to files in
        this method.

        Note:
            The parameter ``return_datasample`` should only be used when
            ``model.forward`` output a list of datasample instance.
        """
        ...
        return dict(predictions=preds, visualization=visualization)
