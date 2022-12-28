from detectron2.utils.events import EventStorage

from mmdet.models import Detectron2Wrapper as _Detectron2Wrapper
from mmengine_template.registry import MODELS


@MODELS.register_module()
class Detectron2Wrapper(_Detectron2Wrapper):

    def __init__(self,
                 detector,
                 pretrained=None,
                 bgr_to_rgb=False,
                 rgb_to_bgr=False) -> None:
        super(_Detectron2Wrapper, self).__init__()
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pretrained = pretrained
        if isinstance(detector, dict):
            self.cfg = detector
            detector = MODELS.build(detector)
        self.d2_model = detector
        self.storage = EventStorage()

    def init_weights(self) -> None:
        """Initialization Backbone.

        NOTE: The initialization of other layers are in Detectron2,
        if users want to change the initialization way, please
        change the code in Detectron2.
        """
        from detectron2.checkpoint import DetectionCheckpointer
        if self.pretrained is not None:
            checkpointer = DetectionCheckpointer(model=self.d2_model)
            checkpointer.load(self.pretrained, checkpointables=[])
