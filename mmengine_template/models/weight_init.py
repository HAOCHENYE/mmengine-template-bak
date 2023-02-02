from mmengine.model.weight_init import BaseInit

from mmengine_template.registry import WEIGHT_INITIALIZERS


@WEIGHT_INITIALIZERS.register_module()
class CustomInitializer(BaseInit):
    ...
