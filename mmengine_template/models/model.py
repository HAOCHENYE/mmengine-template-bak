from mmengine.model import BaseModel

from mmengine_template.registry import MODELS


@MODELS.register_module()
class CustomModel(BaseModel):
    ...
