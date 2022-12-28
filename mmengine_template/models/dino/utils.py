from detectron2.modeling.backbone import BasicStem, ResNet
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper

from mmengine_template.registry import MODELS

MODELS.register_module(
    name='PositionEmbeddingSine', module=PositionEmbeddingSine)
MODELS.register_module(name='HungarianMatcher', module=HungarianMatcher)


@MODELS.register_module(name='D2ResNet')
def build_d2_resnet(stem, stages, out_features, **kwargs):
    stages = ResNet.make_default_stages(**stages)
    stem = BasicStem(**stem)
    resnet = ResNet(
        stem=stem, stages=stages, out_features=out_features, **kwargs)
    return resnet


@MODELS.register_module(name='ChannelMapper')
def build_channel_mapper(*args, norm_layer=None, activation=None, **kwargs):
    if isinstance(norm_layer, dict):
        norm_layer = MODELS.build(norm_layer)

    if isinstance(activation, dict):
        activation = MODELS.build(activation)
    return ChannelMapper(
        *args, norm_layer=norm_layer, activation=activation, **kwargs)
