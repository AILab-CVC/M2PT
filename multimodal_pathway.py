import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalReparamLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True,
                 origin_layer=None,
                 aux_weight=None,
                 is_aux_trainable=True):
        super().__init__(in_features, out_features, bias)
        self.cross_modal_scale = nn.Parameter(torch.zeros(1))
        assert self.weight.size() == aux_weight.size(), 'Target weight and aux weight must have the same shape'
        self.aux_weight = aux_weight
        self.aux_weight.requires_grad_(is_aux_trainable)
        if origin_layer is not None:
            with torch.no_grad():
                self.weight.copy_(origin_layer.weight)
                self.bias.copy_(origin_layer.bias)

    def forward(self, input):
        weight = self.weight + self.cross_modal_scale * self.aux_weight
        return F.linear(input, weight, self.bias)


def build_cross_modal_reparam_linear(origin_layer, aux_layer):
    assert origin_layer.weight.size() == aux_layer.weight.size()
    return CrossModalReparamLinear(in_features=origin_layer.in_features, out_features=origin_layer.out_features, origin_layer=origin_layer,
                                   bias=origin_layer.bias is not None,
                                   aux_weight=aux_layer.weight)


def _get_attr_by_name(obj, attr_name):
    attrs = attr_name.split('.')
    for a in attrs:
        obj = obj.__getattr__(a)
    return obj

def _set_attr_by_name(obj, attr_name, attr_value):
    owner = obj
    attr_names = attr_name.split('.')
    if len(attr_names) > 1:
        for a in attr_names[:-1]:
            owner = owner.__getattr__(a)
    owner.__setattr__(attr_names[-1], attr_value)

def change_original_linear_to_reparam(target_module, aux_module, layer_name):
    origin_linear_layer = _get_attr_by_name(target_module, layer_name)
    aux_linear_layer = _get_attr_by_name(aux_module, layer_name)
    reparam_layer = build_cross_modal_reparam_linear(origin_linear_layer, aux_linear_layer)
    _set_attr_by_name(target_module, layer_name, reparam_layer)


def reparameterize_aux_into_target_model(target_model, aux_model,
                               layer_names=('attn.qkv', 'attn.proj', 'mlp.fc1','mlp.fc2'), main_body_name='blocks'):
    target_transformer_blocks = _get_attr_by_name(target_model, main_body_name)
    aux_transformer_blocks = _get_attr_by_name(aux_model, main_body_name)
    for target_block, aux_block in zip(target_transformer_blocks, aux_transformer_blocks):
        for layer_name in layer_names:
            change_original_linear_to_reparam(target_block, aux_block, layer_name)