import torch
import torch.nn as nn

from .disll_op import SelfDistillMutator, layernorm2layernorm
from .ops import QuantLinear


class SelfDistillAlgorithm(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        self.mutator = SelfDistillMutator()
        self.mutator.prepare_from_supernet(self.model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.training:
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            if labels is not None:  # require loss
                labels = torch.cat([labels, labels], dim=0)
            out = self.model(input_ids, attention_mask, position_ids,
                             past_key_values, inputs_embeds, labels, use_cache,
                             output_attentions, output_hidden_states,
                             return_dict)

            loss = self.mutator.gather_distill_loss()
            if isinstance(loss, torch.Tensor):
                out['loss'] = loss
            return out
        else:
            out = self.model(input_ids, attention_mask, position_ids,
                             past_key_values, inputs_embeds, labels, use_cache,
                             output_attentions, output_hidden_states,
                             return_dict)
            return out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def linear2qlinear(module: nn.Linear):
    from torch.ao.quantization import QConfig, MinMaxObserver, default_observer
    qconfig = QConfig(activation=MinMaxObserver.with_args(dtype=torch.qint8),
                      weight=default_observer.with_args(dtype=torch.qint8))

    new_module = QuantLinear(module.in_features,
                             module.out_features,
                             module.bias is not None,
                             qconfig=qconfig)
    new_module.load_state_dict(module.state_dict(),strict=False)
    return new_module


stu_op_generator = {
    nn.LayerNorm: layernorm2layernorm,
    nn.Linear: linear2qlinear
}


class QatDistillAlgorithm(SelfDistillAlgorithm):

    def __init__(self, model: nn.Module) -> None:
        nn.Module.__init__(self)
        self.model = model

        self.mutator = SelfDistillMutator(stu_op_generator=stu_op_generator)
        self.mutator.prepare_from_supernet(self.model)