from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base
from tensor2tensor.models.lstm import lstm_attention

@registry.register_hparams
def transformer_scan():
	hparams = transformer_base()
	return hparams

@registry.register_hparams
def gru_attention_scan():
	hparams = lstm_attention()
	hparams.learning_rate_constant = 0.001
	hparams.learning_rate_schedule = "constant"
	hparams.add_hparam("stack_size", 10)
	hparams.add_hparam("num_stacks", 10)
	hparams.add_hparam("decoder_type", DECODER_TYPE)
	hparams.num_hidden_layers = 1
	hparams.hidden_size = 50
	hparams.dropout = 0.5
	hparams.add_hparam("eval_throttle_seconds", 100)
	return hparams


