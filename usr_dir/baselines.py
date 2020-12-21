"""RNN LSTM models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.utils import registry, t2t_model

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.keras.layers.core import *
from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.models import lstm, transformer, universal_transformer


def _dropout_gru_cell(hparams, train):
    return tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))


def gru(inputs, sequence_length, hparams, train, name, initial_state=None):
    layers = [_dropout_gru_cell(hparams, train)
              for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        return tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell(layers),
            inputs,
            sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)


def gru_attention_decoder(inputs, hparams, train, name, initial_state,
                          encoder_outputs, encoder_output_length,
                          decoder_input_length):
    layers = [_dropout_gru_cell(hparams, train)
              for _ in range(hparams.num_hidden_layers)]
    if hparams.attention_mechanism == "luong":
        attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
    elif hparams.attention_mechanism == "bahdanau":
        attention_mechanism_class = tf.contrib.seq2seq.BahdanauAttention
    else:
        raise ValueError("Unknown hparams.attention_mechanism = %s, must be "
                         "luong or bahdanau." % hparams.attention_mechanism)
    attention_mechanism = attention_mechanism_class(
        hparams.hidden_size, encoder_outputs,
        memory_sequence_length=encoder_output_length)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        [attention_mechanism]*hparams.num_heads,
        attention_layer_size=[hparams.attention_layer_size]*hparams.num_heads,
        output_attention=(hparams.output_attention == 1),
        alignment_history=True)

    batch_size = common_layers.shape_list(inputs)[0]

    initial_state = cell.zero_state(batch_size, tf.float32).clone(
        cell_state=initial_state)

    with tf.variable_scope(name):
        output, final_output_states = tf.nn.dynamic_rnn(
            cell,
            inputs,
            decoder_input_length,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)
        if hparams.output_attention == 1 and hparams.num_heads > 1:
            output = tf.layers.dense(output, hparams.hidden_size)

        return output, final_output_states


def gru_seq2seq_internal(inputs, targets, hparams, train):
    """The basic gru seq2seq model, main step used for training."""
    with tf.variable_scope("gru_seq2seq"):
        if inputs is not None:
            inputs_length = common_layers.length_from_embedding(inputs)
            # Flatten inputs.
            inputs = common_layers.flatten4d3d(inputs)
            inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
            _, final_encoder_state = gru(inputs, inputs_length, hparams, train,
                                         "encoder")
        else:
            final_encoder_state = None

        shifted_targets = common_layers.shift_right(targets)
        # Add 1 to account for the padding added to the left from shift_right
        targets_length = common_layers.length_from_embedding(shifted_targets) + 1
        decoder_outputs, _ = gru(
            common_layers.flatten4d3d(shifted_targets),
            targets_length,
            hparams,
            train,
            "decoder",
            initial_state=final_encoder_state)
        return tf.expand_dims(decoder_outputs, axis=2)


def gru_seq2seq_internal_attention(inputs, targets, hparams, train,
                                   inputs_length, targets_length):
    """LSTM seq2seq model with attention, main step used for training."""
    with tf.variable_scope("gru_seq2seq_attention"):
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)

        # LSTM encoder.
        inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
        encoder_outputs, final_encoder_state = gru(
            inputs, inputs_length, hparams, train, "encoder")

        # LSTM decoder with attention.
        shifted_targets = common_layers.shift_right(targets)
        # Add 1 to account for the padding added to the left from shift_right
        targets_length = targets_length + 1
        decoder_outputs, final_output_states = gru_attention_decoder(
            common_layers.flatten4d3d(shifted_targets), hparams, train, "decoder",
            final_encoder_state, encoder_outputs, inputs_length, targets_length)
        return tf.expand_dims(decoder_outputs, axis=2), final_output_states


def gru_bid_encoder(inputs, sequence_length, hparams, train, name):
    """Bidirectional LSTM for encoding inputs that are [batch x time x size]."""

    with tf.variable_scope(name):
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [_dropout_gru_cell(hparams, train)
             for _ in range(hparams.num_hidden_layers)])

        cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [_dropout_gru_cell(hparams, train)
             for _ in range(hparams.num_hidden_layers)])

        ((encoder_fw_outputs, encoder_bw_outputs),
         (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            sequence_length,
            dtype=tf.float32,
            time_major=False)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        encoder_states = []

        for i in range(hparams.num_hidden_layers):
            encoder_state = tf.concat(
                values=(encoder_fw_state[i], encoder_bw_state[i]),
                axis=1,
                name="bidirectional_concat")

            encoder_states.append(encoder_state)

        encoder_states = tuple(encoder_states)
        return encoder_outputs, encoder_states


def gru_seq2seq_internal_bid_encoder(inputs, targets, hparams, train):
    """The basic LSTM seq2seq model with bidirectional encoder."""
    with tf.variable_scope("gru_seq2seq_bid_encoder"):
        if inputs is not None:
            inputs_length = common_layers.length_from_embedding(inputs)
            # Flatten inputs.
            inputs = common_layers.flatten4d3d(inputs)
            # LSTM encoder.
            _, final_encoder_state = gru_bid_encoder(
                inputs, inputs_length, hparams, train, "encoder")
        else:
            inputs_length = None
            final_encoder_state = None
        # LSTM decoder.
        shifted_targets = common_layers.shift_right(targets)
        # Add 1 to account for the padding added to the left from shift_right
        targets_length = common_layers.length_from_embedding(shifted_targets) + 1
        hparams_decoder = copy.copy(hparams)
        hparams_decoder.hidden_size = 2 * hparams.hidden_size
        decoder_outputs, _ = gru(
            common_layers.flatten4d3d(shifted_targets),
            targets_length,
            hparams_decoder,
            train,
            "decoder",
            initial_state=final_encoder_state)
        return tf.expand_dims(decoder_outputs, axis=2)


def gru_seq2seq_internal_attention_bid_encoder(inputs, targets, hparams,
                                               train):
    with tf.variable_scope("gru_seq2seq_attention_bid_encoder"):
        inputs_length = common_layers.length_from_embedding(inputs)
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)
        # LSTM encoder.
        encoder_outputs, final_encoder_state = gru_bid_encoder(
            inputs, inputs_length, hparams, train, "encoder")
        # LSTM decoder with attention
        shifted_targets = common_layers.shift_right(targets)
        # Add 1 to account for the padding added to the left from shift_right
        targets_length = common_layers.length_from_embedding(shifted_targets) + 1
        hparams_decoder = copy.copy(hparams)
        hparams_decoder.hidden_size = 2 * hparams.hidden_size
        decoder_outputs = gru_attention_decoder(
            common_layers.flatten4d3d(shifted_targets), hparams_decoder, train,
            "decoder", final_encoder_state, encoder_outputs,
            inputs_length, targets_length)
        return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class GRUSeq2seq(t2t_model.T2TModel):

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        if self._hparams.initializer == "orthogonal":
            raise ValueError("LSTM models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        return gru_seq2seq_internal(features.get("inputs"), features["targets"],
                                    self._hparams, train)


@registry.register_model
class GRUSeq2seqAttention(t2t_model.T2TModel):
    """Seq to seq LSTM with attention."""

    def __init__(self, *args, **kwargs):
        super(GRUSeq2seqAttention, self).__init__(*args, **kwargs)
        self.attention_weights = {}  # For visualizing attention heads.

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        if self._hparams.initializer == "orthogonal":
            raise ValueError("LSTM models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        # This is a temporary fix for varying-length sequences within in a batch.
        # A more complete fix should pass a length tensor from outside so that
        # all the lstm variants can use it.
        input_shape = common_layers.shape_list(features["inputs_raw"])
        flat_input = tf.reshape(features["inputs_raw"],
                                [input_shape[0], input_shape[1]])
        inputs_length = tf.reduce_sum(tf.minimum(flat_input, 1), -1)
        target_shape = common_layers.shape_list(features["targets_raw"])
        flat_target = tf.reshape(features["targets_raw"],
                                 [target_shape[0], target_shape[1]])
        targets_length = tf.reduce_sum(tf.minimum(flat_target, 1), -1)
        outputs, final_output_states = gru_seq2seq_internal_attention(
            features["inputs"], features["targets"], self._hparams, train,
            inputs_length, targets_length)
        self.attention_weights = {"alignment_history": final_output_states.alignment_history[0].stack()}
        # alignment_history = tf.identity(final_output_states.alignment_history, "alignment_history")
        return outputs


@registry.register_model
class GRUSeq2seqBidirectionalEncoder(t2t_model.T2TModel):

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        if self._hparams.initializer == "orthogonal":
            raise ValueError("LSTM models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        return gru_seq2seq_internal_bid_encoder(
            features.get("inputs"), features["targets"], self._hparams, train)



