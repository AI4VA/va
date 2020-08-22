#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
    --test FILE              example testing mode using a restored model
    --train FILE             train file name.
    --valid FILE             valid file name.
    --infer_graph FILE       graph level inference stage.
    --infer_loc FILE         location level inference.
    --graph_rep FILE         infer the graph representations in the FILE
    --tsne_graph             draw the tsne in graph level before and after training
    --tsne_node              draw the tsne in node level
    --loc METHOD             apply different localization methods (graphpointer, hoppity, ggnnonly)
    --top N                  evaluate the accuracy based on top-n prediction
    --window_size N          evaluate the distance between predictions and targets
"""
from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json
import os
import random
from sklearn.manifold import TSNE
import subprocess
from sklearn.metrics import f1_score


from chem_tensorflow import ChemModel
from utils import glorot_init, ThreadedIterator, SMALL_NUMBER, getIndexPositions, attention, MLP
from Attention import Attention

GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells', ])


class SparseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)


    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 50,
            'use_edge_bias': False,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                                    "2": [0],
                                    "4": [0, 2]
            },

            'layer_timesteps': [8],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'edge_weight_dropout_keep_prob': .8
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # self.placeholders['num_graphs_int'] = 1
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['final_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='final_node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')
        self.placeholders['node_activation_values'] = tf.placeholder(tf.float32, [None, 2], name='node_activation_values')
        self.placeholders['graph_raw_representation'] = tf.placeholder(tf.float32, [None, self.params['hidden_size']],
                                                                     name='node_activation_values')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units:
        self.weights = {}  # Used by super-class to place generic things
        self.gnn_weights = GGNNWeights([], [], [], [])
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)
                edge_weights = tf.reshape(edge_weights, [self.num_edge_types, h_dim, h_dim])
                edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])
                self.gnn_weights.edge_weights.append(edge_weights)

                if self.params['use_propagation_attention']:
                    self.gnn_weights.edge_type_attention_weights.append(
                        tf.Variable(np.ones([self.num_edge_types], dtype=np.float32),
                                    name='edge_type_attention_weights_%i' % layer_idx))

                if self.params['use_edge_bias']:
                    self.gnn_weights.edge_biases.append(
                        tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                    name='gnn_edge_biases_%i' % layer_idx))

                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'cudnncompatiblegrucell':
                    assert (activation_name == 'tanh')
                    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
                    cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights.rnn_cells.append(cell)

    def compute_final_node_representations(self) -> tf.Tensor:
        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(self.placeholders['initial_node_representation'])
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int32)[0]

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Used shape abbreviations:
                #   V ~ number of nodes
                #   D ~ state dimension
                #   E ~ number of edges of current type
                #   M ~ number of messages (sum of all E)

                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                if self.params['use_propagation_attention']:
                    message_edge_type_factors = tf.nn.embedding_lookup(
                        params=self.gnn_weights.edge_type_attention_weights[layer_idx],
                        ids=message_edge_types)  # Shape [M]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        messages = []  # list of tensors of messages of shape [E, D]
                        message_source_states = []  # list of tensors of edge source states of shape [E, D]

                        # Collect incoming messages per edge type
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                                self.placeholders['adjacency_lists']):
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                        ids=edge_sources)  # Shape [E, D]
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights.edge_weights[layer_idx][
                                                                       edge_type_idx])  # Shape [E, D]
                            messages.append(all_messages_for_edge_type)
                            message_source_states.append(edge_source_states)

                        messages = tf.concat(messages, axis=0)  # Shape [M, D]

                        if self.params['use_propagation_attention']:
                            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)  # Shape [M, D]
                            message_attention_scores = tf.einsum('mi,mi->m', message_source_states,
                                                                 message_target_states)  # Shape [M]
                            message_attention_scores = message_attention_scores * message_edge_type_factors

                            # The following is softmax-ing over the incoming messages per node.
                            # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
                            # Step (1): Obtain shift constant as max of messages going into a node
                            message_attention_score_max_per_target = tf.unsorted_segment_max(
                                data=message_attention_scores,
                                segment_ids=message_targets,
                                num_segments=num_nodes)  # Shape [V]
                            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
                            message_attention_score_max_per_message = tf.gather(
                                params=message_attention_score_max_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention_scores -= message_attention_score_max_per_message
                            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
                            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
                            message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                                data=message_attention_scores_exped,
                                segment_ids=message_targets,
                                num_segments=num_nodes)  # Shape [V]
                            message_attention_normalisation_sum_per_message = tf.gather(
                                params=message_attention_score_sum_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention = message_attention_scores_exped / (
                                        message_attention_normalisation_sum_per_message + SMALL_NUMBER)  # Shape [M]
                            # Step (4): Weigh messages using the attention prob:
                            messages = messages * tf.expand_dims(message_attention, -1)

                        incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)  # Shape [V, D]

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)  # Shape [V, 1]
                            incoming_messages /= num_incoming_edges + SMALL_NUMBER

                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                        # pass updated vertex features into RNN cell
                        node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                                                                                          node_states_per_layer[-1])[
                            1]  # Shape [V, D]

        return node_states_per_layer[-1]


    def graph_attention(self, last_h, regression_gate, regression_transform, max_nodes_size=1501):
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = regression_gate(gate_input) * regression_transform(last_h)  # [v x h]
        self.placeholders['graph_raw_representation'] = tf.unsorted_segment_max(data=last_h,
                                                                                segment_ids=self.placeholders[
                                                                                    'graph_nodes_list'],
                                                                                num_segments=self.placeholders[
                                                                                    'num_graphs'])
        batch_graph_representation = tf.gather(params=self.placeholders['graph_raw_representation'],
                                               indices=self.placeholders['graph_nodes_list'])  # [v * h]
        graph_length = tf.unsorted_segment_sum(data=tf.ones([tf.shape(gated_outputs)[0]], dtype=tf.int32),
                                               segment_ids=self.placeholders['graph_nodes_list'],
                                               num_segments=self.placeholders['num_graphs'])


        node_rep = tf.add(gated_outputs, batch_graph_representation)
        # node_rep = gated_outputs
        graph_node_score = tf.dynamic_partition(data=node_rep,
                                      partitions=self.placeholders['graph_nodes_list'],
                                      num_partitions=self.params['batch_size'])  # [g * h]

        padded_nodes_list = list()
        mask_list = list()
        for s in graph_node_score:
            padded_s = tf.pad(s, [[0, max_nodes_size - tf.shape(s)[0]], [0, 0]])
            padded_nodes_list.append(padded_s)
            mask = tf.zeros([tf.shape(s)[0]])
            padded_mask = tf.pad(mask, [[0, max_nodes_size - tf.shape(s)[0]]], constant_values=float('-inf'))
            mask_list.append(padded_mask)
        padded_nodes = tf.stack(padded_nodes_list)  # [batch_size * max_node * h]
        masks = tf.stack(mask_list) # [batch_size * max_node]
        attention_inputs = padded_nodes
        # enc_cell = tf.contrib.rnn.GRUCell(self.params['hidden_size'])
        # attention_inputs, _ = tf.nn.dynamic_rnn(enc_cell, padded_nodes, graph_length, dtype=tf.float32) # [batch * max_node * h]
        # enc_cell = tf.contrib.rnn.GRUCell(self.params['hidden_size'] / 2)
        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(enc_cell, enc_cell, padded_nodes, graph_length, dtype=tf.float32)
        # conc_outputs = tf.concat(outputs, 2)
        # alpha = tf.reduce_sum(conc_outputs, 2)

        # mlp = MLP(self.params['hidden_size'], 2, [], self.placeholders['out_layer_dropout_keep_prob'])
        # alpha = mlp(outputs)
        # alpha = tf.reduce_sum(padded_nodes, axis=-1)
        # outputs, _ = attention(attention_inputs, self.params['hidden_size'], masks, return_alphas=True)  # [batch x h]
        # mlp = MLP(self.params['hidden_size'], 2, [], self.placeholders['out_layer_dropout_keep_prob'])
        # alpha = mlp(outputs)  # [batch * 2]
        _, alpha = attention(attention_inputs, self.params['hidden_size'], masks, return_alphas=True) # [b x max_node]
        # outputs, _ = attention(attention_inputs, self.params['hidden_size'], masks, return_alphas=True) # [batch x h]
        # node_scores = tf.reduce_sum(padded_nodes, axis=-1) # [b x max_node]
        # score_value = tf.concat([tf.expand_dims(alpha, -1), tf.expand_dims(node_scores, -1)], axis=-1)
        # initializer = tf.random_normal_initializer(stddev=0.1)
        # w_score = tf.get_variable(name="w_score", shape=[2], initializer=initializer, trainable=True)
        # alpha = tf.tensordot(score_value, w_score, axes=1, name='alpha')
        # alpha = tf.add(alpha, node_scores)
        return alpha

    def gated_regression_for_attention(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = regression_gate(gate_input) * regression_transform(last_h)  # [v x h]
        self.placeholders['graph_raw_representation'] = tf.unsorted_segment_max(data=last_h,
                                                                                segment_ids=self.placeholders[
                                                                                    'graph_nodes_list'],
                                                                                num_segments=self.placeholders[
                                                                                    'num_graphs'])  # [g * h]
        batch_graph_representation = tf.gather(params=self.placeholders['graph_raw_representation'],
                                               indices=self.placeholders['graph_nodes_list'])  # [v * h]
        final_outputs = tf.add(gated_outputs, batch_graph_representation)
        # self.output = graph_representations
        # return final_outputs
        return gated_outputs

    def score_node_sum(self, last_h, max_nodes_size=1501):
        # plain GGNN for node localization
        node_scores = tf.reduce_sum(last_h, axis=1)
        scores = tf.dynamic_partition(data=node_scores,
                                      partitions=self.placeholders['graph_nodes_list'],
                                      num_partitions=self.params['batch_size'])  # [g]

        padded_scores_list = list()
        for s in scores:
            padded_s = tf.pad(s, [[0, max_nodes_size - tf.shape(s)[0]]], constant_values=float('-inf'))
            padded_scores_list.append(padded_s)
        padded_score = tf.stack(padded_scores_list)  # [batch_size * max_node]
        return padded_score

    def ggnn_only_regression(self, last_h):
        regression_gate = MLP(2 * self.params['hidden_size'], 1, [], self.placeholders['out_layer_dropout_keep_prob'])
        regression_transform = MLP(self.params['hidden_size'], 1, [], self.placeholders['out_layer_dropout_keep_prob'])
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]

        # Sum up all nodes per-graph
        graph_representations = tf.unsorted_segment_max(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 1]
        output = tf.squeeze(graph_representations)  # [g]
        return output


    def graph_pointer_regression(self, last_h, regression_gate, regression_transform, max_nodes_size=601):
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = regression_gate(gate_input) * regression_transform(last_h)  # [v x h]
        node_rep = gated_outputs
        graph_node_score = tf.dynamic_partition(data=node_rep,
                                                partitions=self.placeholders['graph_nodes_list'],
                                                num_partitions=self.params['batch_size'])  # [g * h]

        padded_nodes_list = list()
        mask_list = list()
        for s in graph_node_score:
            padded_s = tf.pad(s, [[0, max_nodes_size - tf.shape(s)[0]], [0, 0]])
            padded_nodes_list.append(padded_s)
            mask = tf.zeros([tf.shape(s)[0]])
            padded_mask = tf.pad(mask, [[0, max_nodes_size - tf.shape(s)[0]]], constant_values=float('-inf'))
            mask_list.append(padded_mask)
        padded_nodes = tf.stack(padded_nodes_list)  # [batch_size * max_node * h]
        masks = tf.stack(mask_list)  # [batch_size * max_node]
        attention_inputs = padded_nodes
        outputs, _ = attention(attention_inputs, self.params['hidden_size'], masks, return_alphas=True) # [batch x h]
        mlp = MLP(self.params['hidden_size'], 1, [], self.placeholders['out_layer_dropout_keep_prob'])
        alpha = mlp(outputs)
        return alpha

    def gated_regression_inner_product(self, last_h, regression_gate, regression_transform, max_nodes_size=1501):
        # implementation of Hoppity localization
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = regression_gate(gate_input) * regression_transform(last_h)  # [v x h]
        self.placeholders['graph_raw_representation'] = tf.unsorted_segment_max(data=last_h,
                                                                                segment_ids=self.placeholders[
                                                                                    'graph_nodes_list'],
                                                                                num_segments=self.placeholders[
                                                                                    'num_graphs'])  # [g * h]

        graph_representation = tf.expand_dims(self.placeholders['graph_raw_representation'], 1)
        enc_cell = tf.contrib.rnn.GRUCell(self.params['hidden_size'])
        context_prime, _ = tf.nn.dynamic_rnn(enc_cell, graph_representation, dtype=tf.float32)
        context_vector = tf.squeeze(context_prime, [1])
        batch_graph_representation = tf.gather(params=context_vector,
                                               indices=self.placeholders['graph_nodes_list'])  # [v * h]
        # inner product
        node_scores = tf.reduce_sum(tf.multiply(gated_outputs, batch_graph_representation), axis=1)  # [v]

        # self.placeholders['node_activation_values'] = node_scores
        # For debugging data
        # node_scores = tf.reduce_sum(last_h, axis=1)
        scores = tf.dynamic_partition(data=node_scores,
                                      partitions=self.placeholders['graph_nodes_list'],
                                      num_partitions=self.params['batch_size'])  # [g]

        padded_scores_list = list()
        for s in scores:
            padded_s = tf.pad(s, [[0, max_nodes_size-tf.shape(s)[0]]], constant_values=float('-inf'))
            padded_scores_list.append(padded_s)
        padded_score = tf.stack(padded_scores_list)  # [batch_size * max_node]
        return padded_score


    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        self.placeholders['final_node_representation'] = last_h
        self.placeholders['graph_raw_representation'] = tf.unsorted_segment_sum(data=last_h,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g * h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 2]
        self.placeholders['node_activation_values'] = gated_outputs

        # Sum up all nodes per-graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 2]
        # output = tf.nn.softmax(graph_representations)  # [g * 2]
        self.output = graph_representations
        return graph_representations

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        processed_graphs = []
        for d in raw_data:
            # d = {"targets": d["targets"], "graph": d["graph"], "node_features": d["node_features"]}
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"],
                                     "labels": [d["targets"][task_id][0] for task_id in self.params['task_ids']]})

        if is_training_data:
            np.random.shuffle(processed_graphs)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['labels'][task_id] = None

        return processed_graphs

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0
            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = self.num_edge_types + edge_type
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type



    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while num_graphs < len(data):
        # while num_graphs < self.params['batch_size']:
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0
            graph_offset = 0

            # while num_graphs < len(data) and num_graphs_in_batch < self.params['batch_size']:
            while num_graphs_in_batch < self.params['batch_size']:
                if num_graphs < len(data):
                    cur_graph = data[num_graphs]
                else:
                    # idx = (num_graphs - len(data)) % len(data)
                    idx = random.randint(0, len(data) - 1)
                    cur_graph = data[idx]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                        ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                        'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                target_task_values = []
                target_task_mask = []
                if num_graphs < len(data):
                    for target_val in cur_graph['labels']:
                        if target_val is None:  # This is one of the examples we didn't sample...
                            target_task_values.append(0.)
                            target_task_mask.append(0.)
                        else:
                            target_task_values.append(target_val)
                            target_task_mask.append(1.)
                else:
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                batch_target_task_values.append(target_task_values)
                batch_target_task_mask.append(target_task_mask)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph


            self.num_graphs = num_graphs_in_batch
            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1, 0]),
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

    def evaluate_one_batch(self, data):
        fetch_list = self.output
        batch_feed_dict = self.make_minibatch_iterator(data, is_training=False)

        for item in batch_feed_dict:
            item[self.placeholders['graph_state_keep_prob']] = 1.0
            item[self.placeholders['edge_weight_dropout_keep_prob']] = 1.0
            item[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            item[self.placeholders['target_values']] = [[]]
            item[self.placeholders['target_mask']] = [[]]
            print(self.sess.run(fetch_list, feed_dict=item))

    def example_evaluation(self):
        ''' Demonstration of what test-time code would look like
        we query the model with the first n_example_molecules from the validation file
        '''
        n_example_molecules = 10
        with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]

        for mol in example_molecules:
            print(mol['targets'])

        example_molecules = self.process_raw_graphs(example_molecules, is_training_data=False)
        self.evaluate_one_batch(example_molecules)

    def test_one_epoch(self, data, outDir=None, cls=False):
        # fetch_list = self.output
        loss = 0
        accuracies = []
        tps = []
        fps = []
        fns = []
        predictions = np.array([])
        pred_distance = np.array([])
        target_values = np.array([])
        target_masks = np.array([])
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        processed_graphs = 0
        steps = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training=False), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            # fetch_list = [self.ops['loss'], accuracy_ops, self.ops['TP'], self.ops['FP'], self.ops['FN'], tf.math.argmax(self.output, axis=-1, output_type=tf.int32)]

            if cls:
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['TP'], self.ops['FP'], self.ops['FN'],
                            self.placeholders['predictions'], self.placeholders['target_values'][0, :],
                            self.placeholders['target_mask'][0, :]]
            else:
                fetch_list = [self.ops['loss'], accuracy_ops,
                              self.placeholders['predictions'], self.placeholders['pred_distance'], self.placeholders['target_values'][0, :],
                              self.placeholders['target_mask'][0, :]]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            if cls:
                (batch_loss, batch_accuracies, TP, FP, FN, batch_predictions, batch_target_values, batch_target_masks) = (
                    result[0], result[1],
                    result[2], result[3],
                    result[4], result[5],
                    result[6], result[7])
            else:
                (batch_loss, batch_accuracies, batch_predictions, batch_distance, batch_target_values, batch_target_masks) = (
                                                                                                        result[0], result[1],
                                                                                                        result[2], result[3],
                                                                                                        result[4], result[5])
            # (batch_loss, batch_accuracies, TP, FP, FN, pred) = (result[0], result[1], result[2], result[3], result[4], result[5])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)
            if cls:
                tps.append(np.array(TP))
                fps.append(np.array(FP))
                fns.append(np.array(FN))
            steps += 1
            predictions = np.append(predictions, batch_predictions)
            if self.args.get("--window_size") is not None:
                pred_distance = np.append(pred_distance, batch_distance)
            target_values = np.append(target_values, batch_target_values)
            target_masks = np.append(target_masks, batch_target_masks)
        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        if cls:
            precision = np.sum(tps) / ((np.sum(tps) + np.sum(fps)) + SMALL_NUMBER)
            recall = np.sum(tps) / ((np.sum(tps) + np.sum(fns)) + SMALL_NUMBER)
            f1 = 2 * precision * recall / ((precision + recall) + SMALL_NUMBER)
            test_result = {"loss": "%.5f" % loss, "acc": "%.5f" % accuracies, "precision": "%.5f" % precision,
                           "recall": "%.5f" % recall, "f1": "%.5f" % f1}
            print("\r\x1b[K Test: loss: %.5f | acc: %.5f | precision: %.5f | recall: %.5f | f1: %.5f" %
                  (loss, accuracies, precision, recall, f1))
        else:
            f1 = f1_score(target_values, predictions, labels=np.array(range(1501)), average='weighted')
            test_result = {"loss": "%.5f" % loss, "acc": "%.5f" % accuracies}
            print("\r\x1b[K Test: loss: %.5f | acc: %.5f | f1: %.5f" % (loss, accuracies, f1))

        if outDir is not None:
            test_output = os.path.join(outDir, 'test_result.json')
            np.save(os.path.join(outDir, "predictions.npy"), predictions, allow_pickle=False)
            np.save(os.path.join(outDir, "target_values.npy"), target_values, allow_pickle=False)
            np.save(os.path.join(outDir, "target_masks.npy"), target_masks, allow_pickle=False)
            np.save(os.path.join(outDir, "pred_distance.npy"), pred_distance, allow_pickle=False)
            with open(test_output, 'w') as to:
                json.dump(test_result, to)
            with open(os.path.join(outDir, 'pred.txt'), 'w') as acf:
                acf.write(str(predictions))
        else:
            np.save("predictions.npy", predictions, allow_pickle=False)
            np.save("target_values.npy", target_values, allow_pickle=False)
            np.save("target_masks.npy", target_masks, allow_pickle=False)
        #

        # print(pred)

    def example_test(self, args):
        # n_example_molecules = 10
        if args.get('--test') is not None:
            with open(args.get('--test'), 'r') as valid_file:
                example_molecules = json.load(valid_file)
        else:
            with open('/local/VulnerabilityData/chrome_debian_with_node_embedding/valid_GGNNinput.json',
                      'r') as valid_file:
                example_molecules = json.load(valid_file)

        example_molecules = self.process_raw_graphs(example_molecules, is_training_data=False)
        if args.get('--log_dir') is not None:
            self.test_one_epoch(example_molecules, args.get('--log_dir'))
        else:
            self.test_one_epoch(example_molecules)

    def inference(self, data, dirName=None, single=True):
        steps = 0
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training=False), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.placeholders['graph_nodes_list'], self.placeholders['node_activation_values'],
                          self.placeholders['predictions'], self.placeholders['target_values'][0, :],
                          self.placeholders['num_graphs'], accuracy_ops]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            # we only consider single file inference for now
            (graph_node_list, node_activation_values, predictions, target_values, num_graphs, acc) = (result[0], result[1],
                                                                                                 result[2], result[3],
                                                                                                 result[4], result[5])
            steps += 1
        # print(graph_node_list, node_activation_values, predictions, num_graphs)
        # print("%.5f" % np.sum(acc))
        # print(predictions)
        graph_node_list = list(graph_node_list)
        activation_values = dict()
        # node_activation_values = list(node_activation_values)
        if single and dirName is not None:
            distances = np.subtract(node_activation_values[:, 1], node_activation_values[:, 0])
            #print(distances)
            node_index = np.argsort(distances)
            # print (node_index)
            nodeFile = os.path.join(dirName, 'nodes_normalized.csv')
            with open(os.path.join(dirName, 'pred.txt'), 'w') as acf:
                acf.write(str(predictions))
            activationFile = os.path.join(dirName, 'activation.json')
            with open(nodeFile, 'r') as nf:
                nodes = nf.readlines()
                for idx in node_index[::-1]:
                    # print(nodes[idx].strip())
                    key = nodes[idx].split()[0]
                    content = nodes[idx].split()[1:]
                    activation_values[key] = (str(distances[idx]), content)
            with open(activationFile, 'w') as af:
                json.dump(activation_values, af)
            if str(predictions) == '[1]':
                return 1
            else:
                print(str(predictions))
        else:
            for i in range(num_graphs):
                graph_node_activations = list()
                for idx in getIndexPositions(graph_node_list, i):
                    graph_node_activations.append(node_activation_values[idx])
                graph_node_activations = np.asarray(graph_node_activations)
            print (graph_node_activations)
            print (np.argmax(graph_node_activations, axis=0))
            # return predictions, graph_node_activations


    def example_inference(self, args, single=False):
        if single:
            inferFile = args.get('--infer')
            if inferFile is not None:
                inferDir = os.path.dirname(inferFile)
                with open(inferFile, 'r') as infer_file:
                    examples = json.load(infer_file)
                example_molecules = self.process_raw_graphs(examples, is_training_data=False)
                self.inference(example_molecules, dirName=inferDir)
                # pred, activations = self.inference(example_molecules, dirName=inferDir)
                # with open(inferFile.replace('.json', '_aug.json'), 'w') as i:
                    # examples[0]["activations"] = activations.tolist()
                    # examples[0]["prediction"] = pred.tolist()
                    # json.dump(examples, i)
        else:
            cmd = ''
            cmd += "find " + args.get('--infer') + " -name *GGNNinput_infer.json;"
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
            result = result.stdout.decode('utf-8')
            result = result.split("\n")
            inferResults = list(filter(None, result))
            correct = 0
            cnt = 0
            for inferFile in inferResults:
                inferDir = os.path.dirname(inferFile)
                with open(inferFile, 'r') as infer_file:
                    example_molecules = json.load(infer_file)
                example_molecules = self.process_raw_graphs(example_molecules, is_training_data=False)
                c = self.inference(example_molecules, dirName=inferDir)
                if c:
                    correct += 1
                cnt += 1
                if cnt % 1000 == 0:
                    print ("Inferred samples: ", cnt)
            print ("Prediction acc: ", correct / len(inferResults))


    def loc_inference(self, args):
        inferFile = args.get('--infer')
        if inferFile is not None:
            inferDir = os.path.dirname(inferFile)
            with open(inferFile, 'r') as infer_file:
                examples = json.load(infer_file)
            example_molecules = self.process_raw_graphs(examples, is_training_data=False)
            # self.inference(example_molecules, dirName=inferDir)


    def graph_representation(self, args):
        with open(args.get('--graph_rep'), 'r') as test_file:
            tList = json.load(test_file)
        data = self.process_raw_graphs(tList, is_training_data=False)
        target_values = np.array([])
        target_masks = np.array([])
        steps = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training=False), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.placeholders['graph_raw_representation'], self.placeholders['target_values'][0, :],
                              self.placeholders['target_mask'][0, :]]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_graph_raw_representations, batch_target_values, batch_target_masks) = (result[0], result[1], result[2])
            steps += 1
            if step == 0:
                graph_raw_representation = batch_graph_raw_representations
            else:
                graph_raw_representation = np.append(graph_raw_representation, batch_graph_raw_representations, axis=0)
            target_values = np.append(target_values, batch_target_values)
            target_masks = np.append(target_masks, batch_target_masks)

        gList = graph_raw_representation.tolist()
        tList = target_values.tolist()
        mList = target_masks.tolist()
        graph_json = list()
        for idx, m in enumerate(mList):
            if m == 1:
                sample = dict()
                sample["target"] = int(tList[idx])
                sample["graph_feature"] = gList[idx]
                graph_json.append(sample)

        oFile = os.path.basename(args.get('--graph_rep')).split('.')[0] + '_graph.json'
        if args.get('--log_dir'):
            oPath = os.path.join(args.get('--log_dir'), oFile)
            with open(oPath, 'w') as of:
                json.dump(graph_json, of)
        else:
            with open(oFile, 'w') as of:
                json.dump(graph_json, of)


    def draw_graph_tsne(self, args):
        if args.get('--test') is not None:
            points = list()
            labels = list()
            with open(args.get('--test'), 'r') as test_file:
                tList = json.load(test_file)
            data = self.process_raw_graphs(tList, is_training_data=False)
            '''
            for t in tList:
                graphRepresent = np.sum(np.asarray(t["node_features"]), axis=0)
                points.append(graphRepresent)
                labels.append(t["targets"][0][0])
            points = np.array(points)
            labels = np.array(labels)
            X_embedded = TSNE(n_components=2).fit_transform(points)
            if args.get('--log_dir'):
                np.save(os.path.join(args.get('--log_dir'), "tsne_pre_points.npy"), X_embedded, allow_pickle=False)
                np.save(os.path.join(args.get('--log_dir'), "tsne_pre_labels.npy"), labels, allow_pickle=False)
            else:
                np.save("tsne_pre_points.npy", X_embedded, allow_pickle=False)
                np.save("tsne_pre_labels.npy", labels, allow_pickle=False)
            '''
            target_values = np.array([])
            target_masks = np.array([])
            steps = 0
            batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training=False), max_queue_size=5)
            for step, batch_data in enumerate(batch_iterator):
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.placeholders['graph_raw_representation'], self.placeholders['target_values'][0, :],
                              self.placeholders['target_mask'][0, :]]
                result = self.sess.run(fetch_list, feed_dict=batch_data)
                (batch_graph_raw_representations, batch_target_values, batch_target_masks) = (result[0], result[1], result[2])
                steps += 1
                if step == 0:
                    graph_raw_representation = batch_graph_raw_representations
                else:
                    graph_raw_representation = np.append(graph_raw_representation, batch_graph_raw_representations, axis=0)
                target_values = np.append(target_values, batch_target_values)
                target_masks = np.append(target_masks, batch_target_masks)

            X_prime_embedded = TSNE(n_components=2).fit_transform(graph_raw_representation)
            '''
            target_masks = list(target_masks.astype(int))
            mask_index = list()
            for idx, tm in enumerate(target_masks):
                if tm == 0:
                    mask_index.append(idx)
            print(mask_index)
            '''
            if args.get('--log_dir'):
                np.save(os.path.join(args.get('--log_dir'), "tsne_post_graphs.npy"), X_prime_embedded, allow_pickle=False)
                np.save(os.path.join(args.get('--log_dir'), "tsne_post_labels.npy"), target_values, allow_pickle=False)
            else:
                np.save("tsne_post_graphs.npy", X_prime_embedded, allow_pickle=False)
                np.save("tsne_post_labels.npy", target_values, allow_pickle=False)


    def draw_node_tsne(self, args, pre=True, post=False):
        if args.get('--test') is not None:
            # points = list()
            labels = np.array([])
            with open(args.get('--test'), 'r') as test_file:
                tList = json.load(test_file)
            if pre:
                # labels = tList[0]["node_targets"]
                for st, t in enumerate(tList):
                    if st == 0:
                        points = np.array(t["node_features"])
                    else:
                        points = np.append(points, np.array(t["node_features"]), axis=0)
                    labels = np.append(labels, np.array(t["node_targets"]))
                # points = np.asarray(points)
                # labels = np.array(labels)
                X_embedded = TSNE(n_components=2).fit_transform(points)
                if args.get('--log_dir'):
                    np.save(os.path.join(args.get('--log_dir'), "tsne_pre_nodes.npy"), X_embedded, allow_pickle=False)
                    np.save(os.path.join(args.get('--log_dir'), "tsne_node_labels.npy"), labels, allow_pickle=False)
                else:
                    np.save("tsne_pre_nodes.npy", X_embedded, allow_pickle=False)
                    np.save("tsne_node_labels.npy", labels, allow_pickle=False)

            if post:
                data = self.process_raw_graphs(tList, is_training_data=False)
                for t in tList:
                    labels = np.append(labels, np.array(t["node_targets"]))
                steps = 0
                batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training=False), max_queue_size=5)
                for step, batch_data in enumerate(batch_iterator):
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                    fetch_list = [self.placeholders['final_node_representation']]
                    result = self.sess.run(fetch_list, feed_dict=batch_data)
                    batch_final_node_representation = result[0]
                    steps += 1
                    if step == 0:
                        final_node_representation = batch_final_node_representation
                    else:
                        final_node_representation = np.append(final_node_representation, batch_final_node_representation, axis=0)


                X_prime_embedded = TSNE(n_components=2).fit_transform(final_node_representation)
                if args.get('--log_dir'):
                    np.save(os.path.join(args.get('--log_dir'), "tsne_post_nodes.npy"), X_prime_embedded, allow_pickle=False)
                    np.save(os.path.join(args.get('--log_dir'), "tsne_node_labels.npy"), labels, allow_pickle=False)
                else:
                    np.save("tsne_post_nodes.npy", X_prime_embedded, allow_pickle=False)
                    np.save("tsne_node_labels.npy", labels, allow_pickle=False)


def main():
    args = docopt(__doc__)
    try:
        model = SparseGGNNChemModel(args)
        if args['--evaluate']:
            model.example_evaluation()
        elif args['--graph_rep']:
            model.graph_representation(args)
        elif args['--tsne_graph'] and args['--test']:
            model.draw_graph_tsne(args)
        elif args['--tsne_node'] and args['--test']:
            model.draw_node_tsne(args)
        elif args['--test']:
            model.example_test(args)
        elif args['--infer_graph']:
            model.example_inference(args)
        elif args['--infer_loc']:
            model.loc_inference(args)
        else:
            model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
