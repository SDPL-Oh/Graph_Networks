from graph_nets import utils_tf
from utils import models
# from graph_nets.demos_tf2 import models

import os
import numpy as np
import sonnet as snt
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

class load_tfrecord:
    def __init__(self, n):
        self.node = n

    def read_tfrecord(self, example):
        tfrecord_format = (
            {
            'image/filename': tf.io.FixedLenFeature((), tf.string),
            'image/object/x': tf.io.VarLenFeature(tf.float32),
            'image/object/y': tf.io.VarLenFeature(tf.float32),
            'image/object/Cp': tf.io.VarLenFeature(tf.float32),
            'image/object/aoa': tf.io.VarLenFeature(tf.float32)
            }
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        val_x, val_y, val_cp, val_aoa = self.sparse_function(example['image/object/x'],
                                                             example['image/object/y'],
                                                             example['image/object/Cp'],
                                                             example['image/object/aoa'])
        val_aoa = tf.sparse.to_dense(val_aoa)
        val_cpy = tf.sparse.to_dense(tf.sparse.concat(-1, [val_cp, val_y]))
        val_x = tf.sparse.to_dense(val_x)

        edges_arrange = tf.concat([tf.concat([tf.expand_dims(val_aoa[i], 0),
                                              tf.expand_dims(val_aoa[i], 0)], 0) for i in range(self.node)], 0)

        return edges_arrange, val_cpy, example['image/filename'], val_x

    def sparse_function(self, *args):
        output_args = []
        for arg in args:
            tmp = tf.sparse.reshape(arg, [-1, 1])
            output_args.append(tmp)
        return tuple(output_args)

    def load_dataset(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self, filenames, batch_size, is_training=True):
        dataset = self.load_dataset(filenames)
        if is_training:
            dataset = dataset.shuffle(1000)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).repeat(1)
        return dataset


############## options ##############
node = 160
num_process = 10
next_batch = load_tfrecord(node)
initial_learning_rate = 0.000424
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=30000,
    decay_rate=0.96,
    staircase=True)
my_module = models.EncodeProcessDecode(node_output_size=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

############## directory ##############
model_path = "models/graph_net/p/"
log_path = "logs/graph_net/p/"
plt_path = "dat/plt/graph_net/p/"

def initial_nodes():
    data = pd.DataFrame(np.genfromtxt(
        os.path.join('standard.txt'),
        names=('x', 'y', 'Cp'),
        dtype=float, skip_header=2))
    node_data = pd.concat([data['Cp']/10.0, data['y']], axis=1)
    return np.reshape(np.array(node_data, np.float32), (-1, 2))

initial_node = initial_nodes()
def base_graph_np():
    nodes = initial_node
    edges = []
    y_nodes = initial_node[..., 1:]
    for idx, val in enumerate(y_nodes):
        if idx+1 < len(y_nodes):
            y_dis = abs(val-y_nodes[idx+1])[-1]
            edges.append([y_dis, 0])
            edges.append([y_dis, 0])
        else:
            y_dis = abs(y_nodes[idx]-y_nodes[0])
            edges.append([y_dis, 0])
            edges.append([y_dis, 0])

    senders, receivers = [], []
    for i in range(node - 1):
        left_node = i
        right_node = i + 1
        if right_node < node:
            senders.append(left_node)
            receivers.append(right_node)
        if left_node >= 0:
            senders.append(right_node)
            receivers.append(left_node)

    senders.append(0)
    receivers.append(right_node)
    senders.append(right_node)
    receivers.append(0)

    return {
        "globals": [0.],
        "nodes": nodes,
        "edges": np.array(edges, np.float32),
        "receivers": receivers,
        "senders": senders
    }

def input_graph_to_tuple(edges, nodes, batch_size):
    batches_input_graph = []
    for batch in range(batch_size):
        init = base_graph_np()
        init['edges'][..., 1:2] = edges[batch]
        init['nodes'][..., 0:1] = nodes[batch][..., 0:1]
        batches_input_graph.append(init)
    input_tuple = utils_tf.data_dicts_to_graphs_tuple(batches_input_graph)
    return input_tuple

def target_graph_to_tuple(nodes, batch_size):
    batches_target_graph = []
    for batch in range(batch_size):
        init = base_graph_np()
        init['nodes'][..., 1:2] = nodes[batch][..., 1:2]
        batches_target_graph.append(init)
    target_tuple = utils_tf.data_dicts_to_graphs_tuple(batches_target_graph)
    return target_tuple

def single_loss(target_op, output_ops):
    loss_ops = [
        tf.reduce_mean(
            tf.reduce_sum((output_op.nodes - target_op.nodes[..., 1:]) ** 2, axis=-1))
        for output_op in output_ops]
    return tf.stack(loss_ops)

def average_loss(lbl_nodes, prd_nodes):
    per_example_loss = single_loss(lbl_nodes, prd_nodes)
    return tf.math.reduce_sum(per_example_loss) / num_process

def init_spec():
    init = utils_tf.data_dicts_to_graphs_tuple([base_graph_np()])
    return utils_tf.specs_from_graphs_tuple(init)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predict_graph = my_module(x, num_process)
        loss = average_loss(y, predict_graph)
    grads = tape.gradient(loss, my_module.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, my_module.trainable_variables)
    )
    return loss

@tf.function
def test_step(x, y):
    predict_graph = my_module(x, num_process)
    return average_loss(y, predict_graph)

@tf.function(input_signature=[init_spec()])
def predict_step(x):
    return my_module(x, num_process)

@tf.function
def valid_step(x, y):
    return average_loss(y, x)

def plot_step(x, y, filename, path, x_offset):
    x = pd.DataFrame(np.array(x[-1].nodes))
    y = pd.DataFrame(np.array(y.nodes))
    plt.figure()
    plt.axis([0, 1, -0.5, 0.5])
    plt.plot(x_offset[0], x[0], linewidth=1, c='r', label='Predict')
    plt.plot(x_offset[0], y[1], linewidth=1, c='b', label='Original')
    plt.legend()
    plt.savefig(plt_path + '{}/{}.png'.format(path, os.path.splitext(np.array(filename)[0].decode())[0]))
    plt.close()
    x.to_csv(plt_path + '{}/{}.csv'.format(path, os.path.splitext(np.array(filename)[0].decode())[0]))

def training():
    ############# options ##############
    batch_size = 1
    epochs = 10
    ############# checkpoint setting ##############
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), module=my_module)
    manager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    ############# dataset ##############
    train_dataset = next_batch.get_dataset('dat/val_p_train.record', batch_size)
    test_dataset = next_batch.get_dataset('dat/val_p_test.record', 1)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % epoch)
        for step, (edges_arrange, val_cpy, filename, _) in enumerate(train_dataset):
            step_pre_batch = len(filename)
            ############## training step ##############
            checkpoint.step.assign_add(1)
            input_tuple = input_graph_to_tuple(edges_arrange, val_cpy, step_pre_batch)
            target_tuple = target_graph_to_tuple(val_cpy, step_pre_batch)
            train_loss = train_step(input_tuple, target_tuple)
            if step % 200 == 0:
                print("Training loss (for %d batch) at step %d: %.8f"
                      % (int(step_pre_batch), step, float(train_loss)),
                      "samples: {}".format(filename[-1]),
                      "lr_rate: {:0.6f}".format(optimizer._decayed_lr(tf.float32).numpy()))

            if int(step) % 1000 == 0:
                ############# save validatoin plot image #############
                edges_arrange_val, val_ycp_val, filename_val, x_offset = next(iter(test_dataset))
                val_input_tuple = input_graph_to_tuple(edges_arrange_val, val_ycp_val, 1)
                val_target_tuple = target_graph_to_tuple(val_ycp_val, 1)
                pre = predict_step(val_input_tuple)
                plot_step(pre, val_target_tuple, filename_val, 'test', x_offset)
                ############# save checkpoint ##############
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            if int(step) % 30000 == 0:
                ############# save model ##############
                to_save = snt.Module()
                to_save.inference = predict_step
                to_save.all_variables = list(my_module.variables)
                tf.saved_model.save(to_save, model_path)
                print("Saved module for step {}".format(int(step)))


def validation():
    ############# load model ##############
    val_batch_size = 1
    loaded = tf.saved_model.load(model_path)
    ############# dataset ##############
    test_dataset = next_batch.get_dataset('dat/val_p_test.record', val_batch_size, is_training=False)
    val_loss = 0
    for step_val, (edges_arrange_val, val_ycp_val, filename_val, x_offset) in enumerate(test_dataset):
        step_pre_batch = len(filename_val)
        ############# validation step ##############
        val_input_tuple = input_graph_to_tuple(edges_arrange_val, val_ycp_val, 1)
        val_target_tuple = target_graph_to_tuple(val_ycp_val, 1)
        pre = loaded.inference(val_input_tuple)
        plot_step(pre, val_target_tuple, filename_val, 'test', x_offset)
        val_loss_per = valid_step(pre, val_target_tuple)
        val_loss += val_loss_per
        # print(step_val, val_loss_per)
        ############## plot offset ##############
    print(step_val)
    print("validation loss: %.8f" % (val_loss / (step_val + 1)))




# training()
validation()