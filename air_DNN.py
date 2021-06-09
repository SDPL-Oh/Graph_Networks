import os
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt


mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

class load_tfrecord:
    def norm_sparse(self, cp, aoa):
        cp = cp / 10.0
        cp = tf.sparse.to_dense(cp)
        aoa = aoa / 8.0
        aoa = tf.sparse.to_dense(aoa)
        return cp, aoa

    def read_tfrecord(self, example):
        tfrecord_format = ({
            'image/filename': tf.io.FixedLenFeature((), tf.string),
            'image/object/x': tf.io.VarLenFeature(tf.float32),
            'image/object/y': tf.io.VarLenFeature(tf.float32),
            'image/object/Cp': tf.io.VarLenFeature(tf.float32),
            'image/object/aoa': tf.io.VarLenFeature(tf.float32)
        })
        example = tf.io.parse_single_example(example, tfrecord_format)
        input_cp, input_aoa = self.norm_sparse(example['image/object/Cp'],
                                               example['image/object/aoa'])
        input_x = tf.sparse.to_dense(example['image/object/x'])
        output_y = tf.sparse.to_dense(example['image/object/y'])
        return input_x, input_cp, input_aoa, output_y, example['image/filename']

    def load_dataset(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self, filenames, batch_size):
        dataset = self.load_dataset(filenames)
        dataset = dataset.shuffle(1000)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset


class Custom_model(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(input_size)),
                tf.keras.layers.Dense(160),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(160),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(160),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(output_size, activation=None)
            ],
            name="Densenet",
        )

    def call(self, inputs):
        return self.logic(inputs)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.plt_path = "dat/plt/dense_net/random/"
        self.mode = 'test'

    def on_predict_batch_end(self, batch, logs=None):
        for i in range(logs['outputs']['predict_output'].shape[0]):
            df_x = pd.DataFrame(logs['outputs']['input_x'][i, :])
            df_name = os.path.splitext(logs['outputs']['filename'][i].decode())[0]
            df = pd.DataFrame(logs['outputs']['predict_output'][i, :])
            df.to_csv(self.plt_path + '{}/{}.csv'.format(self.mode, df_name))
            df_tar = pd.DataFrame(logs['outputs']['target_output'][i, :])

            plt.figure()
            plt.axis([0, 1, -0.5, 0.5])
            plt.plot(df_x[0], df[0], linewidth=1, c='r', label='Predict')
            plt.plot(df_x[0], df_tar[0], linewidth=1, c='b', label='Original')
            plt.legend()
            plt.savefig(self.plt_path + '{}/{}.png'.format(self.mode, df_name))
            plt.close()
        print('batch: {}'.format(batch))


class Airfoil_DNN(tf.keras.Model):
    def __init__(self, logic, batch_size):
        super(Airfoil_DNN, self).__init__()
        self.batch_size = batch_size
        self.logic = logic

    def compile(self, optimizer, loss_fn):
        super(Airfoil_DNN, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn


    def compute_loss(self, labels, predictions):
        per_example_loss = self.loss_fn(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=self.batch_size)

    def train_step(self, tf_data):
        input_x, input_cp, input_aoa, output_y, filename = tf_data
        inputs = tf.concat([input_cp, input_aoa[:, 0:1]], axis=1)
        with tf.GradientTape() as tape:
            predict_vector = self.logic(inputs, training=True)
            loss = self.compute_loss(output_y, predict_vector)
        grads = tape.gradient(loss, self.logic.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.logic.trainable_weights)
        )
        return {"loss": loss}

    def test_step(self, tf_data):
        input_x, input_cp, input_aoa, output_y, filename = tf_data
        inputs = tf.concat([input_cp, input_aoa[:, 0:1]], axis=1)
        predict_vector = self.logic(inputs, training=False)
        loss = self.compute_loss(output_y, predict_vector)
        return {"loss": loss}

    def predict_step(self, tf_data):
        input_x, input_cp, input_aoa, output_y, filename = tf_data
        inputs = tf.concat([input_cp, input_aoa[:, 0:1]], axis=1)
        predict_vector = self.logic(inputs, training=False)
        return {"predict_output": predict_vector,
                "filename": filename,
                "target_output": output_y,
                "input_x": input_x}


class model_train:
    def __init__(self):
        self.next_batch = load_tfrecord()
        self.batch_size = 16
        self.epochs = 50
        self.model_path = "models/dense_net/random/"
        self.chk_path = "models/dense_net/random/cp-{epoch:04d}.ckpt"
        self.log_path = "logs/dense_net/random/"


    def callbacks(self):
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            self.log_path, histogram_freq=100
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.chk_path,
            save_weights_only=True,
            verbose=1,
            period=5)
        return tensorboard_cb, checkpoint_cb

    def run(self):
        train_dataset = self.next_batch.get_dataset('dat/val_p_train.record', self.batch_size)
        test_dataset = self.next_batch.get_dataset('dat/val_p_test.record', self.batch_size)
        _, input_cp, input_aoa, output_y, _ = next(iter(train_dataset))

        with mirrored_strategy.scope():
            logic = Custom_model(input_cp.shape[1] + 1, output_y.shape[1])
            ###################### load model ######################
            # logic = tf.keras.models.load_model(self.model_path)
            ########################################################
            models = Airfoil_DNN(logic, self.batch_size)

        models.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        )
        models.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=test_dataset,
            callbacks=[self.callbacks()]
        )
        logic.save(self.model_path)

    def test_run(self):
        test_dataset = self.next_batch.get_dataset('dat/val_test.record', 1)
        logic = tf.keras.models.load_model(self.model_path)
        models = Airfoil_DNN(logic, self.batch_size)
        models.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        )

        # loss = models.evaluate(test_dataset)
        # print("Mean Square Error: {}".format(loss))
        ############## export images ###############
        models.predict(test_dataset,
                       callbacks=[CustomCallback()])


a = model_train()
# a.run()
a.test_run()
