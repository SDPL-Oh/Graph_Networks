from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

class load_tfrecord:
    def __init__(self, image_size):
        self.image_size = image_size

    def decode_image(self, image, aoa):
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.resize(image, [*self.image_size], preserve_aspect_ratio=False)
        image = image / 255.0
        aoas = tf.fill([*self.image_size, 1], tf.sparse.to_dense(aoa)[0])
        aoas = aoas / 8.0
        image = tf.concat([image, aoas], axis=2)
        image = tf.cast(image, tf.float32)
        return image

    def read_tfrecord(self, example):
        tfrecord_format = ({
            'image/height': tf.io.FixedLenFeature((), tf.int64),
            'image/width': tf.io.FixedLenFeature((), tf.int64),
            'image/filename': tf.io.FixedLenFeature((), tf.string),
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/format': tf.io.FixedLenFeature((), tf.string),
            'image/object/x': tf.io.VarLenFeature(tf.float32),
            'image/object/y': tf.io.VarLenFeature(tf.float32),
            'image/object/aoa': tf.io.VarLenFeature(tf.float32)
        })
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self.decode_image(example['image/encoded'], example['image/object/aoa'])
        label = tf.sparse.to_dense(tf.sparse.concat(-1, [example['image/object/x'],
                                                         example['image/object/y']]))

        return image, label, example['image/filename']

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
    def __init__(self, image_size, channel, output_size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(*image_size, channel)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(output_size, activation=None)
            ],
            name="vggnet",
        )

    def call(self, inputs):
        return self.logic(inputs)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.plt_path = "dat/plt/conv_net/x/"
        self.mode = 'test'

    def on_predict_batch_end(self, batch, logs=None):
        for i in range(logs['outputs']['predict_output'].shape[0]):
            df_name = os.path.splitext(logs['outputs']['filename'][i].decode())[0]
            df = pd.DataFrame(logs['outputs']['predict_output'][i, :])
            df.to_csv(self.plt_path + '{}/{}.csv'.format(self.mode, df_name))
            df_tar = pd.DataFrame(logs['outputs']['target_output'][i, :])
            df_x = pd.DataFrame(logs['outputs']['x_coord'][i, :])
            plt.figure()
            plt.axis([0, 1, -0.5, 0.5])
            plt.plot(df_x[0], df[0], linewidth=1, c='r', label='Predict')
            plt.plot(df_x[0], df_tar[0], linewidth=1, c='b', label='Original')
            plt.legend()
            plt.savefig(self.plt_path + '{}/{}.png'.format(self.mode, df_name))
            plt.close()
        print('batch: {}'.format(batch))


class Airfoil_CNN(tf.keras.Model):
    def __init__(self, logic, batch_size):
        super(Airfoil_CNN, self).__init__()
        self.batch_size = batch_size
        self.logic = logic

    def compile(self, optimizer, loss_fn):
        super(Airfoil_CNN, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn


    def compute_loss(self, labels, predictions):
        per_example_loss = self.loss_fn(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=self.batch_size)

    def train_step(self, tf_data):
        input_images, target_vector, filename = tf_data
        _, y_vector = tf.split(target_vector, 2, 1)
        with tf.GradientTape() as tape:
            predict_vector = self.logic(input_images, training=True)
            loss = self.compute_loss(y_vector, predict_vector)
        grads = tape.gradient(loss, self.logic.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.logic.trainable_weights)
        )
        return {"loss": loss}

    def test_step(self, tf_data):
        input_images, target_vector, filename = tf_data
        x_vector, y_vector = tf.split(target_vector, 2, 1)
        predict_vector = self.logic(input_images, training=False)
        loss = self.compute_loss(y_vector, predict_vector)
        return {"loss": loss}

    def predict_step(self, tf_data):
        input_images, target_vector, filename = tf_data
        x_vector, y_vector = tf.split(target_vector, 2, 1)
        predict_output = self.logic(input_images, training=False)
        # data_length = tf.cast(tf.divide(predict_output.shape[1], 2), tf.int32)
        # predict_output_t = tf.stack(
        #     [tf.transpose(predict_output)[:data_length],
        #      tf.transpose(predict_output)[data_length:]],
        #     axis=0)
        # target_output_t = tf.stack(
        #     [tf.transpose(y_vector)[:data_length],
        #      tf.transpose(y_vector)[data_length:]],
        #     axis=0)
        return {"predict_output": predict_output,
                "filename": filename,
                'target_output': y_vector,
                'x_coord': x_vector}


class model_train:
    def __init__(self):
        self.image_size = [256, 256]
        self.channel = 2
        self.next_batch = load_tfrecord(self.image_size)
        self.batch_size = 16
        self.epochs = 30
        self.model_path = "models/conv_net/x/"
        self.chk_path = "models/conv_net/x/cp-{epoch:04d}.ckpt"
        self.log_path = "logs/conv_net/x/"


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
        train_dataset = self.next_batch.get_dataset('dat/img_x_train.record', self.batch_size)
        test_dataset = self.next_batch.get_dataset('dat/img_x_test.record', self.batch_size)
        input_images, label_batch, _ = next(iter(train_dataset))

        with mirrored_strategy.scope():
            logic = Custom_model(self.image_size, self.channel, int(label_batch.shape[-1]/2))
            ###################### load model ######################
            # logic = tf.keras.models.load_model(self.model_path)
            ########################################################
            models = Airfoil_CNN(logic, self.batch_size)

        initial_learning_rate = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True)

        models.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
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
        self.batch_size = 1
        test_dataset = self.next_batch.get_dataset('dat/img_x_test.record', self.batch_size)
        logic = tf.keras.models.load_model(self.model_path)
        models = Airfoil_CNN(logic, self.batch_size)
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


# from absl import app
# from absl import flags
# FLAGS = flags.FLAGS
#
# flags.DEFINE_integer('image_size', 256, 'Image Size')
# flags.DEFINE_integer('batch_size', 16, 'Batch Size')
# flags.DEFINE_integer('epochs', 20, 'Number of epochs')
# flags.DEFINE_integer('load_model', True, 'Load Model for Fine Tuning')
# flags.DEFINE_integer('export_mode', 'test', 'Export Data')

# def run_main(argv):
#     del argv
#     kwargs = {'image_size': FLAGS.image_size, 'batch_size': FLAGS.batch_size,
#               'epochs': FLAGS.epochs, 'load_model': FLAGS.load_model,
#               'export_mode': FLAGS.export_mode}
#     main(**kwargs)
#
# def main(image_size, batch_size, epochs, load_model, export_mode):
#     img_size = [image_size, image_size]
#     pix2pix_object = Pix2pix(epochs, enable_function)
#     train_dataset, _ = create_dataset(
#         os.path.join(path_to_folder, 'train/*.jpg'),
#         os.path.join(path_to_folder, 'test/*.jpg'),
#         buffer_size, batch_size)
#     checkpoint_pr = get_checkpoint_prefix()
#     print ('Training ...')
#     return pix2pix_object.train(train_dataset, checkpoint_pr)
#
# if __name__ == '__main__':
#     app.run(run_main)

