import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics,callbacks
import os
from resnet import resnet18
import datetime





config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


# 图片的预处理
def preprocess(x, y):
    # [-1~1]，将灰度值变换到[-1,1]防止梯度弥散
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

#saver = tf.train.Saver()
#saver.save(sess, 'model.ckpt')
#init = tf.global_variables_initializer()
#saver = tf.train.Saver()


def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    # 查看模型参数
    model.summary()
    # 设置优化器
    optimizer = optimizers.Adam(lr=1e-4)
    # 可视化设置
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    checkpoint_path = "resnet18-{epoch:04d}.ckpt"
    # period=1：每1个epochs 保存一次
    # 用period会报警说已经弃用，建议使用save_freq，save_freq是隔几个batch保存一次模型，period是隔几个epoch保存一次模型
    cp_callback = callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, save_freq=1)


    for epoch in range(500):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model(x)
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth=100)
                # compute loss

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss), step=step)



        # 测试阶段
        total_num = 0
        total_correct = 0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(acc), step=epoch)


if __name__ == '__main__':
    main()
