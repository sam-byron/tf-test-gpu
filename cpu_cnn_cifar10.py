
# Resources
# https://saturncloud.io/blog/how-to-run-tensorflow-on-multiple-cores-and-threads/

import tensorflow as tf
import multiprocessing

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable all GPUS
  tf.config.set_visible_devices([], 'GPU')
  visible_devices = tf.config.get_visible_devices()
  for device in visible_devices:
    assert device.device_type != 'GPU'
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
# # Workers need some inter_ops threads to work properly.
# # This is only needed for this notebook to demo. Real servers
# # should not need this.
# worker_config = tf.compat.v1.ConfigProto()
# worker_config.inter_op_parallelism_threads = 4

# Set the number of CPU threads
num_cpus = multiprocessing.cpu_count()
print(num_cpus)
# num_cpus = 30
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=num_cpus,
    inter_op_parallelism_threads=num_cpus
)
# tf.config.run_functions_eagerly(True)

# Create a new session with the specified configuration
session = tf.compat.v1.Session(config=config)


difficulity=10

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32*difficulity, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64*difficulity, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64*difficulity, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64*difficulity, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], run_eagerly=True)

history = session.run(model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels)))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

session.close()

