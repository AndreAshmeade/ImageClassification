import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

gpus = tf.config.experimental.list_physical_devices('CPU');
#Avoid OOM errors by setting Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU");
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'svg', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_exts))
                os.remove(image_path)
        except Exception as e:
                print('Issue with image {}'.format(image_path))

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#Images represents as numpy arrays
print(batch[0].shape)

#Scale Data
data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

#Split Data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2) + 1
test_size = int(len(data)*.1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Deep Model
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation="relu"))

model.add(Flatten())

model.add(Dense(256, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary();

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Test
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

pic = cv2.imread('./test/test_sample.jpg')
plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(pic, (256, 256))
yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)

if yhat > 0.5:
    print(f'Predicated class is cat')
else:
    print(f'Predicated class is dog')

#Save Model
model.save(os.path.join('models', 'catdogmodel.h5'))
new_model = load_model(os.path.join('models', 'catdogmodel.h5'))
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

if yhatnew > 0.5:
    print(f'Predicated class is cat')
else:
    print(f'Predicated class is dog')











