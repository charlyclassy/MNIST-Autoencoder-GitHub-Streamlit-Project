from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

# Reproducible results
np.random.seed(42)
tf.random.set_seed(42)
OUTPUT = Path('outputs')
OUTPUT.mkdir(exist_ok=True)

print('Loading MNIST dataset...')
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

print('Building autoencoder with a 32-unit bottleneck...')
inputs = layers.Input(shape=(784,), name='image_pixels')
x = layers.Dense(128, activation='relu')(inputs)
bottleneck = layers.Dense(32, activation='relu', name='bottleneck')(x)
x = layers.Dense(128, activation='relu')(bottleneck)
outputs = layers.Dense(784, activation='sigmoid', name='reconstruction')(x)
autoencoder = Model(inputs, outputs, name='mnist_autoencoder')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print('Training for 3 epochs...')
history = autoencoder.fit(x_train, x_train, epochs=3, batch_size=256,
                          shuffle=True, validation_data=(x_test, x_test), verbose=1)

print('Reconstructing 5 test digits...')
originals = x_test[:5]
reconstructed = autoencoder.predict(originals, verbose=0)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'Original {i+1}')
    axes[1, i].set_title(f'Reconstructed {i+1}')
    axes[0, i].axis('off')
    axes[1, i].axis('off')
fig.tight_layout()
fig.savefig(OUTPUT / 'reconstructions.png', dpi=180, bbox_inches='tight')
plt.close(fig)

plt.figure(figsize=(7, 4))
plt.plot(history.history['loss'], marker='o', label='Training loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy loss')
plt.title('MNIST Autoencoder Reconstruction Loss')
plt.xticks([0, 1, 2], ['1', '2', '3'])
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / 'training_loss.png', dpi=180)
plt.close()

autoencoder.save(OUTPUT / 'mnist_autoencoder.keras')
print('\nTraining complete! Files saved to outputs/:')
print('  reconstructions.png')
print('  training_loss.png')
print('  mnist_autoencoder.keras')
