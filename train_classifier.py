"""Train a separate MNIST CNN classifier. Does not change the autoencoder weights.
Run: python train_classifier.py --epochs 5
"""
from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Train an MNIST CNN digit classifier.')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be >= 1')
    tf.keras.utils.set_random_seed(42)
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_all = x_train_all.astype('float32')[..., None] / 255.0
    x_test = x_test.astype('float32')[..., None] / 255.0
    # Hold out 10,000 images from training for validation. The official test set is untouched.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(x_train_all))
    val_idx, train_idx = order[:10000], order[10000:]
    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='softmax'),
    ], name='mnist_digit_classifier')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Training dedicated CNN classifier. This does not retrain the autoencoder.')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=args.epochs, batch_size=128, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Unseen MNIST test accuracy: {accuracy * 100:.2f}%')
    model.save(OUTPUT / 'mnist_classifier.keras')
    metrics = {'test_accuracy': float(accuracy), 'test_loss': float(loss),
               'train_samples': int(len(x_train)), 'validation_samples': int(len(x_val)),
               'test_samples': int(len(x_test)), 'epochs': args.epochs}
    (OUTPUT / 'classifier_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    predicted = np.argmax(model.predict(x_test, batch_size=256, verbose=0), axis=1)
    # No scikit-learn dependency needed for confusion matrix.
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (y_test, predicted), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit',
           ylabel='True digit', title=f'CNN Confusion Matrix — MNIST Test Accuracy: {accuracy:.2%}')
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=8,
                    color='white' if matrix[row, col] > matrix.max() / 2 else 'black')
    fig.colorbar(heatmap, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'classifier_confusion_matrix.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history['accuracy'], label='Training')
    ax.plot(history.history['val_accuracy'], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Accuracy', title='CNN Classification Accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'classifier_training.png', dpi=160)
    plt.close(fig)
    print('Saved model, confusion matrix, training chart, and metrics to outputs/')


if __name__ == '__main__':
    main()
