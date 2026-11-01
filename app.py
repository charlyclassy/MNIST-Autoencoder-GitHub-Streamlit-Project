"""MNIST Studio: a trained reconstruction autoencoder plus an independent CNN classifier."""
from pathlib import Path
import io
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title='MNIST Reconstruction & Classification Studio',
                   page_icon='🔢', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.stApp,[data-testid="stAppViewContainer"]{background:linear-gradient(130deg,#0B1020,#132849)!important;color:#F8FAFC!important}
[data-testid="stHeader"]{background:#0B1020!important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#152541!important;color:#F8FAFC!important}
[data-testid="stSidebar"] :is(h1,h2,h3,h4,p,label,span,small,li),
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [role="radiogroup"] *{color:#F8FAFC!important;-webkit-text-fill-color:#F8FAFC!important;opacity:1!important}
[data-testid="stMain"] :is(h1,h2,h3,h4){color:#F8FAFC!important}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p{color:#E2E8F0}
[data-testid="stMetric"]{background:#1E304D!important;border:1px solid #4B6082;border-radius:16px;padding:15px}
[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *{color:#FFFFFF!important;opacity:1!important}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#6366F1,#7C3AED)!important;color:#FFF!important;border-radius:12px;border:0}
.stButton>button[kind="primary"] *{color:#FFF!important}
.hero{background:linear-gradient(110deg,#29478A,#5C36A3);border-radius:22px;padding:26px 30px;margin:14px 0 24px}
.hero h1{color:white!important;margin:0;font-size:2.15rem}
.hero p{color:#E0E7FF!important;margin:10px 0 0}
.note{background:#1B2B49;border:1px solid #455B83;border-radius:13px;padding:14px 16px;margin:12px 0}
.note p{margin:0;color:#E2E8F0!important}
</style>''', unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
AUTO_PATHS = [ROOT / 'outputs' / 'mnist_autoencoder.keras', ROOT / 'mnist_autoencoder.keras']
CNN_PATHS = [ROOT / 'outputs' / 'mnist_classifier.keras', ROOT / 'mnist_classifier.keras']


def first_file(paths):
    return next((item for item in paths if item.is_file()), None)


@st.cache_resource(show_spinner='Loading trained neural network...')
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data(show_spinner='Loading MNIST test images...')
def mnist_test():
    (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
    return images.astype('float32') / 255.0, labels


def image_from_upload(upload, invert, center):
    image = Image.open(io.BytesIO(upload.getvalue())).convert('L')
    if invert:
        image = ImageOps.invert(image)
    if center:
        # MNIST-like padding: preserve aspect ratio and place the ink in a 20x20 box.
        pixels = np.asarray(image, dtype=np.uint8)
        mask = pixels > max(20, int(np.percentile(pixels, 75) * 0.25))
        if np.any(mask):
            ys, xs = np.where(mask)
            crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
            w, h = crop.size
            scale = 20.0 / max(w, h)
            crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
            canvas = Image.new('L', (28, 28), 0)
            canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
            image = canvas
        else:
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def reconstruct(model, images):
    source = np.asarray(images, dtype=np.float32).reshape(-1, 784)
    result = model.predict(source, verbose=0)
    return np.clip(np.asarray(result).reshape(-1, 28, 28), 0, 1)


def classify(model, images):
    source = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28, 1)
    probabilities = np.asarray(model.predict(source, verbose=0))
    if probabilities.ndim != 2 or probabilities.shape[1] != 10:
        raise ValueError('Classifier must output 10 digit probabilities.')
    return probabilities


def display_pairs(originals, recons, labels=None):
    for i in range(len(originals)):
        a, b = st.columns(2)
        with a:
            st.image(originals[i], caption=f'Original {i + 1}' + (f' (digit {labels[i]})' if labels is not None else ''), width=180, clamp=True)
        with b:
            st.image(recons[i], caption=f'Reconstructed {i + 1}', width=180, clamp=True)


def warning_missing(name, filename, command):
    st.warning(f'{name} model not found: `{filename}`.')
    st.markdown(f'Upload the **actual trained** `{filename}` file into GitHub `outputs/`, or the repository root. ' +
                f'To generate it in VS Code, run `{command}`. Do not upload an empty `.keras` file.')


auto_path, cnn_path = first_file(AUTO_PATHS), first_file(CNN_PATHS)
with st.sidebar:
    st.markdown('## ⚙️ Project settings')
    st.markdown('**Dataset:** MNIST handwritten digits')
    st.markdown('**Input:** 28 × 28 grayscale')
    st.markdown('**Autoencoder:** 32-unit bottleneck, 3 epochs')
    st.markdown('**Classifier:** independent 10-class CNN')
    st.divider()
    view = st.radio('Choose a view', ['Reconstruct MNIST digits', 'Classify MNIST digits',
                                      'Upload my own digit', 'Training results', 'How it works'])
    st.divider()
    st.caption('Reconstruction and classification are separate neural-network tasks.')

st.markdown('''<div class="hero"><h1>🔢 MNIST AI Studio</h1>
<p>Autoencoder Reconstruction + CNN Digit Classification · TensorFlow / Keras</p></div>''',
            unsafe_allow_html=True)

if view == 'Reconstruct MNIST digits':
    st.subheader('Reconstruct original handwritten digits')
    st.write('See how your three-epoch autoencoder recreates unseen MNIST test images.')
    if not auto_path:
        warning_missing('Autoencoder', 'mnist_autoencoder.keras', 'python train_autoencoder.py')
    else:
        n = st.slider('Number of image pairs', 1, 8, 5)
        if st.button('Reconstruct digits', type='primary'):
            try:
                digits, labels = mnist_test()
                output = reconstruct(load_model(str(auto_path)), digits[:n])
                display_pairs(digits[:n], output, labels[:n])
                mse = float(np.mean((digits[:n] - output) ** 2))
                st.metric('Mean squared reconstruction error (shown samples)', f'{mse:.4f}')
            except Exception as exc:
                st.error(f'Reconstruction failed: {exc}')

elif view == 'Classify MNIST digits':
    st.subheader('Predict digits 0–9 with a separate CNN')
    st.write('The CNN predicts a digit label. The autoencoder reconstructs its appearance.')
    if not cnn_path:
        warning_missing('CNN classifier', 'mnist_classifier.keras', 'python train_classifier.py --epochs 5')
    else:
        n = st.slider('Number of test digits', 1, 10, 5)
        if st.button('Classify digits', type='primary'):
            try:
                digits, labels = mnist_test()
                probs = classify(load_model(str(cnn_path)), digits[:n])
                for i in range(n):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(digits[i], width=140, caption=f'True digit: {int(labels[i])}', clamp=True)
                    with col2:
                        st.markdown(f'**Predicted digit: {int(np.argmax(probs[i]))}**')
                        st.write(f'Confidence: {float(probs[i].max()):.1%}')
                        st.progress(float(probs[i].max()))
            except Exception as exc:
                st.error(f'Classification failed: {exc}')

elif view == 'Upload my own digit':
    st.subheader('Upload a single handwritten digit')
    st.write('Upload a PNG/JPG of one digit. White handwriting on black is MNIST-like. ' +
             'The same prepared image is used by both networks.')
    upload = st.file_uploader('Digit image', type=['png', 'jpg', 'jpeg'])
    invert = st.checkbox('Black handwriting on white background (invert)', value=True)
    center = st.checkbox('Crop, centre and preserve digit shape', value=True)
    if upload is not None:
        try:
            image = image_from_upload(upload, invert, center)
            st.image(image, caption='28 × 28 model input (white strokes, black background)', width=175, clamp=True)
            if st.button('Analyse uploaded digit', type='primary'):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown('#### Autoencoder reconstruction')
                    if auto_path:
                        rec = reconstruct(load_model(str(auto_path)), image[None])[0]
                        st.image(rec, caption='Reconstructed image', width=200, clamp=True)
                        st.metric('Pixel mean squared error', f'{np.mean((image-rec)**2):.4f}')
                    else:
                        st.info('Upload your trained autoencoder to enable reconstruction.')
                with cols[1]:
                    st.markdown('#### CNN classification')
                    if cnn_path:
                        probs = classify(load_model(str(cnn_path)), image[None])[0]
                        st.metric('Predicted digit', int(np.argmax(probs)))
                        st.metric('Model confidence', f'{probs.max():.1%}')
                        st.bar_chart({"Probability": [float(v) for v in probs]})
                    else:
                        st.info('Train and upload your separate classifier to enable digit labels.')
        except Exception as exc:
            st.error(f'Image analysis failed: {exc}')

elif view == 'Training results':
    st.subheader('Training and evaluation evidence')
    st.markdown('#### Autoencoder (3 epochs)')
    for name, caption in [('training_loss.png', 'Training and validation reconstruction loss'),
                          ('reconstructions.png', 'Five original and reconstructed image pairs')]:
        path = ROOT / 'outputs' / name
        if path.is_file():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f'Optional image not found: outputs/{name}')
    st.markdown('#### CNN classifier (separate training)')
    metrics = ROOT / 'outputs' / 'classifier_metrics.json'
    if metrics.is_file():
        report = json.loads(metrics.read_text(encoding='utf-8'))
        st.metric('MNIST test accuracy', f'{report["test_accuracy"]:.2%}')
        st.caption('Measured on the unseen MNIST test set after classifier training.')
    else:
        st.info('Train the classifier and upload its outputs to show its measured test accuracy.')
    for name in ['classifier_training.png', 'classifier_confusion_matrix.png']:
        path = ROOT / 'outputs' / name
        if path.is_file():
            st.image(str(path), use_container_width=True)

else:
    st.subheader('How the two models differ')
    st.markdown('''**Autoencoder (unsupervised reconstruction):** The **encoder** compresses an MNIST image from
    784 pixels into a **32-value bottleneck**. The **decoder** rebuilds the pixels. **Reconstruction loss**
    measures how close the reconstructed image is to the original. It does not predict a digit label.

    **CNN classifier (supervised classification):** Learns from images paired with labels 0–9. It predicts
    the most likely digit and outputs 10 class probabilities. Classification performance is checked on
    an unseen MNIST test set and summarised in a confusion matrix.

    **Important:** Confidence is not a guarantee of correctness. Uploaded images may be harder than MNIST
    test images, and preprocessing can affect both models.''')

st.divider()

