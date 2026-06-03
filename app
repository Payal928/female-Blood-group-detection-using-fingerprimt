from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import numpy as np
import os
from datetime import datetime
 
app = Flask(__name__)
app.secret_key = 'bloodgroup_secret_2024'
 
# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# ── Model loading (version-safe) ──────────────────────────────────────────────
model = None
 
def load_model_safe():
    global model
 
    # Strategy 1: Try the .keras format first (most portable)
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            'blood_group_model_girls_augmented.keras',
            compile=False
        )
        print("Loaded blood_group_model_girls_augmented.keras")
        return
    except Exception as e:
        print(f"  .keras load failed: {e}")
 
    # Strategy 2: Patch the .h5 to strip the unsupported 'quantization_config' key
    try:
        import tensorflow as tf
        import h5py, json
 
        h5_path = 'bld_detection_augmented_girls.h5'
 
        with h5py.File(h5_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            config = json.loads(model_config)
 
        # Recursively remove 'quantization_config' from all layer configs
        def strip_quant(cfg):
            if isinstance(cfg, dict):
                cfg.pop('quantization_config', None)
                for v in cfg.values():
                    strip_quant(v)
            elif isinstance(cfg, list):
                for item in cfg:
                    strip_quant(item)
 
        strip_quant(config)
 
        model = tf.keras.models.model_from_json(json.dumps(config))
        model.load_weights(h5_path)
        print("Loaded bld_detection_augmented_girls.h5 (patched)")
        return
    except Exception as e:
        print(f"  Patched .h5 load failed: {e}")
 
    # Strategy 3: Plain load_model as last resort
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            'bld_detection_augmented_girls.h5',
            compile=False
        )
        print("Loaded bld_detection_augmented_girls.h5 (plain)")
    except Exception as e:
        print(f"All model loading strategies failed: {e}")
        raise
 
load_model_safe()
 
# Blood group labels — order must match your training classes
LABELS = {0: 'A', 1: 'AB', 2: 'B', 3: 'O'}
IMG_SIZE = (256, 256)
 
# ── Image preprocessing helper ────────────────────────────────────────────────
def preprocess_image(filepath):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    x   = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)
 
# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        # Demo credentials — replace with a real auth system for production
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('home'))
        error = 'Invalid credentials. Try admin / admin123.'
    return render_template('login.html', error=error)
 
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
 
@app.route('/home')
def home():
    return render_template('home.html')
 
@app.route('/about')
def about():
    return render_template('about.html')
 
@app.route('/detect')
def detect():
    return render_template('detect.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        name    = request.form['name']
        age     = request.form['age']
        phone   = request.form['phone']
        address = request.form['address']
 
        file = request.files.get('fingerprint')
        if not file or file.filename == '':
            return render_template('detect.html', error='Please upload a fingerprint image.')
 
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
 
        x          = preprocess_image(filepath)
        prediction = model.predict(x)
        pred_class = int(np.argmax(prediction))
        label      = LABELS.get(pred_class, 'Unknown')
        confidence = round(float(prediction[0][pred_class]) * 100, 2)
 
        return render_template(
            'report.html',
            name=name, age=age, phone=phone, address=address,
            blood_group=label,
            confidence=confidence,
            image_file=filename,
            date=datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        )
    except Exception as e:
        return render_template('detect.html', error=f'Prediction error: {str(e)}')
 
if __name__ == '__main__':
    app.run(debug=True)
 