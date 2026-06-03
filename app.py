from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import pydoc

import tensorflow as tf
import numpy as np
import os

from datetime import datetime

# PDF Libraries
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    HRFlowable
)

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

# ─────────────────────────────────────────────
# Flask App Config
# ─────────────────────────────────────────────

app = Flask(__name__)

app.secret_key = 'bloodgroup_secret_2024'

# Upload Folders

UPLOAD_FOLDER = 'static/uploads'
PDF_FOLDER = 'static/reports'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────

MODEL_PATH = 'blood_group_model_girls_augmented.keras'

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("✅ Model loaded successfully")

except Exception as e:

    print("❌ Error loading model:", e)

    model = None

# ─────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────

LABELS = {
    0: 'A+',
    1: 'AB+',
    2: 'B+',
    3: 'O+'
}

IMG_SIZE = (256, 256)

# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────

def preprocess_image(filepath):

    img = tf.keras.preprocessing.image.load_img(
        filepath,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ─────────────────────────────────────────────
# Generate PDF Report
# ─────────────────────────────────────────────

def generate_pdf_report(
    name,
    age,
    phone,
    address,
    blood_group,
    confidence,
    image_path,
    date
):

    pdf_filename = f"{name.replace(' ','_')}_report.pdf"

    pdf_path = os.path.join(
        PDF_FOLDER,
        pdf_filename
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=20,
        bottomMargin=30
    )

    elements = []

    # ─────────────────────────────────────
    # STYLES
    # ─────────────────────────────────────

    title_style = ParagraphStyle(
        'title',
        fontName='Times-Bold',
        fontSize=20,
        alignment=TA_CENTER,
        leading=30
    )

    heading_style = ParagraphStyle(
        'heading',
        fontName='Times-Bold',
        fontSize=14,
        leading=24
    )

    normal_style = ParagraphStyle(
        'normal',
        fontName='Times-Roman',
        fontSize=12,
        leading=22
    )

    # ─────────────────────────────────────
    # HEADER IMAGE
    # ─────────────────────────────────────

    header = Image(
        "static/images/college_header.png",
        width=520,
        height=100
    )

    elements.append(header)

    elements.append(Spacer(1,20))

    # ─────────────────────────────────────
    # REPORT TITLE
    # ─────────────────────────────────────

    elements.append(
        Paragraph(
            "Female Blood Group Detection Report",
            title_style
        )
    )

    elements.append(Spacer(1,20))

    # ─────────────────────────────────────
    # PATIENT DETAILS
    # ─────────────────────────────────────

    elements.append(
        Paragraph(
            "PATIENT DETAILS",
            heading_style
        )
    )

    elements.append(Spacer(1,10))

    patient_info = f"""
    <b>Patient Name :</b> {name}<br/>
    <b>Age / Gender :</b> {age} Years / Female<br/>
    <b>Contact Number :</b> {phone}<br/>
    <b>Address :</b> {address}<br/>
    <b>Report Generated On :</b> {date}<br/>
    """

    elements.append(
        Paragraph(
            patient_info,
            normal_style
        )
    )

    elements.append(Spacer(1,20))

    # ─────────────────────────────────────
    # ANALYSIS RESULT
    # ─────────────────────────────────────

    elements.append(
        Paragraph(
            "FINGERPRINT ANALYSIS RESULT",
            heading_style
        )
    )

    elements.append(Spacer(1,15))

    prediction_text = f"""
    <b>Detected Blood Group :</b> {blood_group}<br/>
    <b>Prediction Confidence :</b> {confidence}%<br/>
    <b>Detection Method :</b> AI Powered Deep Learning Model (CNN)<br/>
    """

    elements.append(
        Paragraph(
            prediction_text,
            normal_style
        )
    )

    elements.append(Spacer(1,25))

    # ─────────────────────────────────────
    # FINGERPRINT IMAGE
    # ─────────────────────────────────────

    elements.append(
        Paragraph(
            "FINGERPRINT SAMPLE",
            heading_style
        )
    )

    elements.append(Spacer(1,15))

    fingerprint = Image(
        image_path,
        width=80,
        height=80
    )

    elements.append(fingerprint)

    elements.append(Spacer(1,25))

    # ─────────────────────────────────────
    # INTERPRETATION
    # ─────────────────────────────────────

    elements.append(
        Paragraph(
            "INTERPRETATION",
            heading_style
        )
    )

    elements.append(Spacer(1,10))

    interpretation = f"""
    The Female Blood Group Detection system utilizes Artificial Intelligence
    and Deep Learning techniques to analyze fingerprint patterns and predict
    the probable blood group of an individual.

    Based on the uploaded fingerprint sample, the system has predicted the
    blood group as <b>{blood_group}</b> with a confidence score of
    <b>{confidence}%</b>.

    This prediction is intended for academic research and educational
    purposes and should not be considered a substitute for laboratory blood
    group testing. Clinical blood typing methods remain the most reliable
    approach for determining an individual's blood group.
    """

    elements.append(
        Paragraph(
            interpretation,
            normal_style
        )
    )

    elements.append(Spacer(1,20))

    # ─────────────────────────────────────
    # PROJECT CONTRIBUTORS
    # ─────────────────────────────────────

    elements.append(
        Paragraph(
            "PROJECT CONTRIBUTORS",
            heading_style
        )
    )

    elements.append(Spacer(1,10))

    contributors = """
    <b>Payal Babar</b> — Student of AI & DS<br/>
    <b>Trupti Babar</b> — Student of AI & DS
    """

    elements.append(
        Paragraph(
            contributors,
            normal_style
        )
    )

    elements.append(Spacer(1,25))


    # ─────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────

    elements.append(
        HRFlowable(width="100%")
    )

    elements.append(Spacer(1,20))

    end_text = Paragraph(
        "<b>*** END OF REPORT ***</b>",
        ParagraphStyle(
            'end',
            fontName='Times-Bold',
            fontSize=14,
            alignment=TA_CENTER
        )
    )

    elements.append(end_text)

    # Build PDF

    doc.build(elements)

    return pdf_filename

# ─────────────────────────────────────────────
# Login Route
# ─────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form.get('email')

        password = request.form.get('password')

        if email and password:

            session['user'] = email

            return redirect(url_for('home'))

    return render_template('login.html')

# ─────────────────────────────────────────────
# Logout Route
# ─────────────────────────────────────────────

@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for('login'))

# ─────────────────────────────────────────────
# Home Page
# ─────────────────────────────────────────────

@app.route('/home')
def home():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('home.html')

# ─────────────────────────────────────────────
# About Page
# ─────────────────────────────────────────────

@app.route('/about')
def about():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('about.html')

# ─────────────────────────────────────────────
# Detection Page
# ─────────────────────────────────────────────

@app.route('/detect')
def detect():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('detect.html')

# ─────────────────────────────────────────────
# Prediction Route
# ─────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():

    if 'user' not in session:
        return redirect(url_for('login'))

    try:

        # Form Data

        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        address = request.form['address']

        # Upload Fingerprint

        file = request.files.get('fingerprint')

        if file is None or file.filename == '':

            return render_template(
                'detect.html',
                error='Please upload fingerprint image'
            )

        # Save File

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        # Preprocess Image

        processed_image = preprocess_image(filepath)

        # Prediction

        prediction = model.predict(processed_image)

        pred_class = int(np.argmax(prediction))

        blood_group = LABELS.get(
            pred_class,
            'Unknown'
        )

        confidence = round(
            float(prediction[0][pred_class]) * 100,
            2
        )

        current_date = datetime.now().strftime(
            '%d-%m-%Y %H:%M:%S'
        )

        # Generate PDF

        pdf_file = generate_pdf_report(
            name=name,
            age=age,
            phone=phone,
            address=address,
            blood_group=blood_group,
            confidence=confidence,
            image_path=filepath,
            date=current_date
        )

        # Render Report Page

        return render_template(
            'report.html',
            name=name,
            age=age,
            phone=phone,
            address=address,
            blood_group=blood_group,
            confidence=confidence,
            image_file=filename,
            pdf_file=pdf_file,
            date=current_date
        )

    except Exception as e:

        print("FULL ERROR:", e)

        return render_template(
            'detect.html',
            error=f'Prediction Error: {str(e)}'
        )

# ─────────────────────────────────────────────
# Run Flask App
# ─────────────────────────────────────────────

if __name__ == '__main__':

    app.run(debug=True)
