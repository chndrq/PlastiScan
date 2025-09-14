import os
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from werkzeug.utils import secure_filename
from functools import lru_cache

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # for 10MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'} # Format file yang diizinkan

#Cache model untuk mengjindari loading berulang 
@lru_cache(maxsize=None)
def load_cached_model():
    return load_model('model_v3.h5')

model= load_cached_model()
# Define the class labels (ensure the order matches your training)
class_names = ['PET', 'HDPE', 'LDPE', 'PP', 'PS', 'Lainnya', 'Non Plastik']


# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/detect', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    text = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = "Tidak ada file yang diunggah"
        else:
            file = request.files['image']
            if file.filename == '':
                error = "Tidak ada file yang dipilih"
            elif not allowed_file(file.filename):
                error = "Format file tidak didukung. Harap unggah gambar (PNG, JPG, JPEG)"
            else:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    # Preprocessing gambar
                    img = image.load_img(filepath, target_size=(160, 160))  # Ukuran sesuai dengan model yang dilatih
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    # Prediksi
                    pred = model.predict(img_array)[0]
                    confidence_score = np.max(pred)

                    if confidence_score < 0.7:
                        prediction = "Undefined"
                        confidence = None  # atau float, tergantung penanganan di template
                    else:
                        prediction = class_names[np.argmax(pred)]
                        confidence = confidence_score * 100

                        # Teks
                        if prediction == "PET":
                            text = 'Biasa digunakan untuk botol air mineral, jus, dan minuman ringan. Dapat didaur ulang menjadi serat untuk pakaian atau karpet.'
                        elif prediction == "HDPE":
                            text = "Digunakan untuk botol susu, botol deterjen, dan pipa. Dikenal kuat dan tahan terhadap bahan kimia. Dapat didaur ulang menjadi bangku taman atau pot tanaman."
                        elif prediction == "LDPE":
                            text = "Ditemukan pada kantong plastik, plastik pembungkus (wrap), dan botol yang bisa diremas. Dapat didaur ulang menjadi kantong sampah."
                        elif prediction == "PP":
                            text = "Digunakan untuk wadah makanan, tutup botol, dan komponen otomotif. Tahan panas. Dapat didaur ulang menjadi sikat atau baterai mobil."
                        elif prediction == "PS":
                            text = "Dikenal sebagai styrofoam, digunakan untuk cangkir kopi sekali pakai dan kemasan makanan. Sulit didaur ulang."
                        elif prediction == "Lainnya":
                            text = "Plastik jenis lain yang tidak termasuk dalam kategori di atas. Daur ulangnya bervariasi tergantung jenis dan komposisi."
                        elif prediction == "Non Plastik":
                            text = "Bukan plastik. Daur ulangnya tergantung pada jenis material."
                        else:
                            text = None
                except Exception as e:
                    error = f"Terjadi kesalahan saat memproses gambar: {str(e)}"

    return render_template('frontend.html',
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename,
                            text=text,
                           error=error)

@app.route('/', methods=['GET'])
def frontend():
    return render_template('home.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
