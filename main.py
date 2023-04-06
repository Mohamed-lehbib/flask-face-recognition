from crypt import methods
from flask import Flask, render_template, Response, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import cv2
import face_recognition
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db= SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = 'thisisasecretkey'
 
UPLOAD_FOLDER = 'student_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getId():
    camera = cv2.VideoCapture(0)
    # Load a sample picture and learn how to recognize it.
    def resize(img, size) :
        width = int(img.shape[1]*size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

    path = 'student_images'
    studentImg = []
    known_face_names = []
    myList = os.listdir(path)
    for cl in myList :
        curimg = cv2.imread(f'{path}/{cl}')
        studentImg.append(curimg)
        known_face_names.append(os.path.splitext(cl)[0])

    def findEncoding(images) :
        known_face_encodings = []
        for img in images :
            img = resize(img, 0.50)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodeimg = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encodeimg)
        return known_face_encodings

    known_face_encodings = findEncoding(studentImg)

    process_this_frame = True

    def gen_frames():
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                        # Only process every other frame of video to save time

                        # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                            # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                            # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                    return face_names
    return gen_frames()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class Etudiant(db.Model):
    matricule = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(20), nullable=False)
    prenom = db.Column(db.String(20), nullable=False)
    filliere = db.Column(db.String(20), nullable=False)
    niveau = db.Column(db.String(20), nullable=False)
    salle = db.Column(db.String(20), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
           


@app.route('/index', methods=['POST', 'GET'])
@login_required
def index():
    for name in getId():
        matricules = name
        etudiants = Etudiant.query.filter_by(matricule = matricules)
    
        return render_template('video.html', etudiants = etudiants)


@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    # Load a sample picture and learn how to recognize it.
    def resize(img, size) :
        width = int(img.shape[1]*size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

    path = 'student_images'
    studentImg = []
    known_face_names = []
    myList = os.listdir(path)
    for cl in myList :
        curimg = cv2.imread(f'{path}/{cl}')
        studentImg.append(curimg)
        known_face_names.append(os.path.splitext(cl)[0])

    def findEncoding(images) :
        known_face_encodings = []
        for img in images :
            img = resize(img, 0.50)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodeimg = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encodeimg)
        return known_face_encodings

    known_face_encodings = findEncoding(studentImg)

    process_this_frame = True

    def gen_frames():
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                        # Only process every other frame of video to save time

                        # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                            # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                            # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                            # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                            # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                            # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/liste')
@login_required
def liste():
    etudiants = Etudiant.query.all()
    return render_template('liste.html', etudiants=etudiants)

@app.route('/insert')
@login_required
def insert():
    return render_template('insert.html')

@app.route('/create', methods=['POST'])
def create():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("Pas d'image selectionner")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(request.form['matricule']+'.'+file.filename.rsplit('.', 1)[1].lower())
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash("l'etudiant a été enregister")
        etudiant = Etudiant(matricule = request.form['matricule'],
        nom = request.form['nom'],
        prenom = request.form['prenom'],
        filliere = request.form['filliere'],
        niveau = request.form['niveau'],
        salle = request.form['salle'])
        db.session.add(etudiant)
        db.session.commit()
        return render_template('insert.html', filename=filename)
    else:
        flash("les types d'image permis  - png, jpg, jpeg, gif")
        return redirect(request.url)
    #return redirect(url_for('liste'))

@app.route('/update/<int:matricule>', methods=['POST', 'GET'])
def update(matricule):
    etudiant = Etudiant.query.get_or_404(matricule) 
    if request.method == "POST":
        etudiant.matricule = request.form['matricule']
        etudiant.nom = request.form['nom']
        etudiant.prenom = request.form['prenom']
        etudiant.filliere = request.form['filliere']
        etudiant.niveau = request.form['niveau']
        etudiant.salle = request.form['salle']
        try:
            db.session.commit()
            return redirect('/liste')
        except:
            return "Problem"
    else:
        return render_template('update.html', etudiant = etudiant)

@app.route('/delete/<matricule>')
def delete(matricule):
    etudiant = Etudiant.query.filter_by(matricule=int(matricule)).delete()
    db.session.commit()
    return redirect(url_for('liste'))


if __name__ == '__main__':
    app.run(debug=True)
