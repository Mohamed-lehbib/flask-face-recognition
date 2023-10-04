# flask-face-recognition

it's a face recognition system coded using opencv and face-recognition library and flask
this is a face recognition system with a crud functions and i have created this app using flask and face_recognition library i have used full-stack flask for rendering templates and connecting to the database.
in this app we have to databases one is for images that are stored in a folder and the other one is an sqlite database used to store the info of the students.

# Process

the process that we use in this app is that i have an openCv library that is know for it use in cases like this for the computer vision so we use the openCV library to open the cam and capture or detect the face using some functions built in openCV. Then we get to the process of searching for the identity of this person we encode the images using the face_recognition functions that is provided by the library and then compare it to the image database when we get the identity of the detected face we query the sqlite database to get the infos of the person recognised. Because of this process the app can have a lack of speed that's to upgrade.

# what u need to setup this project

- step 1: clone the project

```
#to Clone the project
git clone https://github.com/Mohamed-lehbib/flask-face-recognition.git
```

- step 2: create a python virtual environment and use it

```
#create the virtual environment
python3 -m venv env

#for mac
source env/bin/activate

# Activate virtual environment on windows

# From command prompt
dev-env\bin\activate.bat

# From power shell
dev-env\Scripts\Activate.ps1
```

- step 3: install required packages

```
#to install the required packages
pip install -r requirements.txt
```

- step 4: run the project

```
#run the project
python main.py
```

- step 5: after u do ur modification u can add and commit the project

```
#to add
git add .

#to commit the project to ur local repository
git commit -m "commit message"
```

- step 6: create ur own remote repository and get the url of the repository to add it to the project

```
#to setup ur remote repository for this project
git remote add origin <your_repo_url>
```

- step 7: push the project to ur remote repository

```
#to push the project and use "main" instead of "master" if that's your branch name
git push -u origin master
```
