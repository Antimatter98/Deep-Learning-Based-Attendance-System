from flask import Flask, render_template, flash, request, redirect, url_for
from flask_wtf import FlaskForm
from forms import SigninForm, SignupForm, PassUrl, RegFormStudent
import secrets
import faceRec_flask # this will be your file name; minus the `.py`
import registerFace
import time
from flask_wtf.csrf import CSRFProtect
from flask_pymongo import PyMongo
import bcrypt
import datetime
import pytz

#DEBUG = True
csrf = CSRFProtect()
app = Flask(__name__)

global currentUser

currentUser = {
    'isAuthenticated': False,
    'role': '',
    'name': ''
}

#app.config['RECAPTCHA_PUBLIC_KEY'] = 'iubhiukfgjbkhfvgkdfm'
#app.config['RECAPTCHA_PARAMETERS'] = {'size': '100%'}
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)
app.config['STATIC_FOLDER'] = 'D:\\Facenet-PCN-test\\static'
app.config['TEMPLATES_FOLDER'] = 'D:\\Facenet-PCN-test\\templates'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/dbas'

csrf.init_app(app)

formData=""
url=""

mongo = PyMongo()

mongo.init_app(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html',
                           template='home-template')

@app.route('/signin-admin', methods=('GET', 'POST'))
def signinAdmin():
    #print(currentUser['isAuthenticated'])
    if currentUser['isAuthenticated'] == False:
        form = SigninForm()
        if form.validate_on_submit():
            global formData
            formData = request.form

            name = request.form.getlist('name')
            email = request.form.getlist('email')
            psk = request.form.getlist('password')

            #print(psk)
            if mongo.db.admin.find({
                'name': name[0],
                'email' : email[0]
            }).count() > 0:

                usr = mongo.db.admin.find_one({'name': name[0], 'email': email[0]})
                #for doc in usr:
                if bcrypt.checkpw(psk[0].encode('utf-8'), usr['password']):
                    #print('password: ', usr['password'])
                    currentUser['isAuthenticated'] = True
                    currentUser['name'] = name[0]
                    currentUser['role'] = 'admin'
                    return redirect(url_for('dashboard'))
                else:
                    return redirect(url_for('home'))
            else:
                return redirect(url_for('home'))
        return render_template('signinAdmin.html',
                            form=form,
                            template='form-template')
    else:
        return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    #print(currentUser['name'])
    if ((currentUser['isAuthenticated'] != False) and (currentUser['role'] == 'teacher')):
        tchr = mongo.db.teachers.find_one({'name': currentUser['name']})

        lectures_data = []
        col_names = ["id", "Lecture", "Date", "Time", "Students"]

        lectures = mongo.db.lecture.find({'teacher': tchr["_id"]})
        if lectures.count() > 0:
            for lec in lectures:
                tmp = {}
                student_dt = []
                tmp["id"] = lec["_id"]
                tmp["Lecture"] = lec["lecture"]
                tmp["Date"] = lec["lecture_date"]
                tmp["Time"] = lec["lecture_time"]
                for student in lec["students"]:
                    stdnt = mongo.db.students.find_one({"_id": student})
                    del stdnt['lectures']
                    del stdnt['password']
                    student_dt.append(stdnt)
                tmp['Students'] = student_dt
                lectures_data.append(tmp)

        print("Lecture Data: ", lectures_data)

        return render_template('dashboard.html',
                            template='home-template',
                            lectures_data=lectures_data,
                            col_names=col_names,
                            currentUser=currentUser)
    
    elif ((currentUser['isAuthenticated'] != False) and (currentUser['role'] == 'student')):
        stdnt = mongo.db.students.find_one({'name': currentUser['name']})

        lectures_data = []
        lectures = []
        col_names = ["id", "Lecture", "Date", "Time", "Teacher"]

        all_lectures = mongo.db.lecture.find({})

        for lect in all_lectures:
            for stud in lect['students']:
                if stud == stdnt['_id']:
                    lectures.append(lect)
                    break

        if len(lectures) > 0:
            for lec in lectures:
                tmp = {}
                #student_dt = []
                tmp["id"] = lec["_id"]
                tmp["Lecture"] = lec["lecture"]
                tmp["Date"] = lec["lecture_date"]
                tmp["Time"] = lec["lecture_time"]
                #for student in lec["students"]:
                tchr = mongo.db.teachers.find_one({"_id": lec['teacher']})
                del tchr['lectures']
                del tchr['password']
                    #student_dt.append(stdnt)
                tmp['Teacher'] = tchr
                lectures_data.append(tmp)

        print("Lecture Data: ", lectures_data)

        return render_template('dashboard.html',
                            template='home-template',
                            lectures_data=lectures_data,
                            col_names=col_names,
                            currentUser=currentUser)
    elif ((currentUser['isAuthenticated'] != False) and (currentUser['role'] == 'admin')):
        return render_template('dashboard.html',
                            template='home-template',
                            lectures_data=[],
                            col_names=[],
                            currentUser=currentUser)
    else:
        return redirect(url_for('home'))

@app.route('/signin-student', methods=('GET', 'POST'))
def signinStudent():
    #print(currentUser['isAuthenticated'])
    if currentUser['isAuthenticated'] == False:
        form = SigninForm()
        if form.validate_on_submit():
            global formData
            formData = request.form

            name = request.form.getlist('name')
            email = request.form.getlist('email')
            psk = request.form.getlist('password')

            print(psk)
            if mongo.db.students.find({
                'name': name[0],
                'email' : email[0]
            }).count() > 0:

                usr = mongo.db.students.find_one({'name': name[0], 'email': email[0]})
                #for doc in usr:
                if bcrypt.checkpw(psk[0].encode('utf-8'), usr['password']):
                    #print('password: ', usr['password'])
                    currentUser['isAuthenticated'] = True
                    currentUser['name'] = name[0]
                    currentUser['role'] = 'student'
                    
                    return redirect(url_for('dashboard'))
                else:
                    return redirect(url_for('home'))
            else:
                return redirect(url_for('home'))
        return render_template('signinStudent.html',
                            form=form,
                            template='form-template')
    else:
        return redirect(url_for('success'))

@app.route('/signin', methods=('GET', 'POST'))
def signin():
    #print(currentUser['isAuthenticated'])
    if currentUser['isAuthenticated'] == False:
        form = SigninForm()
        if form.validate_on_submit():
            global formData
            formData = request.form

            name = request.form.getlist('name')
            email = request.form.getlist('email')
            psk = request.form.getlist('password')

            #print(psk)
            if mongo.db.teachers.find({
                'name': name[0],
                'email' : email[0]
            }).count() > 0:

                usr = mongo.db.teachers.find_one({'name': name[0], 'email': email[0]})
                #for doc in usr:
                if bcrypt.checkpw(psk[0].encode('utf-8'), usr['password']):
                    #print('password: ', usr['password'])
                    currentUser['isAuthenticated'] = True
                    currentUser['name'] = name[0]
                    currentUser['role'] = 'teacher'
                    return redirect(url_for('success'))
                else:
                    return redirect(url_for('home'))
            else:
                return redirect(url_for('home'))
        return render_template('signin.html',
                            form=form,
                            template='form-template')
    else:
        return redirect(url_for('success'))

@app.route('/signup', methods=('GET', 'POST'))
def signup():
    if ((currentUser['isAuthenticated'] != False) and (currentUser['role'] == 'admin')):
        form = SignupForm()
        if form.validate_on_submit():
            #print(form)
            name = request.form.getlist('name')
            email = request.form.getlist('email')
            psk = request.form.getlist('password')
            #print(signupData)
            if mongo.db.teachers.find({'email': email[0]}).count() > 0:
                return redirect(url_for('home'))
            else:
                mongo.db.teachers.insert({
                    '_id': email[0],
                    'name': name[0],
                    'email' : email[0],
                    'password': bcrypt.hashpw(psk[0].encode('utf-8'), bcrypt.gensalt()),
                    'lectures': []
                })

                return redirect(url_for('signin'))
        return render_template('signup.html',
                            form=form,
                            template='form-template',
                            currentUser=currentUser,
                            signupfor='teacher')
    else:
        return redirect(url_for('home'))


@app.route('/success', methods=('GET', 'POST'))
def success():
    if currentUser['isAuthenticated'] != False:
        global formData
        urlForm = PassUrl()
        return render_template('success.html',
                            template='success-template', formData=formData, urlForm=urlForm, currentUser=currentUser)
    else:
        return redirect(url_for('home'))

@app.route('/rec', methods=('GET', 'POST'))
def recognize():
    if currentUser['isAuthenticated'] != False:
        global url 
        url = request.form['url']
        lecture = request.form['lecture']

        current_date_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

        current_time = f"{current_date_time.hour}:{current_date_time.minute}"
        current_date = f"{current_date_time.day}/{current_date_time.month}/{current_date_time.year}"

        rec_names =  faceRec_flask.calcEmbedsRec(url)
        #return redirect(url_for('success'))

        #finding rollno list
        roll_no_list = []
        for name in rec_names:
            if mongo.db.students.find({'name': name}).count() > 0:
                usr = mongo.db.students.find_one({'name': name})
                roll_no_list.append(usr['_id'])
        
        #get teacher's details
        tchr = mongo.db.teachers.find_one({"name":currentUser['name']})
        #tchr_lectures = tchr['lectures']

        #create an entry in lectures doc
        _id = mongo.db.lecture.insert({
            '_id': current_date + "-" + current_time,
            'lecture': lecture,
            'teacher': tchr['_id'],
            'students': roll_no_list,
            'lecture_date_time': current_date_time,
            'lecture_date': current_date,
            'lecture_time': current_time
        })

        #append lecture id to teacher doc
        #tchr_lectures.append(_id)
        mongo.db.teachers.update({'name':currentUser['name']}, {"$push":{"lectures":_id}})

        #append lecture id to students doc
        for name in rec_names:
            stdnt = mongo.db.students.update({'name': name}, {"$push":{"lectures":_id}})


        return redirectToSuccess()
    else:
        return redirect(url_for('home'))
def redirectToSuccess():
    #print(mongo.db)
    #mongo.db.students.insert({'name' : 'abc'})
    #print(data)
    return redirect(url_for('success'))

@app.route('/register', methods=('GET', 'POST'))
def RegStudent():
    if ((currentUser['isAuthenticated'] != False) and (currentUser['role'] == 'admin')):
        form = RegFormStudent()
        if form.validate_on_submit():
            #print(form)
            rollno = request.form.getlist('rollno')
            name = request.form.getlist('name')
            email = request.form.getlist('email')
            psk = request.form.getlist('password')
            #print(signupData)
            return register(rollno, name, email, psk)

        return render_template('signup.html',
                            form=form,
                            template='form-template',
                            currentUser=currentUser,
                            signupfor='student')
    else:
        return redirect(url_for('home'))

def register(rollno, name, email, psk):
    if ((currentUser['isAuthenticated'] != False) and (currentUser['role'] == 'admin')):
        _ = registerFace.registerFace(name[0])

        if mongo.db.students.find({'rollno': rollno[0], 'email': email[0]}).count() > 0:
            return redirect(url_for('home'))
        else:
            mongo.db.students.insert({
                '_id': rollno[0],
                'name': name[0],
                'email' : email[0],
                'password': bcrypt.hashpw(psk[0].encode('utf-8'), bcrypt.gensalt()),
                'lectures': []
            })
            return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('home'))

@app.route('/logout', methods=('GET', 'POST'))
def logout():
    currentUser['isAuthenticated'] = False
    currentUser['name'] = ''
    currentUser['role'] = ''
    return redirect(url_for('home'))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port='8000', debug=True)