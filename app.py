from flask import Flask, render_template, url_for,flash, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func 
from numpy import random
import model

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class questionResponses(db.Model):
    responseId = db.Column(db.Integer, primary_key=True)
    studentId = db.Column(db.String(60), nullable=False)
    problemId = db.Column(db.String(60), nullable=False)
    questionId = db.Column(db.String(60), nullable=False)
    #attempt = db.Column(db.Integer, nullable=False)
    response = db.Column(db.Text, nullable=False)
    grade = db.Column(db.Integer)

    def __repr__(self):
        return f"questionResponses('{self.problemId}', '{self.questionId}', '{self.response}')"

class students(db.Model):
    studentId = db.Column(db.String(60), primary_key=True, nullable=False)
    group = db.Column(db.String(60)) #0 False in experiement, 1 True in control
    
    def __repr__(self):
        return f"students('{self.studentId}', '{self.group}')"

def findGroup(studentId):
    if not studentId:
        return random.choice([0, 1])

    control = random.choice(['0', '1'])
    student = students.query.filter_by(studentId=studentId).all()

    if not student: #Save this students group if first time seen him
        student = students(group=control, studentId=studentId)
        db.session.add(student)
        db.session.commit()
    else: #Get this students group
        control = student[0].group

    return control

class PostResponse(FlaskForm):
    content = TextAreaField('Answer', validators=[DataRequired()])
    grade = SubmitField("Grade Question")
    submit = SubmitField('Submit')

        
@app.route('/' , methods=['GET', 'POST'])

def home():
#p1= User Reference (uuid)
#p2= Class Reference (uuid)
#p3= Assignment Reference (uuid)
#p4= Problem ID Decoded (id) [formerly called “Assistment ID” in this doc]
#p5= Question ID (id) [formerly called “Problem ID” in this doc]
#p6= User Name (if your domain registration request for this item was approved)
#p7= ASSISTments System Reference (uuid)


    userReference = request.args.get('p1')
    classReference = request.args.get('p2')
    assignmentReference = request.args.get('p3')
    problemId = request.args.get('p4')
    questionId = request.args.get('p5')

    control = findGroup(userReference)
    submit = False
    form = PostResponse()

    if form.validate_on_submit():
        if form.grade.data:
            grade = model.calculateGrade(questionId)
            message = 'Your rough estimated score is a ' + str(grade)
           # Need to store this into into a db for each student/question 
           # message2 =  'This is attempt #: ' +  str(responseCount)
           # flash(message2, 'success')
           # responseCount = responseCount + 1
           # temp = questionResponses.query.filter_by(studentId=1, questionId=1 ).all()
            if(problemId):
                responseId = db.session.query(func.max(questionResponses.responseId)).scalar() + 1
                response = questionResponses(responseId=responseId,studentId=userReference, problemId=problemId,questionId=questionId,response=form.content.data, grade=grade)
                db.session.add(response)
                db.session.commit()

        else:
            message = 'Your response has been saved! Please hit next problem to continue'
            if(problemId):
                responseId = db.session.query(func.max(questionResponses.responseId)).scalar() + 1
                response = questionResponses(responseId=responseId,studentId=userReference, problemId=problemId,questionId=questionId,response=form.content.data)
                db.session.add(response)
                db.session.commit()
            submit = True

        flash(message, 'success')
    return render_template('question.html', question='Your question: ' + str(questionId), form=form, submit=submit, control=control)



if __name__ == '__main__':
    app.run()