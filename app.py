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
    assistmentId = db.Column(db.String(60), nullable=False)
    problemId = db.Column(db.String(60), nullable=False)
    questionId = db.Column(db.String(60), nullable=False)
    attempt = db.Column(db.Integer, nullable=False)
    response = db.Column(db.Text, nullable=False)
    grade = db.Column(db.Integer)

    def __repr__(self):
        return f"questionResponses('{self.problemId}', '{self.questionId}', '{self.response}')"

class students(db.Model):
    studentId = db.Column(db.String(60), primary_key=True, nullable=False)
    problemId = db.Column(db.String(60), primary_key=True, nullable=False)
    group = db.Column(db.String(1)) #0 False in experiement, 1 True in control
    
    def __repr__(self):
        return f"students('{self.studentId}', '{self.group}')"

class problems(db.Model):
    problemId = db.Column(db.String(60), primary_key=True, nullable=False)
    experiementProblemId = db.Column(db.String(60), primary_key=True, nullable=False)

def findGroup(studentId,problemId):
    if not studentId:
        return random.choice([0, 1])

    control = random.choice(['0', '1'])
    student = students.query.filter_by(studentId=studentId,problemId=problemId).all()

    if not student: #Save this students group if first time seen him
        student = students(group=control, studentId=studentId,problemId= problemId)
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

    if not userReference: #Allows testing without assistments
        userReference = 14
        questionId = 10
        problemId = 21
        assignmentReference = 4

    control = findGroup(userReference,problemId)
    submit = False
    form = PostResponse()
    answer = ""
    responseCount = 0

    probExp = problems.query.filter_by(problemId=problemId).first()
    if probExp:
        experiementProblemId = probExp.experiementProblemId

    if form.validate_on_submit():

        if form.grade.data:
            grade = model.calculateGrade(questionId, form.content.data)
            responseCount = len(questionResponses.query.filter_by(studentId=userReference, questionId=questionId,assistmentId = assignmentReference).all()) + 1
            message = 'Your estimated grade is ' + str(grade) + ' out of 4. You have a chance to revise before submitting.'
            responseId = questionResponses.query.count() +1
            response = questionResponses(responseId=responseId,studentId=userReference, problemId=problemId,questionId=questionId, assistmentId = assignmentReference,attempt = responseCount, response=form.content.data, grade=grade)
            db.session.add(response)
            db.session.commit()
 
        else:
            message = 'Your response has been saved! Please hit next problem to continue'
            responseId = questionResponses.query.count() +1
            responseCount = len(questionResponses.query.filter_by(studentId=userReference, questionId=questionId,assistmentId = assignmentReference).all()) + 1
            response = questionResponses(responseId=responseId,studentId=userReference, problemId=problemId,questionId=questionId, assistmentId = assignmentReference,attempt = responseCount,response=form.content.data)
            db.session.add(response)
            db.session.commit()
            submit = True
            if responseCount==1:
                answer = form.content.data.replace('\r\n','<br>')

            else:
                firstResponse = questionResponses.query.filter_by(studentId=userReference, questionId=questionId, problemId=problemId, assistmentId = assignmentReference, attempt=1).scalar()
                firstGrade = firstResponse.grade
                firstResponseStrip = firstResponse.response.replace('\r\n','<br>')
                LastResponseStrip = form.content.data.replace('\r\n','<br>')

                answer = 'Attempt 1: (' + str(firstGrade) +') <br>'+ firstResponseStrip + ' <br><br> Attempt '+ str(responseCount) +':  <br> '+LastResponseStrip 
                answer = '"'+answer+'"'
            #flash (answer, 'success')
        flash(message, 'success')
        
    return render_template('question.html', form=form, submit=submit, grade=(control == 0 and responseCount < 1), answer=answer)



if __name__ == '__main__':
    app.run()