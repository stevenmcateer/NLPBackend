from flask import Flask, render_template, url_for,flash, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func 

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

question = 'This is a question'
responseCount=0;

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

    submit=False

    form = PostResponse()
    if form.validate_on_submit():
        if form.grade.data:
            grade = 2
            message = 'Your rough estimated score is a 2' 
            temp = questionResponses.query.filter_by(studentId=1, questionId=1 ).all()
            responseId = db.session.query(func.max(questionResponses.responseId)).scalar() + 1

            response = questionResponses(responseId=responseId,studentId=userReference, problemId=problemId,questionId=questionId,response=form.content.data, grade=2)
            db.session.add(response)
            db.session.commit()
        else:
            message = 'Your response has been saved'
            
            responseId = db.session.query(func.max(questionResponses.responseId)).scalar() + 1
            response = questionResponses(responseId=responseId,studentId=userReference, problemId=problemId,questionId=questionId,response=form.content.data)
            db.session.add(response)
            db.session.commit()
            
            submit=True
        flash(message, 'success')
    return render_template('question.html', question=question, form=form, submit=submit)



if __name__ == '__main__':
    app.run()