from flask import Flask, render_template, url_for,flash
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

question = 'This is a question'
      
class PostResponse(FlaskForm):
    content = TextAreaField('Answer', validators=[DataRequired()])
    grade = SubmitField("Grade Question")
    submit = SubmitField('Submit')
       
        
@app.route('/' , methods=['GET', 'POST'])

def home():
    form = PostResponse()
    if form.validate_on_submit():
        if form.grade.data:
            message = 'Your rough estimated score is a 2' 
        else:
            message = 'Your response has been saved'
            form.content.data = ""
        flash(message, 'success')
    return render_template('question.html', question=question, form=form)



if __name__ == '__main__':
    app.run()