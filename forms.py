from flask_wtf import FlaskForm, RecaptchaField

from flask_wtf import FlaskForm
from wtforms import (StringField,
                     TextAreaField,
                     TextField,
                     SubmitField,
                     PasswordField,
                     DateField,
                     SelectField)
from wtforms.validators import (DataRequired,
                                InputRequired,
                                Email,
                                EqualTo,
                                Length,
                                URL)
class SigninForm(FlaskForm):
    """Signin form."""
    name = StringField('Name', [
        DataRequired()])
    email = StringField('Email', [
        Email(message=('Not a valid email address.')),
        DataRequired()])
    password = PasswordField('Password', [
        DataRequired(message="Please enter a password."),
    ])
    submit = SubmitField('Submit')

class SigninFormStudent(FlaskForm):
    """Signin form."""
    name = StringField('Name', [
        DataRequired()])
    email = StringField('Email', [
        Email(message=('Not a valid email address.')),
        DataRequired()])
    password = PasswordField('Password', [
        DataRequired(message="Please enter a password."),
    ])
    submit = SubmitField('Submit')

class RegFormStudent(FlaskForm):
    """Register form student"""
    rollno = StringField('Roll No', [
        DataRequired()])
    name = StringField('Name', [
        DataRequired()])
    email = StringField('Email', [
        Email(message=('Not a valid email address.')),
        DataRequired()])
    password = PasswordField('Password', [
        DataRequired(message="Please enter a password."),
    ])
    confirmPassword = PasswordField('Repeat Password', [
            InputRequired(),
            EqualTo('password', message='Passwords must match.')
            ])
    submit = SubmitField('Submit')

class SignupForm(FlaskForm):
    """Sign up for a user account."""
    name = StringField('Name', [
        DataRequired()])
    email = StringField('Email', [
        Email(message='Not a valid email address.'),
        DataRequired()])
    password = PasswordField('Password', [
        DataRequired(message="Please enter a password."),
    ])
    confirmPassword = PasswordField('Repeat Password', [
            InputRequired(),
            EqualTo('password', message='Passwords must match.')
            ])
    submit = SubmitField('Submit')

class PassUrl(FlaskForm):
    """Passes url to face recognition algo."""

    url = StringField('Enter URL (dont forget to append "/shot.jpg" after entering http://ipaddress)', [
        DataRequired()])
    lecture = StringField('Lecture Name', [
        DataRequired()])
    submit = SubmitField('Submit')