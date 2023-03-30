from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import os

app = Flask(__name__)

# define the route for the quiz form
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == 'admin' and password == '123456':
        return redirect(url_for('success'))
    else:
        error = 'Invalid username or password. Please try again.'
        return render_template('login.html', error=error)

@app.route('/success')
def success():
    return render_template('detectionPage.html')