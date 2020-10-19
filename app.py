import numpy as np
import flask
from flask import render_template
import pickle

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('scaler.pkl', 'rb') as sc:
            loaded_scaler = pickle.load(sc)
        with open('model.pkl', 'rb') as md:
            loaded_model = pickle.load(md)
        orders = int(flask.request.form['orders'])
        routeКm = float(flask.request.form['routeКm'])
        routeKG = float(flask.request.form['routeKG'])
        routeМ3 = float(flask.request.form['routeМ3'])
        carryingСapacity = float(flask.request.form['carryingСapacity'])
        routeTime = int(flask.request.form['routeTime'])
        stock = int(flask.request.form['stock'])

        resulr_array = np.array([[orders, routeКm, routeKG, routeМ3, carryingСapacity, routeTime, stock]])
        resulr_array = loaded_scaler.transform(resulr_array)
        prediction = loaded_model.predict(resulr_array)
        result = ''
        if prediction[0] == 0:
            result = 'Не успешным'
        else:
            result = "Успешным"

        return render_template('main.html', result=result)


if __name__ == '__main__':
    app.run()
