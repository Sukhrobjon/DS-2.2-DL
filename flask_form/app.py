import math

from flask import Flask, render_template, request
from wtforms import BooleanField, Form, IntegerField, StringField, validators


class Cylinder(Form):
    radius = IntegerField("Radius", [])
    height = IntegerField("Height", [])


app = Flask(__name__)


@app.route("/form", methods=["GET", "POST"])
def get_form():
    form = Cylinder(request.form)
    if request.method == "GET":
        return render_template("cylinder.html", form=form)
    else:
        radius = form.radius.data
        height = form.height.data
        return str(math.pi * (radius ** 2) * height)

if __name__ == "__main":
    app.run("0.0.0.0", port="5000")
