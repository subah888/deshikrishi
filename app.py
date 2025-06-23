# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 17:56:09 2025
@author: Lenovo
"""

from flask import Flask, render_template, request
import json
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/farmers")
def show_farmers():
    with open("farmers.json", "r") as f:
        farmers = json.load(f)
    return render_template("farmers.html", farmers=farmers)

@app.route("/markets")
def show_markets():
    with open("markets.json", "r") as f:
        markets = json.load(f)
    return render_template("markets.html", markets=markets)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    crop = region = None

    if request.method == "POST":
        crop = request.form["crop"]
        region = request.form["region"]
        area = float(request.form["area"])

        df = pd.read_csv("price_history.csv")
        match = df[(df["crop"] == crop) & (df["region"] == region)]

        if not match.empty:
            prediction = round(match["price_per_kg"].mean(), 2)
        else:
            prediction = "Unavailable"

    return render_template("price.html", prediction=prediction, crop=crop, region=region)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_message = ""
    bot_reply = "হ্যালো! আমি AgriBot। কীভাবে সাহায্য করতে পারি?"

    if request.method == "POST":
        user_message = request.form["message"]

        if "market" in user_message.lower():
            bot_reply = "আপনি কোন এলাকার বাজার খুঁজছেন? আপনি 'Find Markets' ফিচার ব্যবহার করতে পারেন।"
        elif "price" in user_message.lower():
            bot_reply = "দয়া করে ফসলের নাম ও এলাকা বলুন, আমি দাম অনুমান করতে চেষ্টা করব।"
        else:
            bot_reply = "দুঃখিত, আমি বুঝতে পারিনি। আপনি অন্যভাবে বলুন।"

        return render_template("chat.html", user_message=user_message, bot_reply=bot_reply)

    return render_template("chat.html")
@app.route("/offer", methods=["GET", "POST"])
def offer():
    message = None
    if request.method == "POST":
        name = request.form["name"]
        crop = request.form["crop"]
        quantity = request.form["quantity"]
        market = request.form["market"]

        # You can later save to a database or file
        message = f"{name} has offered {quantity} kg of {crop} to {market}!"

    return render_template("offer.html", message=message)

@app.route('/listings')
def listings():
    return render_template('listings.html')

@app.route('/agribot')
def agribot():
    return render_template('agribot.html')


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 17:56:09 2025
@author: Lenovo
"""

from flask import Flask, render_template, request
import json
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/farmers")
def show_farmers():
    with open("farmers.json", "r") as f:
        farmers = json.load(f)
    return render_template("farmers.html", farmers=farmers)

@app.route("/markets")
def show_markets():
    with open("markets.json", "r") as f:
        markets = json.load(f)
    return render_template("markets.html", markets=markets)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    crop = region = None

    if request.method == "POST":
        crop = request.form["crop"]
        region = request.form["region"]
        area = float(request.form["area"])

        df = pd.read_csv("price_history.csv")
        match = df[(df["crop"] == crop) & (df["region"] == region)]

        if not match.empty:
            prediction = round(match["price_per_kg"].mean(), 2)
        else:
            prediction = "Unavailable"

    return render_template("price.html", prediction=prediction, crop=crop, region=region)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_message = ""
    bot_reply = "হ্যালো! আমি AgriBot। কীভাবে সাহায্য করতে পারি?"

    if request.method == "POST":
        user_message = request.form["message"]

        if "market" in user_message.lower():
            bot_reply = "আপনি কোন এলাকার বাজার খুঁজছেন? আপনি 'Find Markets' ফিচার ব্যবহার করতে পারেন।"
        elif "price" in user_message.lower():
            bot_reply = "দয়া করে ফসলের নাম ও এলাকা বলুন, আমি দাম অনুমান করতে চেষ্টা করব।"
        else:
            bot_reply = "দুঃখিত, আমি বুঝতে পারিনি। আপনি অন্যভাবে বলুন।"

        return render_template("chat.html", user_message=user_message, bot_reply=bot_reply)

    return render_template("chat.html")
@app.route("/offer", methods=["GET", "POST"])
def offer():
    message = None
    if request.method == "POST":
        name = request.form["name"]
        crop = request.form["crop"]
        quantity = request.form["quantity"]
        market = request.form["market"]

        # You can later save to a database or file
        message = f"{name} has offered {quantity} kg of {crop} to {market}!"

    return render_template("offer.html", message=message)

@app.route('/listings')
def listings():
    return render_template('listings.html')

@app.route('/agribot')
def agribot():
    return render_template('agribot.html')


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
