from flask import Flask, jsonify, render_template, request, redirect, url_for, session
import pandas as pd

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

ATTENDANCE_FILE = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"

# Hardcoded credentials (academic use)
USERNAME = "admin"
PASSWORD = "admin123"

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/attendance")
def get_attendance():
    if not session.get("logged_in"):
        return jsonify([])

    df = pd.read_csv(ATTENDANCE_FILE)
    return jsonify(df.to_dict(orient="records"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
