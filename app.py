import csv
import hmac
import hashlib
import io
import json
import os
import secrets
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    abort,
)
import joblib
import numpy as np
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    psycopg2 = None
    DictCursor = None


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-before-deployment")

MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", "house_price_app.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/house_price_app")
USE_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))

model = joblib.load(MODEL_PATH)

class DatabaseConnection:
    def __init__(self):
        self.is_postgres = USE_POSTGRES
        print("Using PostgreSQL:", self.is_postgres)   
        
        if self.is_postgres:
            print("Connecting to PostgreSQL...") 
            if psycopg2 is None:
                raise RuntimeError("PostgreSQL support requires psycopg2-binary. Run pip install -r requirements.txt.")
            self.connection = psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)
        else:
            self.connection = sqlite3.connect(DATABASE_PATH)
            self.connection.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def _query(self, query):
        if self.is_postgres:
            return query.replace("?", "%s")
        return query

    def execute(self, query, params=()):
        cursor = self.connection.cursor()
        cursor.execute(self._query(query), tuple(params))
        return cursor

    def executescript(self, script):
        if not self.is_postgres:
            return self.connection.executescript(script)

        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()


def db_connect():
    return DatabaseConnection()


def now_text():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def init_database():
    with db_connect() as connection:
        connection.executescript(database_schema())

        admin_exists = connection.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
        if not admin_exists:
            admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com").lower()
            admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
            connection.execute(
                """
                INSERT INTO users (name, email, password_hash, role, created_at)
                VALUES (?, ?, ?, 'admin', ?)
                """,
                ("Admin", admin_email, generate_password_hash(admin_password), now_text()),
            )
        connection.commit()
        ensure_prediction_log_columns(connection)


def database_schema():
    if USE_POSTGRES:
        return """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ip_hash TEXT,
                user_agent TEXT,
                location TEXT NOT NULL,
                city TEXT NOT NULL,
                area INTEGER NOT NULL,
                bedrooms INTEGER NOT NULL,
                bathrooms INTEGER NOT NULL,
                stories INTEGER NOT NULL,
                parking INTEGER NOT NULL,
                furnishingstatus TEXT NOT NULL,
                drawing_room INTEGER,
                kitchen_count INTEGER,
                construction_year INTEGER,
                property_age INTEGER,
                subway_access TEXT,
                elevator TEXT,
                predicted_price DOUBLE PRECISION NOT NULL,
                is_custom_location INTEGER NOT NULL DEFAULT 0,
                is_known_location INTEGER NOT NULL DEFAULT 1,
                is_known_city INTEGER NOT NULL DEFAULT 1,
                warning_text TEXT,
                model_name TEXT
            );

            CREATE TABLE IF NOT EXISTS site_activity (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                session_id TEXT NOT NULL,
                page TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """

    return """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ip_hash TEXT,
            user_agent TEXT,
            location TEXT NOT NULL,
            city TEXT NOT NULL,
            area INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            stories INTEGER NOT NULL,
            parking INTEGER NOT NULL,
            furnishingstatus TEXT NOT NULL,
            drawing_room INTEGER,
            kitchen_count INTEGER,
            construction_year INTEGER,
            property_age INTEGER,
            subway_access TEXT,
            elevator TEXT,
            predicted_price REAL NOT NULL,
            is_custom_location INTEGER NOT NULL DEFAULT 0,
            is_known_location INTEGER NOT NULL DEFAULT 1,
            is_known_city INTEGER NOT NULL DEFAULT 1,
            warning_text TEXT,
            model_name TEXT
        );

        CREATE TABLE IF NOT EXISTS site_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            page TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
    """


def ensure_prediction_log_columns(connection):
    if USE_POSTGRES:
        rows = connection.execute(
            """
            SELECT column_name AS name
            FROM information_schema.columns
            WHERE table_name = 'prediction_logs'
            """
        ).fetchall()
    else:
        rows = connection.execute("PRAGMA table_info(prediction_logs)").fetchall()

    existing_columns = {row["name"] for row in rows}
    optional_columns = {
        "drawing_room": "INTEGER",
        "kitchen_count": "INTEGER",
        "construction_year": "INTEGER",
        "property_age": "INTEGER",
        "subway_access": "TEXT",
        "elevator": "TEXT",
    }
    for column, column_type in optional_columns.items():
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE prediction_logs ADD COLUMN {column} {column_type}")
    connection.commit()


def load_metrics():
    if not METRICS_PATH.exists():
        return {
            "known_locations": [],
            "known_cities": [],
            "best_model": "Model metrics not generated yet",
            "best_model_metrics": {},
            "model_comparison": [],
            "feature_importance": [],
        }

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


metrics_data = load_metrics()
locations = metrics_data.get("known_locations", [])
cities = metrics_data.get("known_cities", [])
AMENITY_COLUMNS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

ROLE_PERMISSIONS = {
    "guest": {"predict"},
    "user": {"predict", "view_own_history"},
    "admin": {
        "predict",
        "view_own_history",
        "view_dashboard",
        "view_prediction_logs",
        "export_csv",
        "delete_prediction",
        "view_users",
    },
}


def clean_text(value):
    text = "" if value is None else str(value).strip()
    text = text.replace("\u00e2\u20ac\u201c", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.split())


def split_location(value):
    location = clean_text(value)
    if " - " in location:
        city, locality = location.split(" - ", 1)
    elif "-" in location:
        city, locality = location.split("-", 1)
    else:
        city, locality = "Unknown", location
    return clean_text(city), clean_text(locality)


def get_user_by_id(user_id):
    if not user_id:
        return None
    with db_connect() as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_current_user():
    user = get_user_by_id(session.get("user_id"))
    if not user:
        return SimpleNamespace(is_authenticated=False, is_admin=False, id=None, name="", email="", role="guest")

    return SimpleNamespace(
        is_authenticated=True,
        is_admin=user["role"] == "admin",
        id=user["id"],
        name=user["name"],
        email=user["email"],
        role=user["role"],
    )


def has_permission(permission):
    user = get_current_user()
    role = user.role if user.is_authenticated else "guest"
    return permission in ROLE_PERMISSIONS.get(role, set())


@app.context_processor
def inject_current_user():
    return {
        "current_user": get_current_user(),
        "has_permission": has_permission,
        "csrf_token": csrf_token,
    }


def login_required(view_function):
    @wraps(view_function)
    def wrapper(*args, **kwargs):
        if not get_current_user().is_authenticated:
            flash("Please login to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view_function(*args, **kwargs)

    return wrapper


def permission_required(permission):
    def decorator(view_function):
        @wraps(view_function)
        def wrapper(*args, **kwargs):
            if not has_permission(permission):
                flash("You do not have permission to access this page.", "error")
                return redirect(url_for("home"))
            return view_function(*args, **kwargs)

        return wrapper

    return decorator


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
    return session["session_id"]


def hash_ip(ip_address):
    if not ip_address:
        return None
    salt = app.config["SECRET_KEY"]
    return hashlib.sha256(f"{salt}:{ip_address}".encode("utf-8")).hexdigest()


def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def validate_csrf_token():
    stored_token = session.get("csrf_token")
    submitted_token = request.form.get("csrf_token", "")
    if not stored_token or not hmac.compare_digest(stored_token, submitted_token):
        abort(400, description="Invalid CSRF token.")


def current_model_name():
    return metrics_data.get("best_model", "Unknown model")


@app.before_request
def track_page_view():
    get_session_id()
    if request.endpoint in {None, "static"} or request.method != "GET":
        return

    user = get_current_user()
    with db_connect() as connection:
        connection.execute(
            """
            INSERT INTO site_activity (user_id, session_id, page, action, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user.id, session["session_id"], request.path, "view", now_text()),
        )
        connection.commit()


def render_index(**context):
    return render_template("index.html", locations=locations, cities=cities, **context)


def build_location(form_data):
    selected_location = clean_text(form_data.get("location"))
    custom_location = clean_text(form_data.get("custom_location"))
    custom_city = clean_text(form_data.get("custom_city"))

    if selected_location == "__custom__":
        city = custom_city or "Unknown"
        locality = custom_location or "Unknown"
        location = f"{city} - {locality}"
        return location, city, location in locations, city in cities, True

    city, _ = split_location(selected_location)
    return selected_location, city, selected_location in locations, city in cities, False


def prediction_warning(is_known_location, is_known_city):
    if not is_known_location and is_known_city:
        return (
            "This locality was not present in training data. "
            "Prediction is estimated using city and property features."
        )
    if not is_known_city:
        return (
            "This city and locality were not present in training data. "
            "Prediction may be less reliable."
        )
    return None


def add_prediction_features(data):
    data = data.copy()
    data["total_rooms"] = data["bedrooms"] + data["bathrooms"]
    data["area_per_bedroom"] = data["area"] / data["bedrooms"] if data["bedrooms"] else data["area"]
    data["area_per_room"] = data["area"] / data["total_rooms"] if data["total_rooms"] else data["area"]
    data["amenity_score"] = sum(1 for column in AMENITY_COLUMNS if data.get(column) == "yes")
    data["has_parking"] = 1 if data["parking"] > 0 else 0
    return data


def optional_int(form_data, field_name):
    value = clean_text(form_data.get(field_name))
    return int(value) if value else None


def optional_text(form_data, field_name):
    value = clean_text(form_data.get(field_name))
    return value if value else None


def collect_context_features(form_data):
    return {
        "drawing_room": optional_int(form_data, "drawing_room"),
        "kitchen_count": optional_int(form_data, "kitchen_count"),
        "construction_year": optional_int(form_data, "construction_year"),
        "property_age": optional_int(form_data, "property_age"),
        "subway_access": optional_text(form_data, "subway_access"),
        "elevator": optional_text(form_data, "elevator"),
    }


def build_prediction_features(source_data, city, location):
    """Build the full feature dict expected by the trained model pipeline."""
    context_features = collect_context_features(source_data)
    data = {
        "area": int(source_data["area"]),
        "bedrooms": int(source_data["bedrooms"]),
        "bathrooms": int(source_data["bathrooms"]),
        "stories": int(source_data["stories"]),
        "parking": int(source_data["parking"]),
        "drawing_room": context_features["drawing_room"] if context_features["drawing_room"] is not None else np.nan,
        "kitchen_count": context_features["kitchen_count"] if context_features["kitchen_count"] is not None else np.nan,
        "construction_year": context_features["construction_year"] if context_features["construction_year"] is not None else np.nan,
        "property_age": context_features["property_age"] if context_features["property_age"] is not None else np.nan,
        "mainroad": source_data["mainroad"],
        "guestroom": source_data["guestroom"],
        "basement": source_data["basement"],
        "hotwaterheating": source_data["hotwaterheating"],
        "airconditioning": source_data["airconditioning"],
        "prefarea": source_data["prefarea"],
        "furnishingstatus": source_data["furnishingstatus"],
        "elevator": context_features["elevator"] if context_features["elevator"] is not None else np.nan,
        "subway_access": context_features["subway_access"] if context_features["subway_access"] is not None else np.nan,
        "city": city,
        "location": location,
    }
    return add_prediction_features(data), context_features


@app.route("/")
def home():
    return render_index(form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    location, city, is_known_location, is_known_city, is_custom_location = build_location(form_data)
    data, context_features = build_prediction_features(form_data, city, location)
    prediction = float(model.predict(pd.DataFrame([data]))[0])
    warning_text = prediction_warning(is_known_location, is_known_city)
    user = get_current_user()

    with db_connect() as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                user_id, session_id, timestamp, ip_hash, user_agent, location, city,
                area, bedrooms, bathrooms, stories, parking, furnishingstatus,
                drawing_room, kitchen_count, construction_year, property_age,
                subway_access, elevator,
                predicted_price, is_custom_location, is_known_location,
                is_known_city, warning_text, model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.id,
                get_session_id(),
                now_text(),
                hash_ip(request.headers.get("X-Forwarded-For", request.remote_addr)),
                request.user_agent.string[:255] if request.user_agent else None,
                location,
                city,
                data["area"],
                data["bedrooms"],
                data["bathrooms"],
                data["stories"],
                data["parking"],
                data["furnishingstatus"],
                context_features["drawing_room"],
                context_features["kitchen_count"],
                context_features["construction_year"],
                context_features["property_age"],
                context_features["subway_access"],
                context_features["elevator"],
                prediction,
                int(is_custom_location),
                int(is_known_location),
                int(is_known_city),
                warning_text,
                current_model_name(),
            ),
        )
        connection.commit()

    return render_index(
        prediction_text=f"Rs. {prediction:,.2f}",
        warning_text=warning_text,
        form_data=form_data,
    )


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", metrics=load_metrics())


@app.route("/register", methods=["GET", "POST"])
def register():
    if get_current_user().is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = clean_text(request.form.get("name"))
        email = clean_text(request.form.get("email")).lower()
        password = request.form.get("password", "")

        if not name or not email or len(password) < 6:
            flash("Enter name, email, and a password with at least 6 characters.", "error")
            return render_template("register.html")

        with db_connect() as connection:
            existing = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                flash("An account with this email already exists.", "error")
                return render_template("register.html")

            if USE_POSTGRES:
                cursor = connection.execute(
                    """
                    INSERT INTO users (name, email, password_hash, role, created_at)
                    VALUES (?, ?, ?, 'user', ?)
                    RETURNING id
                    """,
                    (name, email, generate_password_hash(password), now_text()),
                )
                user_id = cursor.fetchone()[0]
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO users (name, email, password_hash, role, created_at)
                    VALUES (?, ?, ?, 'user', ?)
                    """,
                    (name, email, generate_password_hash(password), now_text()),
                )
                user_id = cursor.lastrowid
            connection.commit()
            session["user_id"] = user_id

        flash("Account created successfully.", "success")
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_current_user().is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = clean_text(request.form.get("email")).lower()
        password = request.form.get("password", "")

        with db_connect() as connection:
            user = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session["user_id"] = user["id"]
        flash("Logged in successfully.", "success")
        return redirect(request.args.get("next") or url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


@app.route("/history")
@permission_required("view_own_history")
def history():
    user = get_current_user()
    with db_connect() as connection:
        predictions = connection.execute(
            """
            SELECT * FROM prediction_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            """,
            (user.id,),
        ).fetchall()
    return render_template("history.html", predictions=predictions)


@app.route("/admin/dashboard")
@permission_required("view_dashboard")
def admin_dashboard():
    today = date.today()
    month_start = today.replace(day=1)
    start_day = today - timedelta(days=13)

    with db_connect() as connection:
        total_predictions = connection.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
        predictions_today = connection.execute(
            "SELECT COUNT(*) FROM prediction_logs WHERE timestamp >= ?",
            (today.isoformat(),),
        ).fetchone()[0]
        predictions_month = connection.execute(
            "SELECT COUNT(*) FROM prediction_logs WHERE timestamp >= ?",
            (month_start.isoformat(),),
        ).fetchone()[0]
        registered_users = connection.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0]
        registered_today = connection.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'user' AND created_at >= ?",
            (today.isoformat(),),
        ).fetchone()[0]
        login_users_today = connection.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM site_activity
            WHERE user_id IS NOT NULL AND timestamp >= ?
            """,
            (today.isoformat(),),
        ).fetchone()[0]
        guest_predictions = connection.execute("SELECT COUNT(*) FROM prediction_logs WHERE user_id IS NULL").fetchone()[0]
        custom_locations = connection.execute("SELECT COUNT(*) FROM prediction_logs WHERE is_custom_location = 1").fetchone()[0]
        avg_price = connection.execute("SELECT AVG(predicted_price) FROM prediction_logs").fetchone()[0] or 0
        most_city = connection.execute(
            "SELECT city, COUNT(*) AS total FROM prediction_logs GROUP BY city ORDER BY total DESC LIMIT 1"
        ).fetchone()
        most_location = connection.execute(
            "SELECT location, COUNT(*) AS total FROM prediction_logs GROUP BY location ORDER BY total DESC LIMIT 1"
        ).fetchone()
        daily_rows = connection.execute(
            """
            SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS total
            FROM prediction_logs
            WHERE timestamp >= ?
            GROUP BY substr(timestamp, 1, 10)
            ORDER BY day
            """,
            (start_day.isoformat(),),
        ).fetchall()
        top_locations = connection.execute(
            """
            SELECT location, COUNT(*) AS total
            FROM prediction_logs
            GROUP BY location
            ORDER BY total DESC
            LIMIT 10
            """
        ).fetchall()
        known_count = connection.execute("SELECT COUNT(*) FROM prediction_logs WHERE is_custom_location = 0").fetchone()[0]
        custom_count = connection.execute("SELECT COUNT(*) FROM prediction_logs WHERE is_custom_location = 1").fetchone()[0]
        top_city_prices = connection.execute(
            """
            SELECT city, AVG(predicted_price) AS avg_price
            FROM prediction_logs
            GROUP BY city
            ORDER BY avg_price DESC
            LIMIT 8
            """
        ).fetchall()
        recent_predictions = connection.execute(
            """
            SELECT prediction_logs.*, users.email AS user_email
            FROM prediction_logs
            LEFT JOIN users ON prediction_logs.user_id = users.id
            ORDER BY prediction_logs.timestamp DESC
            LIMIT 8
            """
        ).fetchall()

    daily_map = {row["day"]: row["total"] for row in daily_rows}
    daily_labels = [(start_day + timedelta(days=i)).isoformat() for i in range(14)]

    cards = {
        "total_predictions": total_predictions,
        "predictions_today": predictions_today,
        "predictions_month": predictions_month,
        "registered_users": registered_users,
        "registered_today": registered_today,
        "login_users_today": login_users_today,
        "guest_predictions": guest_predictions,
        "custom_locations": custom_locations,
        "avg_price": avg_price,
        "most_city": most_city["city"] if most_city else "No data",
        "most_location": most_location["location"] if most_location else "No data",
    }
    charts = {
        "daily_labels": daily_labels,
        "daily_values": [daily_map.get(day, 0) for day in daily_labels],
        "top_location_labels": [row["location"] for row in top_locations],
        "top_location_values": [row["total"] for row in top_locations],
        "known_custom_labels": ["Known dropdown locations", "Custom locations"],
        "known_custom_values": [known_count, custom_count],
        "city_price_labels": [row["city"] for row in top_city_prices],
        "city_price_values": [round(float(row["avg_price"] or 0), 2) for row in top_city_prices],
    }

    return render_template(
        "admin_dashboard.html",
        cards=cards,
        charts=charts,
        recent_predictions=recent_predictions,
    )


def filtered_prediction_query():
    clauses = []
    params = []
    search = clean_text(request.args.get("search"))
    city = clean_text(request.args.get("city"))
    user_type = clean_text(request.args.get("user_type"))
    user_id = clean_text(request.args.get("user_id")) 
    date_from = clean_text(request.args.get("date_from"))
    date_to = clean_text(request.args.get("date_to"))

    if search:
        clauses.append("prediction_logs.location LIKE ?")
        params.append(f"%{search}%")
    if city:
        clauses.append("prediction_logs.city = ?")
        params.append(city)
    if user_type == "guest":
        clauses.append("prediction_logs.user_id IS NULL")
    elif user_type == "registered":
        clauses.append("prediction_logs.user_id IS NOT NULL")
    if user_id:
        clauses.append("prediction_logs.user_id = ?")
        params.append(user_id)
    if date_from:
        clauses.append("prediction_logs.timestamp >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("prediction_logs.timestamp < ?")
        params.append((datetime.fromisoformat(date_to) + timedelta(days=1)).strftime("%Y-%m-%d"))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, params


@app.route("/admin/predictions")
@permission_required("view_prediction_logs")
def admin_predictions():
    where_sql, params = filtered_prediction_query()
    with db_connect() as connection:
        predictions = connection.execute(
            f"""
            SELECT prediction_logs.*, users.email AS user_email
            FROM prediction_logs
            LEFT JOIN users ON prediction_logs.user_id = users.id
            {where_sql}
            ORDER BY prediction_logs.timestamp DESC
            LIMIT 250
            """,
            params,
        ).fetchall()
    return render_template(
        "admin_predictions.html",
        predictions=predictions,
        cities=cities,
        filters=request.args,
    )


@app.route("/admin/predictions/<int:prediction_id>/delete", methods=["POST"])
@permission_required("delete_prediction")
def delete_prediction(prediction_id):
    validate_csrf_token()
    with db_connect() as connection:
        connection.execute("DELETE FROM prediction_logs WHERE id = ?", (prediction_id,))
        connection.commit()
    flash("Prediction record deleted successfully.", "success")
    return redirect(url_for("admin_predictions"))


@app.route("/admin/export.csv")
@permission_required("export_csv")
def admin_export_csv():
    where_sql, params = filtered_prediction_query()
    with db_connect() as connection:
        predictions = connection.execute(
            f"""
            SELECT prediction_logs.*, users.email AS user_email
            FROM prediction_logs
            LEFT JOIN users ON prediction_logs.user_id = users.id
            {where_sql}
            ORDER BY prediction_logs.timestamp DESC
            """,
            params,
        ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "timestamp",
            "user_email",
            "session_id",
            "city",
            "location",
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "parking",
            "furnishingstatus",
            "drawing_room",
            "kitchen_count",
            "construction_year",
            "property_age",
            "subway_access",
            "elevator",
            "predicted_price",
            "is_custom_location",
            "warning_text",
            "model_name",
        ]
    )
    for row in predictions:
        writer.writerow(
            [
                row["id"],
                row["timestamp"],
                row["user_email"] or "guest",
                row["session_id"],
                row["city"],
                row["location"],
                row["area"],
                row["bedrooms"],
                row["bathrooms"],
                row["stories"],
                row["parking"],
                row["furnishingstatus"],
                row["drawing_room"] or "",
                row["kitchen_count"] or "",
                row["construction_year"] or "",
                row["property_age"] or "",
                row["subway_access"] or "",
                row["elevator"] or "",
                round(row["predicted_price"], 2),
                bool(row["is_custom_location"]),
                row["warning_text"] or "",
                row["model_name"] or "",
            ]
        )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_logs.csv"},
    )


@app.route("/admin/users")
@permission_required("view_users")
def admin_users():
    with db_connect() as connection:
        users = connection.execute(
            """
            SELECT users.*, COUNT(prediction_logs.id) AS prediction_count
            FROM users
            LEFT JOIN prediction_logs ON users.id = prediction_logs.user_id
            GROUP BY users.id
            ORDER BY users.created_at DESC
            """
        ).fetchall()
    return render_template("admin_users.html", users=users)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    location, city, is_known_location, is_known_city, is_custom_location = build_location(payload)
    data, _ = build_prediction_features(payload, city, location)
    prediction = float(model.predict(pd.DataFrame([data]))[0])
    return {
        "prediction": round(prediction, 2),
        "formatted_prediction": f"Rs. {prediction:,.2f}",
        "city": city,
        "location": location,
        "is_custom_location": is_custom_location,
        "warning": prediction_warning(is_known_location, is_known_city),
        "model": current_model_name(),
    }


init_database()

if __name__ == "__main__":
    init_database()   
    app.run(debug=True)
