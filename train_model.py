import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("house_prices_data.csv")
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")
CHART_DIR = Path("static") / "charts"

NUMERIC_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "drawing_room",
    "kitchen_count",
    "construction_year",
    "property_age",
    "total_rooms",
    "area_per_bedroom",
    "area_per_room",
    "amenity_score",
    "has_parking",
]
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
    "elevator",
    "subway_access",
    "city",
    "location",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
AMENITY_COLUMNS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]


def clean_text(value):
    text = "" if pd.isna(value) else str(value).strip()
    text = text.replace("\u00e2\u20ac\u201c", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = " ".join(text.split())
    return text


def split_location(value):
    location = clean_text(value)
    if " - " in location:
        city, locality = location.split(" - ", 1)
    elif "-" in location:
        city, locality = location.split("-", 1)
    else:
        city, locality = "Unknown", location
    return clean_text(city), clean_text(locality)


def add_derived_features(df):
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["area_per_bedroom"] = df["area"] / df["bedrooms"].replace(0, np.nan)
    df["area_per_room"] = df["area"] / df["total_rooms"].replace(0, np.nan)
    df["amenity_score"] = sum((df[col] == "yes").astype(int) for col in AMENITY_COLUMNS)
    df["has_parking"] = (df["parking"] > 0).astype(int)
    return df


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(regressor):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    percentage_error = np.abs((y_true - y_pred) / y_true) * 100
    return {
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(math.sqrt(mse)), 2),
        "mape_percent": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
        "accuracy_within_10_percent": round(float(np.mean(percentage_error <= 10) * 100), 2),
        "accuracy_within_20_percent": round(float(np.mean(percentage_error <= 20) * 100), 2),
    }


def get_feature_importance(best_pipeline, X_test, y_test):
    model = best_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        result = permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="r2",
        )
        rows = [
            {
                "feature": feature,
                "importance": round(float(importance), 5),
            }
            for feature, importance in zip(X_test.columns, result.importances_mean)
        ]
        rows.sort(key=lambda item: item["importance"], reverse=True)
        return rows[:15]

    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    rows = []

    for name, value in zip(feature_names, importances):
        clean_name = name.replace("num__", "").replace("cat__", "")
        rows.append(
            {
                "feature": clean_name,
                "importance": round(float(value), 5),
            }
        )

    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows[:15]


def svg_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_actual_vs_predicted_chart(y_true, y_pred):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    left, right, top, bottom = 90, 40, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = np.concatenate([np.asarray(y_true), np.asarray(y_pred)])
    min_v = float(values.min())
    max_v = float(values.max())
    if min_v == max_v:
        max_v = min_v + 1

    def scale_x(value):
        return left + ((float(value) - min_v) / (max_v - min_v)) * plot_w

    def scale_y(value):
        return top + plot_h - ((float(value) - min_v) / (max_v - min_v)) * plot_h

    points = []
    for actual, predicted in zip(y_true, y_pred):
        x = scale_x(actual)
        y = scale_y(predicted)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#0f766e" opacity="0.75" />')

    diagonal = (
        f'<line x1="{scale_x(min_v):.2f}" y1="{scale_y(min_v):.2f}" '
        f'x2="{scale_x(max_v):.2f}" y2="{scale_y(max_v):.2f}" '
        'stroke="#dc2626" stroke-width="3" stroke-dasharray="8 8" />'
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="32" text-anchor="middle" font-family="Segoe UI, Arial" font-size="24" font-weight="700" fill="#111827">Actual vs Predicted Price</text>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>
{diagonal}
{''.join(points)}
<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="16" fill="#374151">Actual Price</text>
<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Segoe UI, Arial" font-size="16" fill="#374151">Predicted Price</text>
<text x="{left}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="12" fill="#6b7280">{min_v:,.0f}</text>
<text x="{left + plot_w}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="12" fill="#6b7280">{max_v:,.0f}</text>
</svg>"""
    (CHART_DIR / "actual_vs_predicted.svg").write_text(svg, encoding="utf-8")


def save_feature_importance_chart(feature_importance):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    rows = feature_importance[:10]
    width, height = 900, 560
    left, right, top = 260, 60, 70
    bar_h, gap = 28, 16
    max_value = max([row["importance"] for row in rows], default=1)
    svg_rows = []

    for index, row in enumerate(rows):
        y = top + index * (bar_h + gap)
        bar_w = 520 * (row["importance"] / max_value if max_value else 0)
        label = svg_escape(row["feature"])
        value = row["importance"]
        svg_rows.append(
            f'<text x="{left - 12}" y="{y + 20}" text-anchor="end" font-family="Segoe UI, Arial" font-size="14" fill="#374151">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="4" fill="#2563eb"/>'
            f'<text x="{left + bar_w + 10:.2f}" y="{y + 20}" font-family="Segoe UI, Arial" font-size="13" fill="#111827">{value:.4f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="35" text-anchor="middle" font-family="Segoe UI, Arial" font-size="24" font-weight="700" fill="#111827">Top Feature Importance</text>
{''.join(svg_rows)}
</svg>"""
    (CHART_DIR / "feature_importance.svg").write_text(svg, encoding="utf-8")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df.columns = [clean_text(col).lower() for col in df.columns]
    text_columns = [
        col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    df[["city", "locality"]] = df["location"].apply(lambda value: pd.Series(split_location(value)))
    df["location"] = df["city"] + " - " + df["locality"]
    df = add_derived_features(df)

    df = df.dropna(subset=["price"])
    X = df[FEATURE_COLUMNS]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidate_models = {
        "Ridge Regression": Ridge(alpha=10),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=2,
            max_features=0.8,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42,
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
        ),
        "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.1,
        ),
    }

    model_comparison = []
    trained_models = {}

    for name, regressor in candidate_models.items():
        pipeline = build_pipeline(regressor)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        metrics["cross_validation_r2_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cross_validation_r2_std"] = round(float(cv_scores.std()), 4)

        trained_models[name] = pipeline
        model_comparison.append({"model": name, **metrics})

    best_result = max(model_comparison, key=lambda row: row["r2_score"])
    best_model_name = best_result["model"]
    best_pipeline = trained_models[best_model_name]
    y_pred_best = best_pipeline.predict(X_test)
    feature_importance = get_feature_importance(best_pipeline, X_test, y_test)

    joblib.dump(best_pipeline, MODEL_PATH)
    save_actual_vs_predicted_chart(y_test, y_pred_best)
    save_feature_importance_chart(feature_importance)

    metrics_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_model_name,
        "best_model_metrics": best_result,
        "model_comparison": sorted(model_comparison, key=lambda row: row["rmse"]),
        "feature_importance": feature_importance,
        "known_locations": sorted(df["location"].dropna().unique().tolist()),
        "known_cities": sorted(df["city"].dropna().unique().tolist()),
        "feature_columns": FEATURE_COLUMNS,
        "notes": [
            "This is a regression problem, so R2, MAE, RMSE and MAPE are more suitable than precision and recall.",
            "Unseen locations are handled by OneHotEncoder(handle_unknown='ignore').",
            "For a new locality in a known city, the model uses city and property features while ignoring the unknown locality token.",
        ],
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=4), encoding="utf-8")

    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(json.dumps(best_result, indent=4))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved charts to {CHART_DIR}")


if __name__ == "__main__":
    main()
    
