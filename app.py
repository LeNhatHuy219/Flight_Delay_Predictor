"""
Flight Delay Predictor — Flask backend.

Chạy local:
    pip install -r requirements.txt
    python app.py
Mở http://127.0.0.1:5000
"""
import datetime
import math
import os

import folium
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ──────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flight_delay_all_airlines_model.pkl")
AIRPORTS_PATH = os.path.join(BASE_DIR, "data", "airports.csv")

app = Flask(__name__)


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {MODEL_PATH}\n"
            f"Hãy đặt file `flight_delay_all_airlines_model.pkl` vào thư mục models/."
        )
    return joblib.load(MODEL_PATH), MODEL_PATH


def load_airport_coords():
    coords = {}
    if not os.path.exists(AIRPORTS_PATH):
        return coords
    try:
        df_airports = pd.read_csv(AIRPORTS_PATH)
        for _, row in df_airports.iterrows():
            if pd.notna(row.get("IATA_CODE")) and pd.notna(row.get("LATITUDE")):
                coords[str(row["IATA_CODE"])] = (
                    float(row["LATITUDE"]),
                    float(row["LONGITUDE"]),
                )
    except Exception as e:
        print(f"Cảnh báo: không đọc được airports.csv ({e})")
    return coords


# Load tại startup
artifact, artifact_path = load_model()
airport_coords = load_airport_coords()

# Unpack artifact
model = artifact["model"]
metrics_info = artifact.get("metrics", {})
feature_columns = artifact.get("feature_columns", [])
origin_options = artifact.get("origin_options", [])
dest_options = artifact.get("dest_options", [])
airline_names = artifact.get("airline_names", {})
artifact_carrier = artifact.get("carrier", "ALL")

airline_options = artifact.get("airline_options", [])
if not airline_options:
    airline_options = [] if artifact_carrier == "ALL" else [artifact_carrier]
airline_options = sorted([str(code) for code in airline_options])

supports_airline_feature = "AIRLINE" in feature_columns
supports_all_airlines = supports_airline_feature and len(airline_options) > 1
model_name = artifact.get("model_name", artifact.get("best_model_name", "Flight delay model"))

print(f"✓ Model loaded: {model_name}")
print(f"✓ Airports: {len(airport_coords)} | Airlines: {len(airline_options)}")


# ──────────────────────────────────────────────
# Predict helpers
# ──────────────────────────────────────────────
def build_pipeline_input(airline, origin, dest, flight_date, flight_time, scheduled_minutes, context_values):
    dep_sec = flight_time.hour * 3600 + flight_time.minute * 60
    dep_hhmm = flight_time.hour * 100 + flight_time.minute
    weekday = int(flight_date.weekday())
    row = {
        "AIRLINE": airline,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": dest,
        "dep_seconds": float(dep_sec),
        "heure_depart_sec": float(dep_sec),
        "SCHEDULED_DEPARTURE": float(dep_hhmm),
        "dep_hour": int(flight_time.hour),
        "dep_minute": int(flight_time.minute),
        "dep_hour_sin": float(np.sin(2 * np.pi * dep_sec / 86400)),
        "dep_hour_cos": float(np.cos(2 * np.pi * dep_sec / 86400)),
        "dep_dayofweek": weekday,
        "DAY_OF_WEEK": weekday + 1,
        "MONTH": int(flight_date.month),
        "DAY": int(flight_date.day),
        "scheduled_time": float(scheduled_minutes),
        "SCHEDULED_TIME": float(scheduled_minutes),
        "prev_avg_delay_3": float(context_values.get("prev_avg_delay_3", 0.0)),
        "prev_max_delay_3": float(context_values.get("prev_max_delay_3", 0.0)),
        "tavg": float(context_values.get("tavg", 0.0)),
        "wspd": float(context_values.get("wspd", 0.0)),
        "prcp": float(context_values.get("prcp", 0.0)),
        "snow": float(context_values.get("snow", 0.0)),
    }
    if feature_columns:
        return pd.DataFrame([{col: row.get(col, 0.0) for col in feature_columns}])
    return pd.DataFrame([row])


def predict_for_airline(airline, origin, dest, flight_date, flight_time, scheduled_minutes, context_values):
    artifact_kind = artifact.get("artifact_kind")
    if artifact_kind in {
        "sklearn_pipeline_multimodel_best",
        "extended_best_model",
        "all_airlines_extended_model",
        "all_airlines_delay_model",
    } or feature_columns:
        X_input = build_pipeline_input(airline, origin, dest, flight_date, flight_time, scheduled_minutes, context_values)
        return float(model.predict(X_input)[0])

    poly = artifact["poly_transformer"]
    le_airport = artifact["le_airport"]
    le_dest = artifact["le_dest"]
    dep_sec = flight_time.hour * 3600 + flight_time.minute * 60
    origin_enc = int(le_airport.transform([origin])[0])
    dest_enc = int(le_dest.transform([dest])[0])
    X_input = np.array([[dep_sec, origin_enc, dest_enc]])
    return float(model.predict(poly.transform(X_input)).flatten()[0])


def estimate_flight_time(origin, dest):
    """Ước lượng thời lượng bay (phút) từ khoảng cách địa lý giữa 2 sân bay."""
    o = airport_coords.get(origin)
    d = airport_coords.get(dest)
    if not o or not d:
        return 120.0
    lat1, lon1 = math.radians(o[0]), math.radians(o[1])
    lat2, lon2 = math.radians(d[0]), math.radians(d[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    km = 6371 * 2 * math.asin(math.sqrt(a))
    return round(km / 800 * 60 + 30, 1)


def make_arc(lat1, lon1, lat2, lon2, n=80):
    pts = []
    for i in range(n + 1):
        t = i / n
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        bulge = math.sin(math.pi * t) * 3.5
        lat += bulge * math.cos(math.radians((lat1 + lat2) / 2))
        pts.append([lat, lon])
    return pts


def draw_flight_map(origin, dest, pred_minutes):
    o_coord = airport_coords.get(origin)
    d_coord = airport_coords.get(dest)
    if not o_coord or not d_coord:
        return None

    olat, olon = o_coord
    dlat, dlon = d_coord

    if pred_minutes <= 5:
        arc_color, dot_color = "#15803d", "#16a34a"
    elif pred_minutes <= 20:
        arc_color, dot_color = "#d97706", "#f59e0b"
    else:
        arc_color, dot_color = "#db2777", "#be185d"

    m = folium.Map(
        location=[(olat + dlat) / 2, (olon + dlon) / 2],
        zoom_start=4,
        tiles="CartoDB positron",
        attr="CartoDB",
    )

    arc_pts = make_arc(olat, olon, dlat, dlon)
    folium.PolyLine(arc_pts, color=arc_color, weight=3, opacity=0.9).add_to(m)

    folium.CircleMarker(
        location=[olat, olon], radius=8, color="white", fill=True,
        fill_color=dot_color, fill_opacity=1, weight=2,
        popup=folium.Popup(f"<b>{origin}</b><br>Xuất phát", max_width=130),
        tooltip=origin,
    ).add_to(m)

    folium.CircleMarker(
        location=[dlat, dlon], radius=8, color="white", fill=True,
        fill_color="#7c3aed", fill_opacity=1, weight=2,
        popup=folium.Popup(f"<b>{dest}</b><br>Điểm đến", max_width=130),
        tooltip=dest,
    ).add_to(m)

    mid_lat, mid_lon = arc_pts[len(arc_pts) // 2]
    folium.Marker(
        location=[mid_lat, mid_lon],
        icon=folium.DivIcon(
            html=f"""<div style="background:white;border:1.5px solid {arc_color};border-radius:8px;
                padding:4px 9px;font-size:11px;font-weight:700;color:{arc_color};white-space:nowrap;
                box-shadow:0 8px 20px rgba(31,41,55,0.14);font-family:Inter,sans-serif;">
                {pred_minutes:.1f} phút</div>""",
            icon_size=(110, 28), icon_anchor=(55, 14),
        ),
    ).add_to(m)

    return m.get_root().render()


def status_for_delay(minutes):
    if minutes <= 5:
        return "on-time", "Đúng giờ hoặc trễ rất nhẹ"
    if minutes <= 20:
        return "slight", "Có khả năng trễ nhẹ"
    return "delayed", "Nguy cơ trễ cao"


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


def _sanitize(obj):
    """Replace NaN / Infinity with None so JSON stays spec-compliant."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


@app.route("/api/meta")
def api_meta():
    """Trả về metadata để UI populate dropdowns và metrics tab."""
    return jsonify(_sanitize({
        "origins": origin_options,
        "dests": dest_options,
        "airlines": airline_options,
        "airline_names": airline_names,
        "supports_all_airlines": supports_all_airlines,
        "model_name": model_name,
        "metrics": metrics_info,
        "source_rows": artifact.get("source_rows"),
        "ml_leaderboard": artifact.get("ml_leaderboard"),
        "combined_leaderboard": artifact.get("combined_leaderboard"),
        "stage_leaderboard": artifact.get("stage_leaderboard"),
    }))


@app.route("/api/feature_importance")
def api_feature_importance():
    """Trả về top-15 feature importances từ LightGBM trong pipeline."""
    try:
        lgbm = model.named_steps["model"]
        preprocessor = model.named_steps["preprocess"]

        # Lấy tên feature sau transform
        try:
            raw_names = list(preprocessor.get_feature_names_out())
        except Exception:
            raw_names = [f"f{i}" for i in range(len(lgbm.feature_importances_))]

        # Xoá prefix "num__" / "cat__" từ ColumnTransformer
        clean = []
        for n in raw_names:
            n = str(n)
            clean.append(n.split("__", 1)[1] if "__" in n else n)

        imps = lgbm.feature_importances_.tolist()
        pairs = sorted(zip(clean, imps), key=lambda x: x[1], reverse=True)[:15]
        total = sum(p[1] for p in pairs) or 1
        return jsonify({
            "features": [p[0] for p in pairs],
            "importances": [round(p[1] / total * 100, 2) for p in pairs],  # % of top-15
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST JSON body:
    {
      "origin": "LAX", "dest": "JFK",
      "date": "2015-01-15", "time": "08:30",
      "scheduled_time": 330, "mode": "single"|"all",
      "airline": "AA",
      "prev_avg_delay_3": 0, "prev_max_delay_3": 0,
      "tavg": 0, "wspd": 0, "prcp": 0, "snow": 0
    }
    """
    data = request.get_json(force=True)

    origin = data.get("origin", "")
    dest = data.get("dest", "")
    date_str = data.get("date", "2015-01-01")
    time_str = data.get("time", "08:00")
    mode = data.get("mode", "single")
    scheduled_minutes = float(data["scheduled_time"]) if data.get("scheduled_time") else estimate_flight_time(origin, dest)

    if origin and dest and origin == dest:
        return jsonify({"error": "Sân bay xuất phát và sân bay đến không được trùng nhau."}), 400

    try:
        flight_date = datetime.date.fromisoformat(date_str)
    except Exception:
        flight_date = datetime.date(2015, 1, 1)

    try:
        h, m = map(int, time_str.split(":"))
        flight_time = datetime.time(h, m)
    except Exception:
        flight_time = datetime.time(8, 0)

    context_values = {
        "prev_avg_delay_3": float(data.get("prev_avg_delay_3", 0)),
        "prev_max_delay_3": float(data.get("prev_max_delay_3", 0)),
        "tavg": float(data.get("tavg", 0)),
        "wspd": float(data.get("wspd", 0)),
        "prcp": float(data.get("prcp", 0)),
        "snow": float(data.get("snow", 0)),
    }

    if mode == "all":
        rows = []
        errors = []
        for code in airline_options:
            try:
                pred = predict_for_airline(code, origin, dest, flight_date, flight_time, scheduled_minutes, context_values)
                rows.append({
                    "airline": code,
                    "name": airline_names.get(code, code),
                    "delay": round(max(0.0, pred), 2),
                })
            except Exception as e:
                errors.append(f"{code}: {e}")
        if not rows:
            return jsonify({"error": "Không thể dự đoán cho bất kỳ hãng nào. " + "; ".join(errors)}), 500
        rows.sort(key=lambda r: r["delay"])
        avg = round(sum(r["delay"] for r in rows) / len(rows), 2)
        status_key, status_text = status_for_delay(avg)
        map_html = draw_flight_map(origin, dest, avg)
        return jsonify({
            "mode": "all",
            "avg_delay": avg,
            "status_key": status_key,
            "status_text": status_text,
            "rankings": rows,
            "map_html": map_html,
        })
    else:
        airline = data.get("airline", airline_options[0] if airline_options else "")
        try:
            pred = predict_for_airline(airline, origin, dest, flight_date, flight_time, scheduled_minutes, context_values)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        pred_display = round(max(0.0, pred), 2)
        status_key, status_text = status_for_delay(pred_display)
        map_html = draw_flight_map(origin, dest, pred_display)
        return jsonify({
            "mode": "single",
            "airline": airline,
            "origin": origin,
            "dest": dest,
            "time": time_str,
            "delay": pred_display,
            "scheduled_minutes": scheduled_minutes,
            "status_key": status_key,
            "status_text": status_text,
            "map_html": map_html,
        })


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)