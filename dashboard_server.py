import csv
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
RESULTS_FILE = PROJECT_ROOT / "outputs" / "live_anomalies.csv"
HISTORY_FILE = PROJECT_ROOT / "data" / "processed" / "SkyGuard_clean_3hourly.csv"


def read_csv(path, limit=None):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[-limit:] if limit else rows


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def anomaly_reason(row):
    if row.get("Ensemble_Anomaly") != "1":
        return "No significant anomaly detected"
    signals = [
        (abs(number(row.get("Temperature_LocalZ")) or 0), "Temperature", number(row.get("Temperature_Diff")), "C change from the previous reading"),
        (abs(number(row.get("Humidity_LocalZ")) or 0), "Humidity", number(row.get("Humidity_Diff")), "% change from the previous reading"),
        (abs(number(row.get("Pressure_LocalZ")) or 0), "Pressure", number(row.get("Pressure_Diff")), "hPa change from the previous reading"),
    ]
    _, signal_name, change, suffix = max(signals, key=lambda signal: signal[0])
    if change is not None:
        return f"Unusual {signal_name.lower()} ({change:+.1f} {suffix}); {row.get('Model_Agreement', '0')}/4 models agree"
    return f"Unusual {signal_name.lower()} pattern; {row.get('Model_Agreement', '0')}/4 models agree"


def dashboard_payload():
    latest = read_csv(RESULTS_FILE)
    for row in latest:
        row["Anomaly_Reason"] = anomaly_reason(row)
    history = read_csv(HISTORY_FILE)
    locations = sorted({row.get("Location", "") for row in latest if row.get("Location")})
    history_by_location = {}
    for location in locations:
        points = [row for row in history if row.get("Location") == location][-24:]
        history_by_location[location] = [
            {
                "time": row.get("DateTime", ""),
                "temperature": number(row.get("Temperature_C")),
                "humidity": number(row.get("Humidity_Percent")),
                "pressure": number(row.get("Pressure_hPa")),
            }
            for row in points
        ]
    return {"updated": latest[-1].get("DateTime") if latest else None, "latest": latest, "history": history_by_location}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_ROOT), **kwargs)

    def send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if urlparse(self.path).path == "/api/data":
            self.send_json(dashboard_payload())
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/refresh":
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            from main import run_live_cycle

            run_live_cycle()
            self.send_json(dashboard_payload())
        except Exception as error:
            self.send_json({"error": f"Live pipeline failed: {type(error).__name__}: {error}"}, 500)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("SkyGuard dashboard running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()