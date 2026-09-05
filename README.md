## SkyGuard AI dashboard

Start the local dashboard from the project root:

```powershell
python dashboard_server.py
```

Open http://127.0.0.1:8000 in a browser. The dashboard reads `outputs/live_anomalies.csv` and the latest historical records. Use **Refresh data** to run one live inference cycle against the station files and update the view.
