# send_alert.py
import requests

SERVER = "http://127.0.0.1:8000"  # or http://<LAN-IP>:8000 for phones
payload = {
    "lat": 21.958, "lon": 96.089,
    "mag": 5.8, "sp_sec": 9.4, "dist_km": 42, "eta_sec": 12,
    "place": "Near Mandalay", "severity": "strong", "score": 0.97, "threshold": 0.75
}
r = requests.post(f"{SERVER}/send", json=payload, timeout=10)
print("Server:", r.status_code, r.text)
