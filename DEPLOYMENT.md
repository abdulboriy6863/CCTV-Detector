# CCTV-Detector Production Deployment & Maintenance Guide

## 📌 1. Server Configuration & Access
* **Server IP:** `192.168.0.28`
* **SSH User:** `bn`
* **Port:** `8000`
* **Web Dashboard:** `http://192.168.0.28:8000/`
* **Swagger API Documentation:** `http://192.168.0.28:8000/docs`
* **Service Name:** `cctv-detector.service`
* **Project Directory:** `/home/bn/CCTV-Detector`

---

## ⚡ 2. Updating & Deploying (Git Pull)
To pull latest changes and restart the live service:
```bash
ssh bn@192.168.0.28
cd /home/bn/CCTV-Detector
git pull origin main
sudo systemctl restart cctv-detector
sudo systemctl status cctv-detector
```

---

## 🧹 3. Database Purge & Clean Production Reset
To clear test records and initialize clean database tables:
```bash
# Purge vehicle snapshots and alert logs
./venv/bin/python scripts/purge_data.py --snapshots -y

# Complete clean reset (snapshots, alerts, cameras, disk images)
./venv/bin/python scripts/purge_data.py --all -y
```

Or trigger directly from Web Dashboard:
Open `http://192.168.0.28:8000/` and click **"시스템 초기화" (System Reset)** in the top navigation bar.

---

## 📊 4. Monitoring & Health Check
* **System Health:** `curl http://192.168.0.28:8000/health`
* **Real-Time Slots HUD:** `curl http://192.168.0.28:8000/api/v1/cameras/slots-status`
* **Summary Analytics:** `curl http://192.168.0.28:8000/api/v1/stats/summary`
* **Live Systemd Logs:** `journalctl -u cctv-detector -f`
