Viewer Monitor Project
======================

This project contains:

1. Flask backend (nginx_logs_api.py)
2. Streamlit dashboard (viewer_dashboard.py)
3. Auto-generated HTML reports saved per check

Structure:
viewer-monitor/
 ├ backend/
 └ dashboard/

Run backend:
cd backend
pip install -r requirements.txt
python3 nginx_logs_api.py

Run dashboard:
cd dashboard
pip install -r requirements.txt
streamlit run viewer_dashboard.py
