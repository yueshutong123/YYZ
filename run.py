import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.join(BASE_DIR, 'web_app')
sys.path.insert(0, WEB_APP_DIR)
sys.path.insert(0, BASE_DIR)

os.chdir(WEB_APP_DIR)

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("  Depression Detection Web System")
    print("  Model: 0.77best_model.pth (Acc: 77.45%, F1: 80.0%)")
    print("  Feature Extraction: eGeMAPS v02 + Face Landmarks")
    print("=" * 60)
    print()
    print("Open http://127.0.0.1:5000 in your browser")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
