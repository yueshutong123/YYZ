import os
import sys
import uuid
import time
import json
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_extractor import extract_visual_features
from audio_extractor import extract_audio_from_video, extract_acoustic_features
from predictor import predict, get_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_tasks = {}


def process_video_task(task_id, video_path):
    try:
        _tasks[task_id]['status'] = 'extracting_audio'
        _tasks[task_id]['progress'] = 10

        audio_path = os.path.join(UPLOAD_DIR, f'{task_id}.wav')
        extract_audio_from_video(video_path, audio_path)

        _tasks[task_id]['status'] = 'extracting_acoustic_features'
        _tasks[task_id]['progress'] = 30
        acoustic_features = extract_acoustic_features(audio_path)

        _tasks[task_id]['status'] = 'extracting_visual_features'
        _tasks[task_id]['progress'] = 40
        visual_features = extract_visual_features(video_path)

        _tasks[task_id]['status'] = 'aligning_features'
        _tasks[task_id]['progress'] = 70
        min_frames = min(visual_features.shape[0], acoustic_features.shape[0])
        visual_features = visual_features[:min_frames]
        acoustic_features = acoustic_features[:min_frames]

        feature_path = os.path.join(RESULTS_DIR, f'{task_id}_features.npz')
        np.savez(feature_path, visual=visual_features, acoustic=acoustic_features)

        _tasks[task_id]['status'] = 'predicting'
        _tasks[task_id]['progress'] = 80
        result = predict(visual_features, acoustic_features)

        _tasks[task_id]['status'] = 'completed'
        _tasks[task_id]['progress'] = 100
        _tasks[task_id]['result'] = result
        _tasks[task_id]['feature_info'] = {
            'visual_frames': int(visual_features.shape[0]),
            'visual_dim': int(visual_features.shape[1]),
            'acoustic_frames': int(acoustic_features.shape[0]),
            'acoustic_dim': int(acoustic_features.shape[1]),
        }

        try:
            os.remove(audio_path)
        except Exception:
            pass

    except Exception as e:
        _tasks[task_id]['status'] = 'error'
        _tasks[task_id]['error'] = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    task_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    video_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
    file.save(video_path)

    _tasks[task_id] = {
        'id': task_id,
        'filename': file.filename,
        'status': 'uploaded',
        'progress': 0,
        'result': None,
        'error': None,
        'created_at': time.time(),
    }

    thread = threading.Thread(target=process_video_task, args=(task_id, video_path), daemon=True)
    thread.start()

    return jsonify({'task_id': task_id, 'status': 'uploaded'})


@app.route('/api/status/<task_id>')
def task_status(task_id):
    task = _tasks.get(task_id)
    if task is None:
        return jsonify({'error': 'Task not found'}), 404

    response = {
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress'],
        'filename': task.get('filename', ''),
    }

    if task['status'] == 'completed':
        response['result'] = task['result']
        response['feature_info'] = task.get('feature_info', {})
    elif task['status'] == 'error':
        response['error'] = task.get('error', 'Unknown error')

    return jsonify(response)


@app.route('/api/model_info')
def model_info():
    _, _, ckpt_info = get_model()
    return jsonify({
        'model_path': '0.77best_model.pth',
        'epoch': ckpt_info.get('epoch', 'unknown'),
        'val_accuracy': round(ckpt_info.get('val_acc', 0) * 100, 2),
        'val_f1': round(ckpt_info.get('val_f1', 0) * 100, 2),
        'visual_input_dim': 136,
        'acoustic_input_dim': 25,
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    })


@app.teardown_appcontext
def cleanup(error):
    pass


if __name__ == '__main__':
    print("=" * 60)
    print("  Depression Detection Web System")
    print("  Model: 0.77best_model.pth (Acc: 77%)")
    print("  D-Vlog Dataset based on eGeMAPS + OpenFace features")
    print("=" * 60)
    print("\nPreloading model and normalization stats...")
    try:
        get_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Model preload failed: {e}")
        print("Model will be loaded on first inference request.")
    print(f"\nStarting server at http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
