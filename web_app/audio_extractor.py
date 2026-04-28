import os
import subprocess
import numpy as np
import soundfile as sf
import opensmile


def extract_audio_from_video(video_path, audio_path, sample_rate=16000):
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate), '-ac', '1',
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path


def extract_acoustic_features(audio_path, target_sample_rate=16000):
    signal, sr = sf.read(audio_path)
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    if sr != target_sample_rate:
        import librosa
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sample_rate)
        sr = target_sample_rate

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    features = smile.process_signal(signal, sr)
    feature_array = features.to_numpy().astype(np.float64)
    return feature_array
