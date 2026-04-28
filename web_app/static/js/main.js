const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const btnRemove = document.getElementById('btnRemove');
const btnAnalyze = document.getElementById('btnAnalyze');
const resultCard = document.getElementById('resultCard');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const statusText = document.getElementById('statusText');
const resultContent = document.getElementById('resultContent');
const resultMain = document.getElementById('resultMain');
const errorContent = document.getElementById('errorContent');
const errorText = document.getElementById('errorText');
const featureInfo = document.getElementById('featureInfo');

let selectedFile = null;
let pollTimer = null;
let taskId = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

btnRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    btnAnalyze.disabled = true;
    uploadArea.style.display = '';
});

function handleFile(file) {
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-ms-wmv', 'video/webm'];
    const validExts = ['.mp4', '.avi', '.mov', '.wmv', '.webm'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
        alert('请上传有效的视频文件（MP4、AVI、MOV、WMV）');
        return;
    }

    if (file.size > 1024 * 1024 * 1024) {
        alert('文件大小超过 1GB 限制');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name + ' (' + formatSize(file.size) + ')';
    fileInfo.style.display = 'flex';
    btnAnalyze.disabled = false;
    uploadArea.style.display = 'none';

    hideResults();
}

function formatSize(bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);

    btnAnalyze.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    document.querySelector('.btn-loading').style.display = 'inline';
    resultCard.style.display = '';
    resultContent.style.display = 'none';
    errorContent.style.display = 'none';
    updateProgress(0, '正在上传视频...');

    try {
        const uploadResp = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (!uploadResp.ok) {
            const err = await uploadResp.json();
            throw new Error(err.error || 'Upload failed');
        }

        const data = await uploadResp.json();
        taskId = data.task_id;

        pollStatus();
    } catch (err) {
        showError(err.message);
        resetButton();
    }
});

function pollStatus() {
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch('/api/status/' + taskId);
            if (!resp.ok) {
                clearInterval(pollTimer);
                showError('获取任务状态失败');
                resetButton();
                return;
            }

            const data = await resp.json();

            const statusMessages = {
                'uploaded': '等待处理...',
                'extracting_audio': '正在从视频中提取音频...',
                'extracting_acoustic_features': '正在提取音频特征 (eGeMAPS v02)...',
                'extracting_visual_features': '正在提取视觉特征 (人脸关键点)...',
                'aligning_features': '正在对齐音视频特征序列...',
                'predicting': '正在运行抑郁症检测模型...',
                'completed': '分析完成！',
            };

            updateProgress(data.progress, statusMessages[data.status] || data.status);

            if (data.status === 'completed') {
                clearInterval(pollTimer);
                showResult(data);
                resetButton();
            } else if (data.status === 'error') {
                clearInterval(pollTimer);
                showError(data.error || '未知错误');
                resetButton();
            }
        } catch (err) {
            clearInterval(pollTimer);
            showError('连接错误: ' + err.message);
            resetButton();
        }
    }, 1500);
}

function updateProgress(progress, text) {
    progressBar.style.background = 'linear-gradient(90deg, #6eb5ff, #7c5cfc)';
    progressBar.style.width = progress + '%';
    progressText.textContent = Math.round(progress) + '%';
    statusText.textContent = text;
}

function showResult(data) {
    const result = data.result;
    resultContent.style.display = '';
    errorContent.style.display = 'none';

    const isDepression = result.prediction === 'depression';
    resultMain.className = 'result-main ' + (isDepression ? 'depression' : 'normal');
    resultMain.innerHTML = `
        <div class="result-label">${isDepression ? '检测到抑郁倾向' : '未检测到抑郁倾向'}</div>
        <div class="result-subtitle">${isDepression ? 
            '模型预测该视频存在抑郁相关特征。' : 
            '模型未发现显著的抑郁相关特征。'}</div>
    `;

    document.getElementById('depBar').style.width = result.depression_confidence + '%';
    document.getElementById('depValue').textContent = result.depression_confidence + '%';
    document.getElementById('normBar').style.width = result.normal_confidence + '%';
    document.getElementById('normValue').textContent = result.normal_confidence + '%';

    if (data.feature_info) {
        const fi = data.feature_info;
        featureInfo.innerHTML = `
            <span>视觉: ${fi.visual_frames} 帧 × ${fi.visual_dim} 维</span>
            <span>音频: ${fi.acoustic_frames} 帧 × ${fi.acoustic_dim} 维</span>
        `;
    }
}

function showError(msg) {
    errorContent.style.display = '';
    resultContent.style.display = 'none';
    errorText.textContent = msg;
}

function hideResults() {
    resultCard.style.display = 'none';
    resultContent.style.display = 'none';
    errorContent.style.display = 'none';
}

function resetButton() {
    btnAnalyze.disabled = false;
    document.querySelector('.btn-text').style.display = 'inline';
    document.querySelector('.btn-loading').style.display = 'none';
}
