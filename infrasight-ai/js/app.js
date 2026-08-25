import { createSceneSummary, formatPercent, summarizeDetections } from './analytics.js';

const elements = {
  modelStatus: document.querySelector('#modelStatus'),
  imageUpload: document.querySelector('#imageUpload'),
  cameraButton: document.querySelector('#cameraButton'),
  stage: document.querySelector('#stage'),
  emptyState: document.querySelector('#emptyState'),
  sourceImage: document.querySelector('#sourceImage'),
  camera: document.querySelector('#camera'),
  canvas: document.querySelector('#canvas'),
  confidenceRange: document.querySelector('#confidenceRange'),
  confidenceValue: document.querySelector('#confidenceValue'),
  objectCount: document.querySelector('#objectCount'),
  vehicleCount: document.querySelector('#vehicleCount'),
  personCount: document.querySelector('#personCount'),
  avgConfidence: document.querySelector('#avgConfidence'),
  sceneSummary: document.querySelector('#sceneSummary'),
  detectionList: document.querySelector('#detectionList'),
  topClass: document.querySelector('#topClass')
};

const ctx = elements.canvas.getContext('2d');
let model = null;
let cameraStream = null;
let cameraLoop = null;
let activeMode = null;

const BOX_COLORS = ['#22d3ee', '#f59e0b', '#34d399', '#f472b6', '#a78bfa', '#fb7185'];

function setModelStatus(text, state = '') {
  elements.modelStatus.classList.remove('ready', 'error');
  if (state) elements.modelStatus.classList.add(state);
  elements.modelStatus.querySelector('span:last-child').textContent = text;
}

async function loadModel() {
  try {
    setModelStatus('Loading model');
    await waitForGlobal('tf');
    await waitForGlobal('cocoSsd');
    await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
    await tf.ready();
    model = await cocoSsd.load({ base: 'mobilenet_v2' });
    setModelStatus('Model ready', 'ready');
    elements.cameraButton.disabled = false;
  } catch (error) {
    console.error(error);
    setModelStatus('Model unavailable', 'error');
    elements.sceneSummary.textContent = 'The detection model could not be loaded. Check your network connection and refresh the page.';
  }
}

function waitForGlobal(name, timeoutMs = 12000) {
  return new Promise((resolve, reject) => {
    const start = performance.now();
    const timer = setInterval(() => {
      if (window[name]) {
        clearInterval(timer);
        resolve(window[name]);
      } else if (performance.now() - start > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`${name} did not load in time`));
      }
    }, 50);
  });
}

function confidenceThreshold() {
  return Number(elements.confidenceRange.value) / 100;
}

function prepareCanvas(width, height) {
  elements.canvas.width = width;
  elements.canvas.height = height;
  elements.canvas.hidden = false;
  elements.emptyState.hidden = true;
  elements.stage.dataset.empty = 'false';
}

function drawPredictions(predictions) {
  ctx.lineWidth = Math.max(2, elements.canvas.width / 500);
  ctx.font = `${Math.max(14, elements.canvas.width / 55)}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = 'top';

  predictions.forEach((prediction, index) => {
    const [x, y, width, height] = prediction.bbox;
    const color = BOX_COLORS[index % BOX_COLORS.length];
    const label = `${prediction.class} ${formatPercent(prediction.score)}`;

    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, width, height);

    const textWidth = ctx.measureText(label).width;
    const textHeight = Math.max(20, elements.canvas.width / 42);
    const labelY = Math.max(0, y - textHeight);
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, textWidth + 12, textHeight);
    ctx.fillStyle = '#071015';
    ctx.fillText(label, x + 6, labelY + 2);
  });
}

function renderAnalytics(predictions) {
  const summary = summarizeDetections(predictions);
  elements.objectCount.textContent = summary.objects;
  elements.vehicleCount.textContent = summary.vehicles;
  elements.personCount.textContent = summary.people;
  elements.avgConfidence.textContent = summary.objects ? formatPercent(summary.averageConfidence) : '—';
  elements.sceneSummary.textContent = createSceneSummary(summary);
  elements.topClass.textContent = summary.topClass ? `Top: ${summary.topClass}` : 'No data';

  if (!predictions.length) {
    elements.detectionList.innerHTML = '<p class="muted">No detections met the selected threshold.</p>';
    return;
  }

  const rows = [...predictions]
    .sort((a, b) => b.score - a.score)
    .map((prediction) => `
      <div class="detection-row">
        <span class="detection-name">${escapeHtml(prediction.class)}</span>
        <span class="detection-confidence">${formatPercent(prediction.score)}</span>
      </div>`)
    .join('');
  elements.detectionList.innerHTML = rows;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function analyzeImage() {
  if (!model || !elements.sourceImage.src) return;
  setModelStatus('Analyzing image');
  const predictions = await model.detect(elements.sourceImage, 20, confidenceThreshold());
  prepareCanvas(elements.sourceImage.naturalWidth, elements.sourceImage.naturalHeight);
  ctx.drawImage(elements.sourceImage, 0, 0);
  drawPredictions(predictions);
  renderAnalytics(predictions);
  setModelStatus('Model ready', 'ready');
}

async function handleImage(file) {
  if (!file || !file.type.startsWith('image/')) return;
  stopCamera();
  activeMode = 'image';
  const objectUrl = URL.createObjectURL(file);
  elements.sourceImage.onload = async () => {
    try {
      await analyzeImage();
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  };
  elements.sourceImage.src = objectUrl;
}

async function startCamera() {
  if (!model) return;
  if (cameraStream) {
    stopCamera();
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    elements.camera.srcObject = cameraStream;
    await elements.camera.play();
    activeMode = 'camera';
    elements.cameraButton.textContent = 'Stop camera';
    prepareCanvas(elements.camera.videoWidth || 960, elements.camera.videoHeight || 540);
    cameraLoop = window.setInterval(analyzeCameraFrame, 650);
    await analyzeCameraFrame();
  } catch (error) {
    console.error(error);
    elements.sceneSummary.textContent = 'Camera access was not available. You can still upload an image.';
    stopCamera();
  }
}

async function analyzeCameraFrame() {
  if (!model || !cameraStream || elements.camera.readyState < 2) return;
  const width = elements.camera.videoWidth;
  const height = elements.camera.videoHeight;
  if (!width || !height) return;
  prepareCanvas(width, height);
  ctx.drawImage(elements.camera, 0, 0, width, height);
  const predictions = await model.detect(elements.camera, 20, confidenceThreshold());
  ctx.drawImage(elements.camera, 0, 0, width, height);
  drawPredictions(predictions);
  renderAnalytics(predictions);
}

function stopCamera() {
  if (cameraLoop) window.clearInterval(cameraLoop);
  cameraLoop = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  elements.camera.srcObject = null;
  elements.cameraButton.textContent = 'Start camera';
  if (activeMode === 'camera') activeMode = null;
}

elements.imageUpload.addEventListener('change', (event) => handleImage(event.target.files?.[0]));
elements.cameraButton.addEventListener('click', startCamera);
elements.confidenceRange.addEventListener('input', () => {
  elements.confidenceValue.textContent = `${elements.confidenceRange.value}%`;
});
elements.confidenceRange.addEventListener('change', async () => {
  if (activeMode === 'image') await analyzeImage();
  if (activeMode === 'camera') await analyzeCameraFrame();
});
window.addEventListener('beforeunload', stopCamera);

elements.cameraButton.disabled = true;
loadModel();
