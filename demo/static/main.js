// ================== GLOBAL STATE ==================
let evtSource = null;
let audioEnabled = true;   // sound enabled by default
let logCounter = 0;

// dataset for offline + live metrics (STEAD / INGV / USGS_real)
let currentDataset = 'STEAD';

// user location (default)
let userLat = 16.8661;
let userLon = 96.1951;

// DOM references
const alertEl = document.getElementById('alertBanner');
const alertMsgEl = document.getElementById('alertMessage');
const alertCountdownEl = document.getElementById('alertCountdown');
const placeEl = document.getElementById('placeName');
const intensityBadge = document.getElementById('intensityBadge');
const shakePopup = document.getElementById('shakePopup');
const shakeDetailsEl = document.getElementById('shakeDetails');

const alertSoundDanger = document.getElementById('alertSoundDanger');
const alertSoundWarning = document.getElementById('alertSoundWarning');

let countdownTimer = null;
let countdownTarget = null;
let lastAlertLevel = 'safe';

// ================== MAP SETUP ==================
const map = L.map('map', { zoomControl: true }).setView([userLat, userLon], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18
}).addTo(map);

let epicMarker = null, userMarker = null, distCircle = null;

// ================== MAGNITUDE TIMELINE (Chart.js) ==================
let magTimeline = [];
let magIndex = [];
const ctxMag = document.getElementById('chartMagTimeline').getContext('2d');
const magChart = new Chart(ctxMag, {
  type: 'line',
  data: {
    labels: magIndex,
    datasets: [{
      label: 'Predicted Magnitude',
      data: magTimeline,
      borderWidth: 1
    }]
  },
  options: {
    responsive: true,
    animation: false,
    scales: {
      x: { title: { display: true, text: 'Event index' } },
      y: { title: { display: true, text: 'Magnitude' } }
    }
  }
});

function updateMagChart(mag) {
  magTimeline.push(mag);
  magIndex.push(magIndex.length + 1);
  if (magTimeline.length > 200) {
    magTimeline.shift();
    magIndex.shift();
  }
  magChart.update();
}

// ================== METRICS ==================
let nMag = 0, sumSqMag = 0;
let nDist = 0, sumSqDist = 0;
let nSP = 0, sumSqSP = 0;

// classification counts
let TP = 0, FP = 0, TN = 0, FN = 0;

function updateLiveClassificationMetrics() {
  const total = TP + FP + TN + FN;
  const acc = total ? (TP + TN) / total : null;
  const prec = (TP + FP) ? TP / (TP + FP) : null;
  const rec = (TP + FN) ? TP / (TP + FN) : null;
  const f1 = (prec && rec) ? (2 * prec * rec / (prec + rec)) : null;

  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = (v == null) ? '—' : v.toFixed(3);
  };

  setVal('m_accuracy_live', acc);
  setVal('m_precision_live', prec);
  setVal('m_recall_live', rec);
  setVal('m_f1_live', f1);
}

function clearLiveMetrics() {
  [
    'm_accuracy_live', 'm_precision_live', 'm_recall_live', 'm_f1_live',
    'm_mag_rmse_live', 'm_dist_rmse_live', 'm_sp_rmse_live'
  ].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });
}

// ================== UTILS ==================
function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function setIntensity(mag, distKm) {
  let level = 'Low';
  let cls = 'badge-low';

  if (mag >= 4.5 && distKm < 200) { level = 'Moderate'; cls = 'badge-med'; }
  if (mag >= 5.5 && distKm < 150) { level = 'Strong'; cls = 'badge-high'; }
  if (mag >= 6.5 && distKm < 120) { level = 'Severe'; cls = 'badge-severe'; }

  intensityBadge.textContent = level;
  intensityBadge.className = 'badge ' + cls;
}

// reverse geocoding
const placeCache = {};
async function reverseGeocode(lat, lon) {
  const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
  if (placeCache[key] !== undefined) return placeCache[key];

  const fallback = null;
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`;
    const res = await fetch(url, { headers: { 'Accept-Language': 'en' } });
    if (!res.ok) {
      placeCache[key] = fallback;
      return fallback;
    }
    const data = await res.json();
    let name = data.display_name;
    if (!name) {
      placeCache[key] = fallback;
      return fallback;
    }
    placeCache[key] = name;
    return name;
  } catch {
    placeCache[key] = fallback;
    return fallback;
  }
}

// ================== MAP UPDATE ==================
async function updateMap(epic, distUserKm) {
  const lat = epic.lat;
  const lon = epic.lon;

  if (!epicMarker) {
    epicMarker = L.marker([lat, lon]).addTo(map);
  } else {
    epicMarker.setLatLng([lat, lon]);
  }

  if (!userMarker) {
    userMarker = L.circleMarker([userLat, userLon], { radius: 6 }).addTo(map);
  } else {
    userMarker.setLatLng([userLat, userLon]);
  }

  const radius_m = Math.max(50, (distUserKm || 10) * 1000);
  if (!distCircle) {
    distCircle = L.circle([lat, lon], {
      radius: radius_m,
      color: 'red',
      fillOpacity: 0.06
    }).addTo(map);
  } else {
    distCircle.setLatLng([lat, lon]);
    distCircle.setRadius(radius_m);
  }

  const group = L.featureGroup([epicMarker, userMarker]);
  map.fitBounds(group.getBounds().pad(0.4));

  const place = await reverseGeocode(lat, lon);
  const coordInfo = `Lat ${lat.toFixed(2)}, Lon ${lon.toFixed(2)}`;
  const label = place ? `${place} (${coordInfo})` : coordInfo;

  placeEl.textContent = label;

  epicMarker.bindPopup(`Epicenter<br>${label}`);
  epicMarker.bindTooltip(label, {
    permanent: true,
    direction: 'right',
    offset: [8, 0],
    className: 'epicenter-label'
  }).openTooltip();
}

// ================== LOG TABLE ==================
function addLogRow(obj, distUserKm, level) {
  const tbody = document.querySelector('#logTable tbody');
  logCounter += 1;

  const tr = document.createElement('tr');
  if (level === 'warning' || level === 'danger') {
    tr.classList.add('alert-row');
  }

  tr.innerHTML = `
    <td>${logCounter}</td>
    <td>${obj.magnitude.toFixed(2)}</td>
    <td>${distUserKm.toFixed(1)}</td>
    <td>${(obj.eta_seconds || 0).toFixed(1)}</td>
    <td>${level.toUpperCase()}</td>
  `;
  tbody.appendChild(tr);

  const container = document.querySelector('.log-card');
  if (container) container.scrollTop = container.scrollHeight;
}

// ================== VOICE + AUDIO ==================
function speak(text) {
  if (!audioEnabled) return;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    window.speechSynthesis.speak(u);
  } catch {}
}

function stopAllAudio() {
  try {
    if (alertSoundDanger) {
      alertSoundDanger.pause();
      alertSoundDanger.currentTime = 0;
    }
    if (alertSoundWarning) {
      alertSoundWarning.pause();
      alertSoundWarning.currentTime = 0;
    }
  } catch {}
  try {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  } catch {}
}

// ================== ALERT LOGIC ==================
function computeAlertLevel(mag) {
  if (mag < 1.0) return 'safe';
  if (mag <= 2) return 'watch';
  if (mag <= 3) return 'warning';
  return 'danger';
}

function bannerTextForLevel(level, mag, distKm, etaSec) {
  const m = mag.toFixed(2);
  const d = distKm.toFixed(1);
  const e = etaSec > 0 ? etaSec.toFixed(1) : '—';

  if (level === 'safe')
    return `‼ SAFE — M ${m} at ${d} km, ETA ≈ ${e} s. No dangerous shaking expected.`;

  if (level === 'watch')
    return `‼ WATCH — M ${m} at ${d} km, ETA ≈ ${e} s. Small earthquake detected. No strong shaking expected near your location.`;

  if (level === 'warning')
    return `‼ WARNING — Moderate earthquake detected. M ${m}, Distance ${d} km, ETA ≈ ${e} s. You may feel shaking. Stay alert and move to a safe place if needed.`;

  if (level === 'danger')
    return `‼ DANGER — Strong earthquake approaching! M ${m}, Distance ${d} km, ETA ≈ ${e} s. Take protective action NOW: drop to the ground, take cover under sturdy furniture, and protect your head and neck.`;

  return '‼ Monitoring seismic activity…';
}

function stopCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = null;
  alertCountdownEl.textContent = '';
}

function startCountdown(etaSec) {
  if (!etaSec || etaSec <= 0 || etaSec > 60) {
    alertCountdownEl.textContent = '';
    return;
  }
  countdownTarget = Date.now() / 1000 + etaSec;

  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    const remain = countdownTarget - Date.now() / 1000;
    if (remain <= 0) {
      alertCountdownEl.textContent = 'Shaking now!';
      stopCountdown();
      return;
    }
    alertCountdownEl.textContent = `Shaking in ${remain.toFixed(1)} s`;
  }, 200);
}

function showShakePopup(mag, distKm, etaSec, level) {
  if (!shakePopup) return;
  const m = mag.toFixed(2);
  const d = distKm.toFixed(1);
  const e = etaSec > 0 ? etaSec.toFixed(1) : '—';

  shakeDetailsEl.textContent =
    `Level: ${level.toUpperCase()} — M ${m}, Dist ${d} km, ETA ≈ ${e} s.`;

  shakePopup.classList.remove('popup-warning', 'popup-danger');
  if (level === 'warning') {
    shakePopup.classList.add('popup-warning');
  } else if (level === 'danger') {
    shakePopup.classList.add('popup-danger');
  }

  shakePopup.classList.remove('hidden');
  setTimeout(() => {
    shakePopup.classList.add('hidden');
  }, 3000);
}

function updateAlertUI(level, text, etaSec, mag, distKm) {
  if (level === 'safe' || level === 'watch') {
    stopAllAudio();
  }

  alertEl.classList.remove('alert-safe', 'alert-watch', 'alert-warning', 'alert-danger');
  alertEl.classList.add(`alert-${level}`);
  alertMsgEl.textContent = text;

  const mapDiv = document.getElementById('map');
  if (mapDiv) {
    mapDiv.classList.remove('shake');
    if (level === 'warning' || level === 'danger') {
      mapDiv.classList.add('shake');
    }
  }

  if (level === 'warning' || level === 'danger') {
    startCountdown(etaSec);
  } else {
    stopCountdown();
  }

  if ((level === 'warning' || level === 'danger') && level !== lastAlertLevel) {
    if (audioEnabled) {
      const snd = (level === 'danger') ? alertSoundDanger : alertSoundWarning;
      if (snd) {
        try {
          snd.currentTime = 0;
          snd.play().catch(() => {});
        } catch {}
      }
    }

    const voiceText = (level === 'danger')
      ? 'Danger. Strong earthquake approaching. Take protective action now: drop, take cover, and protect your head and neck.'
      : 'Warning. Moderate earthquake detected. You may feel shaking.';
    speak(voiceText);

    showShakePopup(mag, distKm, etaSec, level);
  }

  lastAlertLevel = level;
}

// ================== WAVEFORM PREVIEW ==================
const wfCanvas = document.getElementById('waveformCanvas');
const wfCtx = wfCanvas.getContext('2d');
const WF_POINTS = 400;
let wfData = new Array(WF_POINTS).fill(0.5);
let currentWaveAmp = 0.15;
let waveformActive = true;

function drawWaveform() {
  const w = wfCanvas.width;
  const h = wfCanvas.height;
  wfCtx.clearRect(0, 0, w, h);

  wfCtx.strokeStyle = '#4b5563';
  wfCtx.lineWidth = 1;
  wfCtx.beginPath();
  wfCtx.moveTo(0, h / 2);
  wfCtx.lineTo(w, h / 2);
  wfCtx.stroke();

  wfCtx.strokeStyle = '#22c55e';
  wfCtx.lineWidth = 2;
  wfCtx.beginPath();
  wfData.forEach((val, idx) => {
    const x = (idx / (WF_POINTS - 1)) * w;
    const y = h / 2 - (val - 0.5) * (h * 0.8);
    if (idx === 0) wfCtx.moveTo(x, y);
    else wfCtx.lineTo(x, y);
  });
  wfCtx.stroke();
}

function updateWaveAmpForLevel(level) {
  if (level === 'safe') currentWaveAmp = 0.18;
  else if (level === 'watch') currentWaveAmp = 0.32;
  else if (level === 'warning') currentWaveAmp = 0.55;
  else if (level === 'danger') currentWaveAmp = 0.85;
}

function waveformLoop() {
  const w = wfCanvas.width;
  const h = wfCanvas.height;
  if (w === 0 || h === 0) {
    // not ready
  } else {
    if (waveformActive) {
      let val = currentWaveAmp + (Math.random() - 0.5) * 0.12;
      val = Math.max(0.0, Math.min(1.0, val));
      wfData.push(val);
      if (wfData.length > WF_POINTS) wfData.shift();
    }
    drawWaveform();
  }
  requestAnimationFrame(waveformLoop);
}
requestAnimationFrame(waveformLoop);

// ================== SSE CONNECTION ==================
function disconnectSSE() {
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
}

function connectSSE() {
  if (evtSource) return;
  evtSource = new EventSource('/stream');

  document.getElementById('statusText').textContent = 'Running';

  evtSource.onmessage = async (e) => {
    try {
      const obj = JSON.parse(e.data);

      let mag = Number(obj.magnitude);
      if (!isFinite(mag)) mag = 0.0;

      const epic = obj.epicenter;
      const eta = Number(obj.eta_seconds || 0);

      let distUserKm = Number(obj.distance_km || 0);
      if (epic) {
        distUserKm = haversine(epic.lat, epic.lon, userLat, userLon);
      }

      document.getElementById('mag').textContent = mag.toFixed(2);
      document.getElementById('dist').textContent = distUserKm.toFixed(1);
      document.getElementById('eta').textContent = eta.toFixed(1);
      document.getElementById('latency').textContent =
        obj.inference_ms ? `${obj.inference_ms} ms` : '—';

      if (epic) await updateMap(epic, distUserKm);
      setIntensity(mag, distUserKm);

      const level = computeAlertLevel(mag);
      const msg = bannerTextForLevel(level, mag, distUserKm, eta);
      updateAlertUI(level, msg, eta, mag, distUserKm);
      updateWaveAmpForLevel(level);

      addLogRow(obj, distUserKm, level);
      updateMagChart(mag);

      // RMSE metrics
      if (obj.true_mag != null) {
        const errM = mag - Number(obj.true_mag);
        sumSqMag += errM * errM;
        nMag += 1;
        const el = document.getElementById('m_mag_rmse_live');
        if (el) {
          el.textContent = Math.sqrt(sumSqMag / nMag).toFixed(3);
        }
      }
      if (obj.true_dist_km != null && obj.distance_km != null) {
        const errD = Number(obj.distance_km) - Number(obj.true_dist_km);
        sumSqDist += errD * errD;
        nDist += 1;
        const el = document.getElementById('m_dist_rmse_live');
        if (el) {
          el.textContent = Math.sqrt(sumSqDist / nDist).toFixed(3);
        }
      }
      if (obj.sp_true_sec != null && obj.sp_pred_sec != null) {
        const errS = Number(obj.sp_pred_sec) - Number(obj.sp_true_sec);
        sumSqSP += errS * errS;
        nSP += 1;
        const el = document.getElementById('m_sp_rmse_live');
        if (el) {
          el.textContent = Math.sqrt(sumSqSP / nSP).toFixed(3);
        }
      }

      // classification
      if (obj.true_label != null && obj.pred_label != null) {
        const t = Number(obj.true_label);
        const p = Number(obj.pred_label);
        if (t === 1 && p === 1) TP++;
        else if (t === 0 && p === 1) FP++;
        else if (t === 0 && p === 0) TN++;
        else if (t === 1 && p === 0) FN++;
        updateLiveClassificationMetrics();
      }

    } catch (err) {
      console.error('SSE parse error', err);
    }
  };

  evtSource.addEventListener('done', () => {
    disconnectSSE();
    stopCountdown();
    stopAllAudio();
    waveformActive = false;
    document.getElementById('statusText').textContent = 'Stopped';
  });

  evtSource.onerror = () => {
    console.warn('SSE error');
  };
}

// ================== METRICS LOADING ==================
async function loadMetricsForCurrentDataset() {
  try {
    const ds = currentDataset;
    const res = await fetch(`/metrics?dataset=${encodeURIComponent(ds)}`);
    if (!res.ok) return;
    const j = await res.json();

    const setVal = (id, v) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = v != null ? Number(v).toFixed(3) : '—';
    };

    // offline classification
    const clf = j.classification || {};
    setVal('m_accuracy', clf.accuracy);
    setVal('m_precision', clf.precision);
    setVal('m_recall', clf.recall);
    setVal('m_f1', clf.f1);

    // NEW: offline regression (from metrics_cross.json)
    const reg = j.regression || {};
    setVal('m_mag_mae_offline', reg.mag_mae);
    setVal('m_dist_mae_offline', reg.dist_mae_km);
    setVal('m_sp_mae_offline', reg.sp_mae_s);

    const dsName = j.display_name || j.dataset || ds;
    document.getElementById('m_dataset_name').textContent = dsName;

    if (j.operational && j.operational.inference_ms != null) {
      document.getElementById('m_latency2').textContent =
        j.operational.inference_ms.toString();
    } else {
      document.getElementById('m_latency2').textContent = '—';
    }
  } catch (e) {
    console.warn('metrics fetch error', e);
  }
}

// ================== BUTTON HANDLERS ==================
document.getElementById('btnStart').addEventListener('click', async () => {
  // reset live metrics for new stream
  nMag = nDist = nSP = 0;
  sumSqMag = sumSqDist = sumSqSP = 0;
  TP = FP = TN = FN = 0;
  logCounter = 0;

  [
    'm_mag_rmse_live', 'm_dist_rmse_live', 'm_sp_rmse_live',
    'm_accuracy_live', 'm_precision_live', 'm_recall_live', 'm_f1_live'
  ].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });

  const tbody = document.querySelector('#logTable tbody');
  tbody.innerHTML = '';

  magTimeline.length = 0;
  magIndex.length = 0;
  magChart.update();

  wfData = new Array(WF_POINTS).fill(0.5);
  waveformActive = true;

  connectSSE();
  const res = await fetch('/start_sim', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset: currentDataset })
  });
  if (!res.ok) {
    const j = await res.json().catch(() => null);
    alert(j && j.message ? j.message : 'Failed to start simulation');
    disconnectSSE();
    waveformActive = false;
    document.getElementById('statusText').textContent = 'Stopped';
    return;
  }

  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnStop').disabled = false;
});

document.getElementById('btnStop').addEventListener('click', async () => {
  await fetch('/stop_sim', { method: 'POST' });

  disconnectSSE();
  stopCountdown();
  stopAllAudio();
  waveformActive = false;
  document.getElementById('statusText').textContent = 'Stopped';

  const mapDiv = document.getElementById('map');
  if (mapDiv) mapDiv.classList.remove('shake');

  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled = true;
});

document.getElementById('btnSound').addEventListener('click', () => {
  audioEnabled = !audioEnabled;
  document.getElementById('btnSound').textContent =
    audioEnabled ? 'Sound Enabled ✓' : 'Sound Disabled ✕';
});

document.getElementById('btnPerf').addEventListener('click', async () => {
  const panel = document.getElementById('perfPanel');
  if (panel.classList.contains('hidden')) {
    await loadMetricsForCurrentDataset();
    panel.classList.remove('hidden');
  } else {
    panel.classList.add('hidden');
  }
});

// user location change
document.getElementById('userLocation').addEventListener('change', (e) => {
  const [latStr, lonStr] = e.target.value.split(',');
  userLat = parseFloat(latStr);
  userLon = parseFloat(lonStr);
  if (userMarker) userMarker.setLatLng([userLat, userLon]);
  map.setView([userLat, userLon], 5);
});

// dataset selection change (affects streaming + metrics)
document.getElementById('datasetSelect').addEventListener('change', async (e) => {
  currentDataset = e.target.value;
  const panel = document.getElementById('perfPanel');
  if (!panel.classList.contains('hidden')) {
    await loadMetricsForCurrentDataset();
  }
});

// initial safe banner + waveform + status
updateAlertUI('safe', '‼ SAFE — Monitoring seismic activity…', 0, 0, 999);
drawWaveform();
document.getElementById('statusText').textContent = 'Stopped';
