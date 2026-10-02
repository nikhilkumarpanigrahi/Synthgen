// State Management
const state = {
  currentView: 'datasets',
  activeCategory: 'real',
  activeClass: '',
  inventory: null,
  lossChart: null,
  valAccChart: null
};

// DOM Init
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initExplorer();
  initPipeline();
  initAudit();
  initBenchmark();
  initUpload();
  initExport();
  
  // Initial API loads
  loadInventory();
  loadExplorerSamples();
  loadAuditData();
  loadBenchmarkData();
});

// Navigation Handling
function initNavigation() {
  const items = document.querySelectorAll('.nav-item');
  items.forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      items.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
      const activePanel = document.getElementById(`view-${view}`);
      if (activePanel) activePanel.classList.add('active');
      
      state.currentView = view;
      if (view === 'benchmarks') {
        setTimeout(renderBenchmarkCharts, 50);
      }
    });
  });
}

// Inventory & Telemetry
async function loadInventory() {
  try {
    const res = await fetch('/api/dataset/inventory');
    const data = await res.json();
    state.inventory = data;

    // Sidebar
    document.getElementById('sideRealCount').textContent = data.real.total;
    document.getElementById('sideSynthCount').textContent = data.synthetic.total;
    document.getElementById('sideImbalance').textContent = `1 : ${data.real.imbalance_ratio}`;

    // Metric Strip
    const counts = state.activeCategory === 'real' ? data.real.counts : data.synthetic.counts;
    document.getElementById('cntPothole').textContent = counts.pothole || 0;
    document.getElementById('cntCrack').textContent = counts.surface_crack || 0;
    document.getElementById('cntNormal').textContent = counts.normal_road || 0;
    document.getElementById('cntTotalAll').textContent = state.activeCategory === 'real' ? data.real.total : data.synthetic.total;
  } catch (err) {
    console.error('Failed to load inventory', err);
  }
}

// Dataset Explorer
function initExplorer() {
  // Category switch
  const segBtns = document.querySelectorAll('.seg-btn');
  segBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      segBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeCategory = btn.dataset.cat;
      loadInventory();
      loadExplorerSamples();
    });
  });

  // Class filter
  const filter = document.getElementById('classFilter');
  filter.addEventListener('change', (e) => {
    state.activeClass = e.target.value;
    loadExplorerSamples();
  });

  // Refresh btn
  document.getElementById('btnRefreshDataset').addEventListener('click', () => {
    loadInventory();
    loadExplorerSamples();
  });
}

async function loadExplorerSamples() {
  const grid = document.getElementById('explorerGrid');
  grid.innerHTML = `
    <div class="skeleton-card"></div>
    <div class="skeleton-card"></div>
    <div class="skeleton-card"></div>
    <div class="skeleton-card"></div>
    <div class="skeleton-card"></div>
    <div class="skeleton-card"></div>
  `;

  try {
    const url = `/api/dataset/samples?category=${state.activeCategory}${state.activeClass ? `&class_name=${state.activeClass}` : ''}&limit=48`;
    const res = await fetch(url);
    const data = await res.json();

    document.getElementById('datasetCountDisplay').textContent = `Showing ${data.items.length} of ${data.total} items`;

    if (data.items.length === 0) {
      grid.innerHTML = `<div class="placeholder-box">No samples found for category: ${state.activeCategory} with filter: ${state.activeClass || 'all'}</div>`;
      return;
    }

    grid.innerHTML = '';
    data.items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'sample-card';
      card.innerHTML = `
        <img class="sample-thumbnail" src="${item.url}" alt="${item.filename}" loading="lazy" />
        <div class="sample-footer">
          <span class="badge-label ${item.class}">${item.class}</span>
          <span class="sample-meta-sub">${Math.round(item.size_bytes / 1024)} KB</span>
        </div>
      `;
      card.addEventListener('click', () => openImageInspector(item));
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<div class="placeholder-box" style="color: var(--color-danger);">Error fetching dataset samples.</div>`;
  }
}

// Pipeline Dispatcher
function initPipeline() {
  const slider = document.getElementById('cfgScaleSlider');
  const display = document.getElementById('cfgScaleVal');
  slider.addEventListener('input', (e) => display.textContent = parseFloat(e.target.value).toFixed(1));

  const launchBtn = document.getElementById('btnLaunchJob');
  const progWrap = document.getElementById('jobProgressWrapper');
  const progBar = document.getElementById('jobProgressBar');
  const progText = document.getElementById('jobProgressText');
  const durationText = document.getElementById('jobDurationText');
  const statusBadge = document.getElementById('jobStatusBadge');
  const logBox = document.getElementById('terminalLogBox');
  const resultsGrid = document.getElementById('outputPreviewGrid');

  launchBtn.addEventListener('click', async () => {
    const arch = document.getElementById('cfgArchitecture').value;
    const targetClass = document.getElementById('cfgTargetClass').value;
    const count = parseInt(document.getElementById('cfgBatchCount').value, 10);
    const cfgScale = parseFloat(slider.value);
    const seed = parseInt(document.getElementById('cfgSeed').value, 10);
    const resolution = parseInt(document.getElementById('cfgResolution').value, 10);

    launchBtn.disabled = true;
    statusBadge.textContent = 'RUNNING';
    statusBadge.className = 'status-badge running';
    progWrap.style.display = 'block';
    progBar.style.width = '10%';
    progText.textContent = '10%';
    
    appendLog(logBox, `[DISPATCH] Creating job on worker node...`);

    try {
      // 1. Create job
      const createRes = await fetch('/api/jobs/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          architecture: arch,
          target_class: targetClass,
          count: count,
          cfg_scale: cfgScale,
          seed: seed,
          resolution: resolution
        })
      });
      const createData = await createRes.json();
      const jobId = createData.job_id;
      appendLog(logBox, `[JOB_ID] ${jobId} allocated. Running PyTorch tensors...`);

      // 2. Execute job
      const runRes = await fetch(`/api/jobs/${jobId}/run`, { method: 'POST' });
      const jobResult = await runRes.json();

      progBar.style.width = '100%';
      progText.textContent = '100%';
      durationText.textContent = `Completed in ${jobResult.duration_ms}ms`;
      statusBadge.textContent = 'COMPLETED';
      statusBadge.className = 'status-badge completed';
      launchBtn.disabled = false;

      jobResult.logs.forEach(l => appendLog(logBox, l));

      // Display samples
      resultsGrid.innerHTML = '';
      jobResult.samples.forEach(s => {
        const c = document.createElement('div');
        c.className = 'sample-card';
        c.innerHTML = `
          <img class="sample-thumbnail" src="${s.url}" alt="${s.id}" />
          <div class="sample-footer">
            <span class="badge-label ${s.class}">${s.class}</span>
            <span class="sample-meta-sub">${s.resolution}</span>
          </div>
        `;
        c.addEventListener('click', () => openImageInspector({
          filename: s.filename,
          class: s.class,
          url: s.url,
          category: 'synthetic',
          size_bytes: 45000,
          created_at: Date.now() / 1000
        }));
        resultsGrid.appendChild(c);
      });

      // Update telemetry
      loadInventory();
      loadAuditData();

    } catch (err) {
      launchBtn.disabled = false;
      statusBadge.textContent = 'FAILED';
      appendLog(logBox, `[ERROR] Pipeline run failed: ${err.message}`);
    }
  });
}

function appendLog(box, line) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.textContent = line;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// Quality & Audit
function initAudit() {
  document.getElementById('btnRunAudit').addEventListener('click', () => loadAuditData());
}

async function loadAuditData() {
  try {
    const res = await fetch('/api/quality/audit');
    const data = await res.json();

    document.getElementById('valFid').textContent = data.metrics.fid.value;
    document.getElementById('tagFid').textContent = data.metrics.fid.status;

    document.getElementById('valIS').textContent = data.metrics.inception_score.value;
    document.getElementById('tagIS').textContent = data.metrics.inception_score.status;

    document.getElementById('valDiv').textContent = data.metrics.diversity_index.value;
    document.getElementById('tagDiv').textContent = data.metrics.diversity_index.status;

    document.getElementById('valPriv').textContent = data.metrics.privacy_retention.value;
    document.getElementById('tagPriv').textContent = data.metrics.privacy_retention.status;

    // Table
    const tbody = document.getElementById('archTableBody');
    tbody.innerHTML = '';
    data.architecture_benchmarks.forEach(b => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${b.architecture}</strong></td>
        <td>${b.fid}</td>
        <td>${b.diversity}</td>
        <td>${b.throughput_fps} fps</td>
        <td>${b.memory_mb} MB</td>
        <td><span class="status-indicator">STANDARDIZED</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Memorization Pairs
    const memGrid = document.getElementById('memorizationGrid');
    if (data.memorization_audit && data.memorization_audit.length > 0) {
      memGrid.innerHTML = '';
      data.memorization_audit.forEach((p, idx) => {
        const card = document.createElement('div');
        card.className = 'pair-card';
        card.innerHTML = `
          <div class="pair-images">
            <div>
              <img src="${p.synthetic_image}" alt="Synth" />
              <div style="font-size:9px; color:var(--text-muted); text-align:center;">Synthetic Sample</div>
            </div>
            <div>
              <img src="${p.closest_real_image}" alt="Real" />
              <div style="font-size:9px; color:var(--text-muted); text-align:center;">Nearest Real Match</div>
            </div>
          </div>
          <div class="pair-meta">
            <span>Euclidean Distance: ${p.euclidean_distance}</span>
            <span style="color:var(--color-success);">${p.memorization_status}</span>
          </div>
        `;
        memGrid.appendChild(card);
      });
    } else {
      memGrid.innerHTML = `<div class="placeholder-box" style="grid-column: 1/-1;">Generate more synthetic samples to execute nearest-neighbor memorization audit.</div>`;
    }
  } catch (err) {
    console.error('Audit failed to load', err);
  }
}

// Benchmarks
function initBenchmark() {
  document.getElementById('btnTrainBenchmark').addEventListener('click', async () => {
    const btn = document.getElementById('btnTrainBenchmark');
    btn.disabled = true;
    btn.textContent = 'Training & Evaluating Models...';
    try {
      const res = await fetch('/api/benchmark/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backbone: 'ResNet-18', epochs: 15, learning_rate: 0.001 })
      });
      const data = await res.json();
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Re-Train & Evaluate CNN Models
      `;
      applyBenchmarkData(data);
    } catch (err) {
      btn.disabled = false;
      console.error(err);
    }
  });
}

async function loadBenchmarkData() {
  try {
    const res = await fetch('/api/benchmark/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backbone: 'ResNet-18', epochs: 15, learning_rate: 0.001 })
    });
    const data = await res.json();
    applyBenchmarkData(data);
  } catch (err) {
    console.error(err);
  }
}

function applyBenchmarkData(data) {
  // Deltas
  const s = data.summary_comparison;
  document.getElementById('baseAccVal').textContent = `${s.accuracy.baseline}%`;
  document.getElementById('augAccVal').textContent = `${s.accuracy.augmented}%`;
  document.getElementById('gainAcc').textContent = s.accuracy.delta;

  document.getElementById('baseF1Val').textContent = `${s.macro_f1.baseline}%`;
  document.getElementById('augF1Val').textContent = `${s.macro_f1.augmented}%`;
  document.getElementById('gainF1').textContent = s.macro_f1.delta;

  document.getElementById('basePotholeVal').textContent = `${s.rare_defect_recall.baseline}%`;
  document.getElementById('augPotholeVal').textContent = `${s.rare_defect_recall.augmented}%`;
  document.getElementById('gainPothole').textContent = `${s.rare_defect_recall.delta} Gain`;

  // Tables
  renderReportTable('tableReportBaseline', data.classification_reports.baseline);
  renderReportTable('tableReportAugmented', data.classification_reports.augmented);

  // Matrices
  renderMatrix('matrixBaseline', data.confusion_matrices.baseline);
  renderMatrix('matrixAugmented', data.confusion_matrices.augmented);

  // Charts
  renderLossChart(data.training_curves);
  renderValAccChart(data.training_curves);
}

function renderReportTable(tableId, report) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = '';
  report.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.class}</td>
      <td>${r.precision}</td>
      <td>${r.recall}</td>
      <td><strong>${r.f1_score}</strong></td>
      <td>${r.support}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderMatrix(containerId, cm) {
  const box = document.getElementById(containerId);
  box.innerHTML = `
    <div class="matrix-cell head">True \\ Pred</div>
    <div class="matrix-cell head">Pothole</div>
    <div class="matrix-cell head">Crack</div>
    <div class="matrix-cell head">Normal</div>
  `;

  cm.matrix.forEach((row, i) => {
    box.innerHTML += `<div class="matrix-cell head">${cm.labels[i].replace('_', ' ')}</div>`;
    row.forEach((val, j) => {
      const isDiag = i === j;
      const isErr = !isDiag && val > 0;
      box.innerHTML += `<div class="matrix-cell ${isDiag ? 'diag' : (isErr ? 'err' : '')}">${val}</div>`;
    });
  });
}

function renderLossChart(curves) {
  const ctx = document.getElementById('chartLossConvergence');
  if (!ctx) return;
  if (state.lossChart) state.lossChart.destroy();

  state.lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: curves.epochs.map(e => `Ep ${e}`),
      datasets: [
        {
          label: 'Baseline (Real Data)',
          data: curves.loss_baseline,
          borderColor: '#8B949E',
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.1
        },
        {
          label: 'Augmented (Real + Synthetic)',
          data: curves.loss_augmented,
          borderColor: '#1F6FEB',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#21262D' }, ticks: { color: '#8B949E', font: { size: 10 } } },
        y: { grid: { color: '#21262D' }, ticks: { color: '#8B949E', font: { size: 10 } } }
      },
      plugins: {
        legend: { labels: { color: '#F0F6FC', font: { size: 10 } } }
      }
    }
  });
}

function renderValAccChart(curves) {
  const ctx = document.getElementById('chartValAcc');
  if (!ctx) return;
  if (state.valAccChart) state.valAccChart.destroy();

  state.valAccChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: curves.epochs.map(e => `Ep ${e}`),
      datasets: [
        {
          label: 'Baseline Validation Accuracy (%)',
          data: curves.val_acc_baseline,
          borderColor: '#D29922',
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.1
        },
        {
          label: 'Augmented Validation Accuracy (%)',
          data: curves.val_acc_augmented,
          borderColor: '#2EA043',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#21262D' }, ticks: { color: '#8B949E', font: { size: 10 } } },
        y: { min: 40, max: 100, grid: { color: '#21262D' }, ticks: { color: '#8B949E', font: { size: 10 } } }
      },
      plugins: {
        legend: { labels: { color: '#F0F6FC', font: { size: 10 } } }
      }
    }
  });
}

// Upload & Modal
function initUpload() {
  const modal = document.getElementById('uploadModal');
  const openBtn = document.getElementById('btnUploadModal');
  const closeBtn = document.getElementById('btnCloseUpload');
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('fileInput');
  const feedback = document.getElementById('uploadFeedback');

  openBtn.addEventListener('click', () => modal.classList.add('open'));
  closeBtn.addEventListener('click', () => modal.classList.remove('open'));
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('open'); });

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFile);

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--color-primary)'; });
  dropzone.addEventListener('dragleave', () => dropzone.style.borderColor = 'var(--border-default)');
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--border-default)';
    if (e.dataTransfer.files.length) handleUploadFile(e.dataTransfer.files[0]);
  });

  function handleFile() {
    if (fileInput.files.length) handleUploadFile(fileInput.files[0]);
  }

  async function handleUploadFile(file) {
    const targetClass = document.getElementById('uploadClassSelect').value;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_class', targetClass);

    feedback.style.display = 'block';
    feedback.textContent = `Uploading ${file.name}...`;

    try {
      const res = await fetch('/api/dataset/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.status === 'SUCCESS') {
        feedback.textContent = `Successfully ingested ${file.name} into ${targetClass}.`;
        loadInventory();
        loadExplorerSamples();
        setTimeout(() => { modal.classList.remove('open'); feedback.style.display = 'none'; }, 1000);
      }
    } catch (err) {
      feedback.textContent = 'Upload failed. Ensure image is JPG/PNG.';
    }
  }
}

// Image Inspector Modal
function openImageInspector(item) {
  const modal = document.getElementById('inspectorModal');
  const title = document.getElementById('inspectorTitle');
  const img = document.getElementById('inspectorImage');
  const meta = document.getElementById('inspectorMeta');

  title.textContent = `Sample: ${item.filename}`;
  img.src = item.url;
  meta.innerHTML = `
    <div><strong>Class:</strong> ${item.class}</div>
    <div><strong>Source Split:</strong> ${item.category}</div>
    <div><strong>Format:</strong> JPEG RGB (256x256)</div>
    <div><strong>File Size:</strong> ${Math.round(item.size_bytes / 1024)} KB</div>
    <div><strong>Resource URI:</strong> ${item.url}</div>
    <div style="margin-top: 10px;">
      <a href="${item.url}" download="${item.filename}" class="action-btn" style="display:inline-flex;">Download File</a>
    </div>
  `;

  modal.classList.add('open');
  document.getElementById('btnCloseInspector').onclick = () => modal.classList.remove('open');
  modal.onclick = (e) => { if (e.target === modal) modal.classList.remove('open'); };
}

// Export Handler
function initExport() {
  document.getElementById('btnHeaderExport').addEventListener('click', () => {
    window.location.href = '/api/export/archive?format=raw';
  });
}
