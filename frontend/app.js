// Synthetix ML Platform — Frontend Logic Engine
const state = {
  currentView: 'datasets',
  activeCategory: 'real',
  activeClass: '',
  inventory: null,
  availableDatasets: [],
  lossChart: null,
  valAccChart: null,
  activeJobId: null
};

// Application Initialization
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initGuidedWorkflow();
  initDatasetSwitcher();
  initExplorer();
  initPipeline();
  initAudit();
  initBenchmark();
  initUpload();
  initExport();
  initInspector();

  // Load baseline system status & datasets
  loadSystemStatus();
  loadAvailableDatasets();
});

// 1. Navigation & Workflow Controller
function switchView(viewName) {
  state.currentView = viewName;

  // Sidebar buttons
  document.querySelectorAll('.nav-item').forEach(b => {
    b.classList.toggle('active', b.dataset.view === viewName);
  });

  // Top Workflow Stepper
  document.querySelectorAll('.flow-step').forEach(s => {
    s.classList.toggle('active', s.dataset.flow === viewName);
  });

  // Main Panels
  document.querySelectorAll('.view-panel').forEach(p => {
    p.classList.toggle('active', p.id === `view-${viewName}`);
  });

  if (viewName === 'benchmarks' && state.lastBenchmarkData) {
    setTimeout(renderBenchmarkCharts, 50);
  }
}

function initNavigation() {
  const items = document.querySelectorAll('.nav-item');
  items.forEach(btn => {
    btn.addEventListener('click', () => {
      switchView(btn.dataset.view);
    });
  });
}

function initGuidedWorkflow() {
  // Stepper clicks
  document.querySelectorAll('.flow-step').forEach(step => {
    step.addEventListener('click', () => {
      switchView(step.dataset.flow);
    });
  });

  // Dismiss quick guide banner
  const btnDismiss = document.getElementById('btnDismissGuide');
  const banner = document.getElementById('quickGuideBanner');
  if (btnDismiss && banner) {
    btnDismiss.addEventListener('click', () => {
      banner.style.display = 'none';
    });
  }

  // 1-Click Auto Run Complete Experiment
  const btnAutoRun = document.getElementById('btnAutoRunExperiment');
  if (btnAutoRun) {
    btnAutoRun.addEventListener('click', runFullAutoExperiment);
  }
}

async function runFullAutoExperiment() {
  const btn = document.getElementById('btnAutoRunExperiment');
  btn.disabled = true;
  btn.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M12 2a10 10 0 0 1 10 10"></path>
    </svg>
    <span>Running Complete Experiment...</span>
  `;

  try {
    // Step 1: Ensure dataset inventory is loaded
    await loadInventory();
    const rareClass = state.inventory?.analysis?.underrepresented_class || 'pothole';
    
    // Step 2: Switch to Generative Synthesis & launch generation
    switchView('pipelines');
    const targetSelect = document.getElementById('cfgTargetClass');
    if (targetSelect) targetSelect.value = rareClass;
    
    await launchGenerationJob();
    const step2 = document.querySelector('.flow-step[data-flow="pipelines"]');
    if (step2) step2.classList.add('completed');

    // Step 3: Switch to Quality & Audit
    switchView('quality');
    await loadAuditData();
    const step3 = document.querySelector('.flow-step[data-flow="quality"]');
    if (step3) step3.classList.add('completed');

    // Step 4: Switch to Benchmarks and train PyTorch CNNs
    switchView('benchmarks');
    await runClassifierBenchmark();
    const step4 = document.querySelector('.flow-step[data-flow="benchmarks"]');
    if (step4) step4.classList.add('completed');

    // Smooth scroll to empirical research question outcome
    const resultCard = document.getElementById('researchResultCard');
    if (resultCard) {
      resultCard.scrollIntoView({ behavior: 'smooth' });
    }
  } catch (err) {
    console.error('Auto experiment run error:', err);
    alert('Experiment encountered an issue: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
      </svg>
      <span>⚡ 1-Click Run Full Experiment</span>
    `;
  }
}

// 2. System Status & Dataset Switcher
async function loadSystemStatus() {
  try {
    const res = await fetch('/api/system/status');
    const data = await res.json();
    const statusEl = document.getElementById('deviceStatusText');
    if (statusEl && data.engine) {
      statusEl.textContent = `Engine: ${data.engine}`;
    }
  } catch (err) {
    console.warn('System status fetch failed', err);
  }
}

async function loadAvailableDatasets() {
  try {
    const res = await fetch('/api/datasets/available');
    const datasets = await res.json();
    state.availableDatasets = datasets;

    const select = document.getElementById('datasetSelector');
    select.innerHTML = '';

    datasets.forEach(ds => {
      const opt = document.createElement('option');
      opt.value = ds.id;
      opt.textContent = ds.name;
      if (ds.is_active) opt.selected = true;
      select.appendChild(opt);
    });

    // Also load active inventory
    await loadInventory();
    loadExplorerSamples();
    loadAuditData();
  } catch (err) {
    console.error('Failed to load available datasets', err);
  }
}

function initDatasetSwitcher() {
  const select = document.getElementById('datasetSelector');
  select.addEventListener('change', async (e) => {
    const newDsId = e.target.value;
    try {
      select.disabled = true;
      const res = await fetch('/api/datasets/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: newDsId })
      });
      const data = await res.json();
      state.inventory = data;

      // Reset benchmark view cards
      const resultCard = document.getElementById('researchResultCard');
      if (resultCard) resultCard.style.display = 'none';
      const hypothesisBadge = document.getElementById('hypothesisStatusText');
      if (hypothesisBadge) {
        hypothesisBadge.textContent = 'Awaiting Model Benchmark';
        hypothesisBadge.classList.remove('proven');
      }

      await loadInventory();
      loadExplorerSamples();
      loadAuditData();
    } catch (err) {
      console.error('Failed to select dataset', err);
    } finally {
      select.disabled = false;
    }
  });
}

// 3. Inventory & Distribution Telemetry
async function loadInventory() {
  try {
    const res = await fetch('/api/dataset/inventory');
    const data = await res.json();
    state.inventory = data;

    const analysis = data.analysis || {};
    const rareClass = analysis.underrepresented_class || 'minority_class';
    const imbalanceRatio = analysis.imbalance_ratio || 1.0;
    const quota = analysis.recommended_augmentation_quota || 0;

    // Sidebar Telemetry
    document.getElementById('sideDomainId').textContent = data.dataset_id;
    document.getElementById('sideRealCount').textContent = data.total_real;
    document.getElementById('sideSynthCount').textContent = data.total_synthetic;
    document.getElementById('sideImbalance').textContent = `1 : ${imbalanceRatio}`;
    document.getElementById('sideRareClass').textContent = rareClass;
    document.getElementById('sideQuota').textContent = `+${quota} samples`;

    // Header Research Context
    const researchCtx = document.getElementById('researchContextText');
    if (researchCtx) {
      researchCtx.innerHTML = `Active Dataset: <strong>${data.dataset_id}</strong> &bull; Auto-Detected Minority: <span style="color:#D29922; font-weight:600;">${rareClass}</span> (${analysis.minority_count} real vs ${analysis.majority_count} majority, ${imbalanceRatio}x imbalance) &bull; Target Quota: +${quota} to balance`;
    }

    // Dynamic Class Metrics Strip
    renderDynamicClassStrip(data);

    // Populate class filter dropdown
    const filter = document.getElementById('classFilter');
    const prevFilterVal = filter.value;
    filter.innerHTML = '<option value="">All Classes (All)</option>';
    data.classes.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = `Class: ${c}`;
      if (c === prevFilterVal) opt.selected = true;
      filter.appendChild(opt);
    });

    // Populate Generative Target Class dropdown & auto-select rare class
    const targetSelect = document.getElementById('cfgTargetClass');
    if (targetSelect) {
      targetSelect.innerHTML = '';
      data.classes.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = (c === rareClass) ? `${c} ★ (Compensate Rare Scarcity)` : c;
        if (c === rareClass) opt.selected = true;
        targetSelect.appendChild(opt);
      });
    }

    // Populate Upload Target Class dropdown
    const uploadSelect = document.getElementById('uploadClassSelect');
    if (uploadSelect) {
      uploadSelect.innerHTML = '';
      data.classes.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        uploadSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Failed to load inventory', err);
  }
}

function renderDynamicClassStrip(data) {
  const container = document.getElementById('dynamicClassStrip');
  if (!container) return;
  container.innerHTML = '';

  const isReal = state.activeCategory === 'real';
  const counts = isReal ? data.class_counts_real : data.class_counts_synthetic;
  const rareClass = data.analysis ? data.analysis.underrepresented_class : '';

  data.classes.forEach(c => {
    const count = counts[c] || 0;
    const isRare = (c === rareClass);
    
    const card = document.createElement('div');
    card.className = 'strip-item';
    card.innerHTML = `
      <span class="strip-label">Class: ${c}</span>
      <span class="strip-val">${count}</span>
      <span class="strip-sub ${isRare ? 'status-warn' : 'status-ok'}">
        ${isRare ? 'Underrepresented Minority' : 'Standard Class'}
      </span>
    `;
    container.appendChild(card);
  });

  // Total summary card
  const totalCard = document.createElement('div');
  totalCard.className = 'strip-item';
  totalCard.innerHTML = `
    <span class="strip-label">Total In View</span>
    <span class="strip-val">${isReal ? data.total_real : data.total_synthetic}</span>
    <span class="strip-sub">${isReal ? 'Real Ground Truth' : 'Synthetically Generated'}</span>
  `;
  container.appendChild(totalCard);
}

// 4. Dataset Explorer
function initExplorer() {
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

  const filter = document.getElementById('classFilter');
  filter.addEventListener('change', (e) => {
    state.activeClass = e.target.value;
    loadExplorerSamples();
  });

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
  `;

  try {
    let url = `/api/dataset/samples?category=${state.activeCategory}&limit=40`;
    if (state.activeClass) {
      url += `&class_name=${encodeURIComponent(state.activeClass)}`;
    }

    const res = await fetch(url);
    const data = await res.json();

    document.getElementById('datasetCountDisplay').textContent = `Showing ${data.items.length} of ${data.total} samples`;
    grid.innerHTML = '';

    if (data.items.length === 0) {
      grid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 48px; text-align: center; color: var(--text-muted); background: var(--bg-surface); border: 1px dashed var(--border-default); border-radius: var(--radius-md);">
          No samples present in this view. Use the Generative Synthesis pipeline to synthesize new samples for underrepresented classes.
        </div>
      `;
      return;
    }

    data.items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'image-card';
      card.innerHTML = `
        <div class="image-thumb-wrap">
          <img src="${item.url}" alt="${item.filename}" loading="lazy" />
          <span class="card-badge ${item.category === 'real' ? 'badge-real' : 'badge-synth'}">${item.category.toUpperCase()}</span>
        </div>
        <div class="card-meta">
          <span class="card-class-tag">${item.class}</span>
          <span class="card-file-dim">${Math.round(item.size_bytes / 1024)} KB</span>
        </div>
      `;

      card.addEventListener('click', () => openInspector(item));
      grid.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load explorer samples', err);
    grid.innerHTML = '<div style="color:var(--color-danger); padding:20px;">Failed to fetch sample inventory from server.</div>';
  }
}

// 5. Generative Synthesis Pipeline
function initPipeline() {
  const slider = document.getElementById('cfgScaleSlider');
  const display = document.getElementById('cfgScaleVal');
  if (slider && display) {
    slider.addEventListener('input', (e) => {
      display.textContent = parseFloat(e.target.value).toFixed(1);
    });
  }

  const launchBtn = document.getElementById('btnLaunchJob');
  if (launchBtn) {
    launchBtn.addEventListener('click', launchGenerationJob);
  }
}

async function launchGenerationJob() {
  const arch = document.getElementById('cfgArchitecture').value;
  const targetClass = document.getElementById('cfgTargetClass').value;
  const count = parseInt(document.getElementById('cfgBatchCount').value, 10);
  const resolution = parseInt(document.getElementById('cfgResolution').value, 10);
  const cfgScale = parseFloat(document.getElementById('cfgScaleSlider').value);
  const seed = parseInt(document.getElementById('cfgSeed').value, 10);

  const launchBtn = document.getElementById('btnLaunchJob');
  const badge = document.getElementById('jobStatusBadge');
  const progWrapper = document.getElementById('jobProgressWrapper');
  const progBar = document.getElementById('jobProgressBar');
  const progText = document.getElementById('jobProgressText');
  const durText = document.getElementById('jobDurationText');
  const logBox = document.getElementById('terminalLogBox');

  launchBtn.disabled = true;
  badge.textContent = 'QUEUED';
  badge.className = 'status-badge status-warn';
  progWrapper.style.display = 'block';
  progBar.style.width = '10%';
  progText.textContent = '10%';
  durText.textContent = 'Allocating tensors...';

  logBox.innerHTML = `
    <div class="log-line">[DISPATCH] Queuing synthesis job for target_class=${targetClass} using ${arch}...</div>
  `;

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

    if (!createRes.ok) {
      const err = await createRes.json();
      throw new Error(err.detail || 'Job creation failed');
    }

    const { job_id } = await createRes.json();
    state.activeJobId = job_id;

    badge.textContent = 'SYNTHESIZING';
    progBar.style.width = '40%';
    progText.textContent = '40%';
    logBox.innerHTML += `<div class="log-line">[SYSTEM] Job ${job_id} assigned to PyTorch GPU/MPS acceleration context.</div>`;

    // 2. Run job
    const runRes = await fetch(`/api/jobs/${job_id}/run`, { method: 'POST' });
    const jobResult = await runRes.json();

    // 3. Render telemetry
    badge.textContent = 'COMPLETED';
    badge.className = 'status-badge status-ok';
    progBar.style.width = '100%';
    progText.textContent = '100%';
    durText.textContent = `Completed in ${jobResult.duration_ms}ms (${Math.round(jobResult.duration_ms / count)}ms/sample)`;

    logBox.innerHTML = '';
    jobResult.logs.forEach(l => {
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = l;
      logBox.appendChild(line);
    });
    logBox.scrollTop = logBox.scrollHeight;

    // Render output grid
    renderOutputArtifacts(jobResult.samples);

    // Refresh datasets & inventory
    await loadInventory();
    loadAuditData();
  } catch (err) {
    badge.textContent = 'FAILED';
    badge.className = 'status-badge status-danger';
    logBox.innerHTML += `<div class="log-line" style="color:var(--color-danger);">[ERROR] ${err.message}</div>`;
  } finally {
    launchBtn.disabled = false;
  }
}

function renderOutputArtifacts(samples) {
  const container = document.getElementById('outputPreviewGrid');
  if (!container) return;
  container.innerHTML = '';

  if (!samples || samples.length === 0) {
    container.innerHTML = '<div class="placeholder-box">No artifacts generated.</div>';
    return;
  }

  samples.forEach(s => {
    const card = document.createElement('div');
    card.className = 'image-card';
    card.innerHTML = `
      <div class="image-thumb-wrap">
        <img src="${s.url}" alt="${s.filename}" />
        <span class="card-badge badge-synth">${s.architecture.toUpperCase()}</span>
      </div>
      <div class="card-meta">
        <span class="card-class-tag">${s.class}</span>
        <span class="card-file-dim">${s.resolution}</span>
      </div>
    `;
    card.addEventListener('click', () => openInspector({
      url: s.url,
      filename: s.filename,
      class: s.class,
      category: 'synthetic',
      size_bytes: 42000,
      metadata: s.metadata
    }));
    container.appendChild(card);
  });
}

// 6. Quality & Privacy Audit
function initAudit() {
  const runBtn = document.getElementById('btnRunAudit');
  if (runBtn) {
    runBtn.addEventListener('click', loadAuditData);
  }
}

async function loadAuditData() {
  try {
    const res = await fetch('/api/quality/audit');
    const data = await res.json();

    const m = data.metrics || {};
    
    // Fréchet Inception Distance
    if (m.fid) {
      document.getElementById('valFid').textContent = m.fid.value;
      const tag = document.getElementById('tagFid');
      tag.textContent = m.fid.status;
      tag.className = `kpi-tag ${m.fid.status === 'OPTIMAL' ? 'status-ok' : 'status-warn'}`;
    }

    // Inception Score
    if (m.inception_score) {
      document.getElementById('valIS').textContent = m.inception_score.value;
      const tag = document.getElementById('tagIS');
      tag.textContent = m.inception_score.status;
      tag.className = `kpi-tag ${m.inception_score.status === 'OPTIMAL' ? 'status-ok' : 'status-warn'}`;
    }

    // Diversity Index
    if (m.diversity_index) {
      document.getElementById('valDiv').textContent = m.diversity_index.value;
      const tag = document.getElementById('tagDiv');
      tag.textContent = m.diversity_index.status;
      tag.className = `kpi-tag ${m.diversity_index.status === 'OPTIMAL' ? 'status-ok' : 'status-warn'}`;
    }

    // Privacy retention
    if (m.privacy_retention) {
      document.getElementById('valPriv').textContent = m.privacy_retention.value;
      const tag = document.getElementById('tagPriv');
      tag.textContent = m.privacy_retention.status;
      tag.className = 'kpi-tag status-ok';
    }

    // Architecture Table
    const tbody = document.getElementById('archTableBody');
    if (tbody && data.architecture_benchmarks) {
      tbody.innerHTML = '';
      data.architecture_benchmarks.forEach(a => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${a.architecture}</strong></td>
          <td><span class="metric-mono">${a.fid}</span></td>
          <td><span class="metric-mono">${a.diversity}</span></td>
          <td><span class="metric-mono">${a.throughput_fps} fps</span></td>
          <td><span class="metric-mono">${a.memory_mb} MB</span></td>
          <td><span class="status-badge status-ok">VERIFIED</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Memorization Audit Pairs
    const memGrid = document.getElementById('memorizationGrid');
    if (memGrid && data.memorization_audit) {
      memGrid.innerHTML = '';
      if (data.memorization_audit.length === 0) {
        memGrid.innerHTML = '<div style="color:var(--text-muted); padding:16px;">Generate synthetic samples to perform nearest-neighbor memorization audit.</div>';
      } else {
        data.memorization_audit.forEach((pair, idx) => {
          const item = document.createElement('div');
          item.className = 'mem-pair-card';
          item.innerHTML = `
            <div class="mem-pair-header">
              <span>Audit Sample #${idx + 1}</span>
              <span class="status-badge status-ok">${pair.memorization_status}</span>
            </div>
            <div class="mem-images-split">
              <div class="mem-img-box">
                <img src="${pair.synthetic_image}" alt="Synthetic Sample" />
                <span class="mem-img-tag">Synthesized</span>
              </div>
              <div class="mem-distance-indicator">
                <span class="mem-dist-val">&Delta; L2 = ${pair.euclidean_distance}</span>
                <span class="mem-dist-sub">Euclidean Dist</span>
              </div>
              <div class="mem-img-box">
                <img src="${pair.closest_real_image}" alt="Nearest Real Sample" />
                <span class="mem-img-tag">Closest Real</span>
              </div>
            </div>
          `;
          memGrid.appendChild(item);
        });
      }
    }
  } catch (err) {
    console.error('Failed to load audit data', err);
  }
}

// 7. Classifier Benchmark & Research Hypothesis Engine
function initBenchmark() {
  const trainBtn = document.getElementById('btnTrainBenchmark');
  if (trainBtn) {
    trainBtn.addEventListener('click', runClassifierBenchmark);
  }
}

async function runClassifierBenchmark() {
  const trainBtn = document.getElementById('btnTrainBenchmark');
  trainBtn.disabled = true;
  trainBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M12 2a10 10 0 0 1 10 10"></path>
    </svg>
    Training PyTorch CNN Models (MPS)...
  `;

  try {
    const res = await fetch('/api/benchmark/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        backbone: 'DefectConvNet-V2',
        epochs: 10,
        learning_rate: 0.002
      })
    });

    if (!res.ok) {
      throw new Error('Benchmark training failed');
    }

    const data = await res.json();
    state.lastBenchmarkData = data;
    renderBenchmarkResults(data);
  } catch (err) {
    console.error('Failed to run benchmark', err);
    alert('Benchmark training failed: ' + err.message);
  } finally {
    trainBtn.disabled = false;
    trainBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3"/>
      </svg>
      Train & Evaluate PyTorch CNN Models
    `;
  }
}

function renderBenchmarkResults(data) {
  const s = data.summary_comparison;
  const resMeta = data.research_question_result || {};

  // Research Outcome Banner & Card
  const resultCard = document.getElementById('researchResultCard');
  if (resultCard) {
    resultCard.style.display = 'block';
    const headingEl = document.getElementById('resultHeadingText');
    const explEl = document.getElementById('resultExplanationText');
    const badgeEl = document.getElementById('hypothesisBadge');

    if (resMeta.hypothesis_proven) {
      headingEl.textContent = `Hypothesis Confirmed: Augmenting with Synthetic ${resMeta.rare_class_name} Boosts Real Unseen Recall by ${resMeta.rare_recall_delta}`;
      badgeEl.textContent = 'PROVEN • STATISTICALLY SIGNIFICANT';
      badgeEl.className = 'status-badge status-ok';
    } else {
      headingEl.textContent = `Ablation Result: Augmenting with Synthetic ${resMeta.rare_class_name} Retained Baseline Parity`;
      badgeEl.textContent = 'EVALUATED • NEUTRAL DELTA';
      badgeEl.className = 'status-badge status-warn';
    }
    explEl.textContent = resMeta.scientific_finding;
  }

  // Top Global Header Hypothesis Status
  const topBadge = document.getElementById('hypothesisStatusText');
  if (topBadge) {
    if (resMeta.hypothesis_proven) {
      topBadge.textContent = `Hypothesis Proven: ${resMeta.rare_recall_delta} Unseen Rare Recall Jump`;
      topBadge.classList.add('proven');
    } else {
      topBadge.textContent = 'Baseline Parity Maintained';
    }
  }

  // Delta Strip
  document.getElementById('baseAccVal').textContent = `${s.accuracy.baseline}%`;
  document.getElementById('augAccVal').textContent = `${s.accuracy.augmented}%`;
  document.getElementById('gainAcc').textContent = s.accuracy.delta;

  document.getElementById('baseF1Val').textContent = `${s.macro_f1.baseline}%`;
  document.getElementById('augF1Val').textContent = `${s.macro_f1.augmented}%`;
  document.getElementById('gainF1').textContent = s.macro_f1.delta;

  const rareClass = data.rare_class || 'Minority Class';
  const deltaTitle = document.getElementById('deltaRareTitle');
  if (deltaTitle) {
    deltaTitle.textContent = `Minority Class (${rareClass}) Recall on Unseen Real Test`;
  }

  document.getElementById('basePotholeVal').textContent = `${s.rare_defect_recall.baseline}%`;
  document.getElementById('augPotholeVal').textContent = `${s.rare_defect_recall.augmented}%`;
  document.getElementById('gainPothole').textContent = s.rare_defect_recall.delta;

  // Update Dataset Sizes Badges
  const baseSize = data.dataset_sizes.baseline_train;
  const augSize = data.dataset_sizes.synthetic_augmented;
  const valSize = data.dataset_sizes.validation_set;

  document.getElementById('badgeReportBaseline').textContent = `Real Only N=${baseSize} (Val N=${valSize})`;
  document.getElementById('badgeReportAugmented').textContent = `Real+Synth N=${augSize} (Val N=${valSize})`;

  // Classification Reports Tables
  renderReportTable('tableReportBaseline', data.classification_reports.baseline);
  renderReportTable('tableReportAugmented', data.classification_reports.augmented);

  // Confusion Matrices
  renderConfusionMatrix('matrixBaseline', data.confusion_matrices.baseline);
  renderConfusionMatrix('matrixAugmented', data.confusion_matrices.augmented);

  // Render Charts
  renderBenchmarkCharts();
}

function renderReportTable(tableId, report) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  report.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${row.class}</strong></td>
      <td><span class="metric-mono">${row.precision.toFixed(2)}</span></td>
      <td><span class="metric-mono">${row.recall.toFixed(2)}</span></td>
      <td><span class="metric-mono">${row.f1_score.toFixed(2)}</span></td>
      <td><span class="metric-mono">${row.support}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderConfusionMatrix(containerId, cmData) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const labels = cmData.labels;
  const matrix = cmData.matrix;
  const n = labels.length;

  container.style.display = 'grid';
  container.style.gridTemplateColumns = `repeat(${n + 1}, auto)`;
  container.style.gap = '4px';

  // Corner blank
  const corner = document.createElement('div');
  corner.className = 'cm-cell cm-header';
  corner.textContent = 'Actual \\ Pred';
  container.appendChild(corner);

  // Header row
  labels.forEach(l => {
    const h = document.createElement('div');
    h.className = 'cm-cell cm-header';
    h.textContent = l;
    container.appendChild(h);
  });

  // Matrix rows
  for (let r = 0; r < n; r++) {
    const rowLabel = document.createElement('div');
    rowLabel.className = 'cm-cell cm-header';
    rowLabel.textContent = labels[r];
    container.appendChild(rowLabel);

    for (let c = 0; c < n; c++) {
      const cell = document.createElement('div');
      const val = matrix[r][c];
      const isDiag = (r === c);
      cell.className = `cm-cell ${isDiag ? 'cm-diag' : (val > 0 ? 'cm-off' : '')}`;
      cell.textContent = val;
      container.appendChild(cell);
    }
  }
}

function renderBenchmarkCharts() {
  if (!state.lastBenchmarkData) return;
  const curves = state.lastBenchmarkData.training_curves;

  // 1. Loss Chart
  const ctxLoss = document.getElementById('chartLossConvergence');
  if (ctxLoss) {
    if (state.lossChart) state.lossChart.destroy();
    state.lossChart = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: curves.epochs.map(e => `Ep ${e}`),
        datasets: [
          {
            label: 'Baseline (Real Only)',
            data: curves.loss_baseline,
            borderColor: '#F85149',
            backgroundColor: 'rgba(248, 81, 73, 0.1)',
            tension: 0.3,
            borderWidth: 2
          },
          {
            label: 'Augmented (Real + Synthetic)',
            data: curves.loss_augmented,
            borderColor: '#2EA043',
            backgroundColor: 'rgba(46, 160, 67, 0.1)',
            tension: 0.3,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8B949E', font: { family: '-apple-system', size: 11 } } }
        },
        scales: {
          x: { grid: { color: '#21262D' }, ticks: { color: '#8B949E' } },
          y: { grid: { color: '#21262D' }, ticks: { color: '#8B949E' } }
        }
      }
    });
  }

  // 2. Val Acc Chart
  const ctxAcc = document.getElementById('chartValAcc');
  if (ctxAcc) {
    if (state.valAccChart) state.valAccChart.destroy();
    state.valAccChart = new Chart(ctxAcc, {
      type: 'line',
      data: {
        labels: curves.epochs.map(e => `Ep ${e}`),
        datasets: [
          {
            label: 'Baseline Validation Acc (%)',
            data: curves.val_acc_baseline,
            borderColor: '#D29922',
            tension: 0.3,
            borderWidth: 2
          },
          {
            label: 'Augmented Validation Acc (%)',
            data: curves.val_acc_augmented,
            borderColor: '#388BFD',
            tension: 0.3,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8B949E', font: { family: '-apple-system', size: 11 } } }
        },
        scales: {
          x: { grid: { color: '#21262D' }, ticks: { color: '#8B949E' } },
          y: {
            min: 0,
            max: 100,
            grid: { color: '#21262D' },
            ticks: { color: '#8B949E', callback: v => `${v}%` }
          }
        }
      }
    });
  }
}

// 8. Upload Modal & Dataset Ingestion
function initUpload() {
  const modal = document.getElementById('uploadModal');
  const openBtn = document.getElementById('btnUploadModal');
  const closeBtn = document.getElementById('btnCloseUpload');

  if (openBtn) openBtn.addEventListener('click', () => modal.classList.add('active'));
  if (closeBtn) closeBtn.addEventListener('click', () => modal.classList.remove('active'));

  // Switch tabs in upload modal
  const tabZip = document.getElementById('tabUploadZip');
  const tabSingle = document.getElementById('tabUploadSingle');
  const secZip = document.getElementById('sectionUploadZip');
  const secSingle = document.getElementById('sectionUploadSingle');

  if (tabZip && tabSingle) {
    tabZip.addEventListener('click', () => {
      tabZip.classList.add('active');
      tabSingle.classList.remove('active');
      secZip.style.display = 'block';
      secSingle.style.display = 'none';
    });

    tabSingle.addEventListener('click', () => {
      tabSingle.classList.add('active');
      tabZip.classList.remove('active');
      secSingle.style.display = 'block';
      secZip.style.display = 'none';
    });
  }

  // ZIP Upload
  const zipDrop = document.getElementById('zipDropzone');
  const zipInput = document.getElementById('zipFileInput');
  if (zipDrop && zipInput) {
    zipDrop.addEventListener('click', () => zipInput.click());
    zipInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleZipUpload(e.target.files[0]);
    });
  }

  // Single Image Upload
  const singleDrop = document.getElementById('singleDropzone');
  const fileInput = document.getElementById('fileInput');
  if (singleDrop && fileInput) {
    singleDrop.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleSingleUpload(e.target.files[0]);
    });
  }
}

async function handleZipUpload(file) {
  const feedback = document.getElementById('uploadFeedback');
  feedback.style.display = 'block';
  feedback.style.color = '#388BFD';
  feedback.textContent = `Uploading and extracting "${file.name}"...`;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/api/datasets/upload-zip', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const result = await res.json();

    feedback.style.color = '#2EA043';
    feedback.textContent = `Dataset "${file.name}" successfully extracted and activated with ${result.dataset.classes.length} classes!`;

    // Reload available datasets & inventory
    await loadAvailableDatasets();
    setTimeout(() => {
      document.getElementById('uploadModal').classList.remove('active');
      feedback.style.display = 'none';
    }, 1500);
  } catch (err) {
    feedback.style.color = '#F85149';
    feedback.textContent = `Upload error: ${err.message}`;
  }
}

async function handleSingleUpload(file) {
  const targetClass = document.getElementById('uploadClassSelect').value;
  const feedback = document.getElementById('uploadFeedback');
  feedback.style.display = 'block';
  feedback.style.color = '#388BFD';
  feedback.textContent = `Uploading image to class '${targetClass}'...`;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('target_class', targetClass);

  try {
    const res = await fetch('/api/dataset/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Upload failed');

    feedback.style.color = '#2EA043';
    feedback.textContent = `Uploaded sample to ${targetClass}!`;

    await loadInventory();
    loadExplorerSamples();
    setTimeout(() => {
      document.getElementById('uploadModal').classList.remove('active');
      feedback.style.display = 'none';
    }, 1200);
  } catch (err) {
    feedback.style.color = '#F85149';
    feedback.textContent = `Error: ${err.message}`;
  }
}

// 9. Export Dataset Archive
function initExport() {
  const btn = document.getElementById('btnHeaderExport');
  if (btn) {
    btn.addEventListener('click', () => {
      window.location.href = '/api/export/archive?format=raw';
    });
  }
}

// 10. Image Detail Inspector
function initInspector() {
  const modal = document.getElementById('inspectorModal');
  const closeBtn = document.getElementById('btnCloseInspector');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => modal.classList.remove('active'));
  }
}

function openInspector(item) {
  const modal = document.getElementById('inspectorModal');
  document.getElementById('inspectorTitle').textContent = `Sample: ${item.filename}`;
  document.getElementById('inspectorImage').src = item.url;

  const metaEl = document.getElementById('inspectorMeta');
  let metaHtml = `
    <div><strong>Dataset:</strong> ${state.inventory ? state.inventory.dataset_id : 'active'}</div>
    <div><strong>Class:</strong> ${item.class}</div>
    <div><strong>Category:</strong> ${item.category.toUpperCase()}</div>
    <div><strong>Resolution:</strong> 256 x 256 RGB</div>
    <div><strong>File Size:</strong> ${Math.round(item.size_bytes / 1024)} KB</div>
    <div><strong>Storage Path:</strong> ${item.url}</div>
  `;

  if (item.metadata) {
    metaHtml += `<div style="margin-top:8px; border-top:1px solid var(--border-default); padding-top:6px;"><strong>Generative Parameters:</strong></div>`;
    for (const [k, v] of Object.entries(item.metadata)) {
      metaHtml += `<div>&bull; ${k}: <span style="color:#58A6FF;">${typeof v === 'object' ? JSON.stringify(v) : v}</span></div>`;
    }
  }

  metaEl.innerHTML = metaHtml;
  modal.classList.add('active');
}
