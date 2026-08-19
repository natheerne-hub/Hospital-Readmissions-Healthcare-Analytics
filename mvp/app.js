const views = document.querySelectorAll('.view');
const navItems = document.querySelectorAll('.nav-item');

navItems.forEach((button) => {
  button.addEventListener('click', () => {
    navItems.forEach((item) => item.classList.remove('active'));
    views.forEach((view) => view.classList.remove('active-view'));
    button.classList.add('active');
    document.getElementById(button.dataset.view).classList.add('active-view');
  });
});

function fmt(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function renderConditionEvidence(data) {
  const container = document.getElementById('conditionBars');
  container.innerHTML = '';

  if (Array.isArray(data.conditions) && data.conditions.length) {
    const validRates = data.conditions
      .map((row) => Number(row.mean_predicted_rate))
      .filter((value) => Number.isFinite(value));
    const maxRate = Math.max(...validRates, 1);

    data.conditions.forEach((row) => {
      const rate = Number(row.mean_predicted_rate);
      const width = Number.isFinite(rate) ? Math.max(4, (rate / maxRate) * 100) : 0;
      const line = document.createElement('div');
      line.className = 'bar-row';
      line.innerHTML = `<span>${row.condition}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong>${Number.isFinite(rate) ? `${fmt(rate, 2)}%` : '—'}</strong>`;
      container.appendChild(line);
    });
    return;
  }

  (data.condition_findings || []).forEach((finding) => {
    const line = document.createElement('div');
    line.className = 'evidence-line';
    line.textContent = finding;
    container.appendChild(line);
  });
}

function renderHrrpData(data) {
  const kpiCards = document.querySelectorAll('#overview .kpi-grid .kpi strong');
  const k = data.kpis || {};
  if (kpiCards.length >= 4) {
    kpiCards[0].textContent = fmt(k.hospitals);
    kpiCards[1].textContent = fmt(k.records);
    kpiCards[2].textContent = fmt(k.valid_err_records);
    kpiCards[3].textContent = `${fmt(k.err_above_1_pct, 1)}%`;
  }

  const signalCards = document.querySelectorAll('#signals .kpi-grid .kpi strong');
  if (signalCards.length >= 3) {
    signalCards[0].textContent = fmt(k.persistent_high_err_hospitals);
    signalCards[1].textContent = fmt(k.persistent_low_err_hospitals);
    signalCards[2].textContent = fmt(k.persistent_signal_min_conditions);
  }

  const qualityMetrics = document.querySelectorAll('#quality .metric');
  if (qualityMetrics.length >= 3) {
    qualityMetrics[0].textContent = fmt(k.duplicate_rows);
    qualityMetrics[2].textContent = `${fmt(k.valid_err_records)} / ${fmt(k.records)}`;
  }

  renderConditionEvidence(data);
}

async function loadHrrpData() {
  try {
    const response = await fetch('data/hrrp_summary.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderHrrpData(data);
  } catch (error) {
    const container = document.getElementById('conditionBars');
    container.innerHTML = '<div class="warning"><strong>Data asset unavailable.</strong> Run <code>python mvp/build_mvp_data.py</code> from the repository root to regenerate the traceable HRRP summary.</div>';
    console.error('Unable to load HRRP MVP data:', error);
  }
}

const hospitals = [
  { name: 'Demo Medical Center A', err: 1.087, conditions: 6 },
  { name: 'Demo Regional Hospital B', err: 0.968, conditions: 5 },
  { name: 'Demo Community Hospital C', err: 1.012, conditions: 6 },
];

const select = document.getElementById('hospitalSelect');
const meanErr = document.getElementById('meanErr');
const conditionCount = document.getElementById('conditionCount');
const result = document.getElementById('signalResult');

hospitals.forEach((hospital, index) => {
  const option = document.createElement('option');
  option.value = index;
  option.textContent = hospital.name;
  select.appendChild(option);
});

function renderHospital() {
  const hospital = hospitals[Number(select.value)];
  meanErr.value = hospital.err.toFixed(3);
  conditionCount.value = hospital.conditions;
  let label = 'Near expected';
  let explanation = 'Mean ERR is close to 1.0. Treat this as a screening signal, not a quality verdict.';

  if (hospital.err > 1.05) {
    label = 'Elevated readmission signal';
    explanation = 'Mean ERR is above 1.05 in this synthetic demo record. A real review would drill into condition coverage, reporting completeness, and operational context.';
  } else if (hospital.err < 0.98) {
    label = 'Lower readmission signal';
    explanation = 'Mean ERR is below 0.98 in this synthetic demo record. This still does not establish superior overall hospital quality.';
  }
  result.innerHTML = `<strong>${label}</strong><p>${explanation}</p>`;
}

select.addEventListener('change', renderHospital);
renderHospital();

let modelRegistry = null;

async function loadModelRegistry() {
  const response = await fetch('../modeling/model_registry.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Model registry HTTP ${response.status}`);
  modelRegistry = await response.json();
  return modelRegistry;
}

function unitMatches(model, unit) {
  if (model.unit_of_analysis === unit) return true;
  if (model.id === 'dubai-nabidh-external-validation' && unit === 'authorized Dubai sandbox data') return true;
  return false;
}

function evaluateRoute() {
  const task = document.getElementById('taskSelect').value;
  const unit = document.getElementById('unitSelect').value;
  const status = document.getElementById('routeStatus');
  const output = document.getElementById('routeResult');

  if (!modelRegistry) {
    status.value = 'REGISTRY UNAVAILABLE';
    output.innerHTML = '<strong>Routing blocked</strong><p>The model registry has not loaded, so the platform will not guess a model.</p>';
    return;
  }

  const candidates = modelRegistry.models.filter((model) => model.task === task);
  const model = candidates.find((candidate) => unitMatches(candidate, unit));

  if (!model) {
    status.value = 'NO VALID ROUTE';
    output.innerHTML = `<strong>No compatible model</strong><p>No registered model matches both the requested task and the selected data unit. The platform will not mix datasets or substitute a different model to manufacture a result.</p>`;
    return;
  }

  if (task === 'hospital_level_readmission_intelligence') {
    status.value = 'ANALYTICS READY';
    output.innerHTML = `<strong>Route: ${model.id}</strong><p>Use ${model.data_source} for hospital-level KPIs and performance signals. Patient probability is not permitted from this evidence layer.</p>`;
    return;
  }

  if (model.patient_probability_allowed === true) {
    status.value = 'PROBABILITY ENABLED';
    output.innerHTML = `<strong>Route: ${model.id}</strong><p>This registered model is explicitly unlocked for probability output. The response must still include model version, population, validation status, and limitations.</p>`;
    return;
  }

  status.value = 'OUTPUT LOCKED';
  const reason = model.unlock_rule || 'The registered model is not currently approved for probability output.';
  output.innerHTML = `<strong>Route identified: ${model.id}</strong><p>${reason}</p><p><b>Current status:</b> ${model.status}. No patient risk percentage will be shown.</p>`;
}

document.getElementById('routeButton').addEventListener('click', evaluateRoute);

loadModelRegistry()
  .then(() => evaluateRoute())
  .catch((error) => {
    console.error('Unable to load model registry:', error);
    evaluateRoute();
  });

loadHrrpData();
