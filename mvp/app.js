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

const conditionDemo = [
  ['Heart Failure', 84, 'Highest'],
  ['COPD', 77, 'High'],
  ['Pneumonia', 69, 'Moderate'],
  ['AMI', 63, 'Moderate'],
  ['Hip/Knee', 43, 'Lower'],
  ['CABG', 38, 'Lower'],
];

const bars = document.getElementById('conditionBars');
conditionDemo.forEach(([name, width, label]) => {
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `<span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong>${label}</strong>`;
  bars.appendChild(row);
});

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
    explanation = 'Mean ERR is above 1.05 in this synthetic demo record. A real review would drill into condition-level coverage, confidence, reporting completeness, and operational context.';
  } else if (hospital.err < 0.98) {
    label = 'Lower readmission signal';
    explanation = 'Mean ERR is below 0.98 in this synthetic demo record. This still does not establish superior overall hospital quality.';
  }
  result.innerHTML = `<strong>${label}</strong><p>${explanation}</p>`;
}

select.addEventListener('change', renderHospital);
renderHospital();
