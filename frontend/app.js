/* Cricket Match Intelligence — Frontend v3 */
const API = '';

let currentMatchId = null;
let currentInnings = 1;
let inningsAvailable = [1];
let allInningsData = {};   // innings_number -> parse response
let allAnalyticsData = {}; // innings_number -> analytics response
let momentumChart = null;
let phaseChart = null;

// ── Tab navigation ──────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => { t.classList.add('hidden'); t.classList.remove('active'); });
    btn.classList.add('active');
    const tab = document.getElementById('tab-' + btn.dataset.tab);
    tab.classList.remove('hidden');
    tab.classList.add('active');
  });
});

// ── Sample commentary ───────────────────────────────────
const SAMPLE = `0.1: Cummins to Rohit, no run. Tight line outside off, Rohit defends solidly.
0.2: Cummins to Rohit, 1 run. Pushed to mid-on, easy single.
0.3: Cummins to Kohli, four! Driven beautifully through covers, races to the fence.
0.4: Cummins to Kohli, no run. Dot ball, played and missed outside off.
0.5: Cummins to Kohli, 1 run. Nudged to fine leg.
0.6: Cummins to Rohit, six! Pulled over mid-wicket, launched into the stands!
1.1: Hazlewood to Kohli, no run. Defended back down the pitch.
1.2: Hazlewood to Kohli, four! Cut hard through point, boundary!
1.3: Hazlewood to Kohli, 1 run. Pushed to cover, quick single.
1.4: Hazlewood to Rohit, out! Rohit dismissed, caught at slip by Smith. Caught.
1.5: Hazlewood to Gill, no run. Dot ball, beaten outside off.
1.6: Hazlewood to Gill, 2 runs. Driven to long-off, 2 runs.
2.1: Cummins to Kohli, six! Slog swept over deep mid-wicket, maximum!
2.2: Cummins to Kohli, four! Driven through extra cover, boundary!
2.3: Cummins to Kohli, 1 run. Kohli reaches his fifty! Half-century for the master.
2.4: Cummins to Gill, no run. Defended.
2.5: Cummins to Gill, out! Gill dismissed, bowled by Cummins. Stumps shattered.
2.6: Cummins to Pant, 1 run. Flicked to fine leg.
3.1: Hazlewood to Kohli, wide. Down the leg side, wide called.
3.2: Hazlewood to Kohli, no run. Dot ball.
3.3: Hazlewood to Kohli, four! Pulled through mid-wicket, boundary!
3.4: Hazlewood to Kohli, out! Kohli dismissed, caught by Warner at slip. Caught.
3.5: Hazlewood to Iyer, no run. Defended.
3.6: Hazlewood to Iyer, 1 run. Pushed to mid-on.
4.1: Starc to Pant, six! Pant smashed over long-on, into the stands!
4.2: Starc to Pant, four! Driven through covers, four!
4.3: Starc to Pant, no run. Beaten outside off.
4.4: Starc to Pant, 1 run. Nudged to fine leg.
4.5: Starc to Iyer, no run. Dot ball, defended.
4.6: Starc to Iyer, 2 runs. Driven to long-off, 2 runs.
5.1: Cummins to Pant, out! Pant dismissed, stumped by Carey. Stumped.
5.2: Cummins to Jadeja, no run. Defended.
5.3: Cummins to Jadeja, 1 run. Pushed to mid-on.
5.4: Cummins to Iyer, no run. Dot ball.
5.5: Cummins to Iyer, four! Cut through point, boundary!
5.6: Cummins to Iyer, 1 run. Pushed to cover.`;

document.getElementById('load-sample').addEventListener('click', () => {
  document.getElementById('commentary-input').value = SAMPLE;
  document.getElementById('team1').value = 'India';
  document.getElementById('team2').value = 'Australia';
  document.getElementById('innings').value = '1';
  uploadedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-info').classList.add('hidden');
  document.getElementById('commentary-input').placeholder = 'Paste commentary here...';
});

// ── File upload ─────────────────────────────────────────
let uploadedFile = null;

document.getElementById('file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  uploadedFile = file;
  const info = document.getElementById('file-info');
  info.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB) — ready to parse`;
  info.classList.remove('hidden');
  document.getElementById('commentary-input').value = '';
  document.getElementById('commentary-input').placeholder = `File loaded: ${file.name}. Click Parse to process it.`;
});

// ── Parse ───────────────────────────────────────────────
document.getElementById('parse-btn').addEventListener('click', async () => {
  const commentary = document.getElementById('commentary-input').value.trim();
  const btn = document.getElementById('parse-btn');
  const status = document.getElementById('parse-status');

  if (!commentary && !uploadedFile) {
    showToast('Paste commentary or upload a file first', 'error');
    return;
  }

  btn.disabled = true;
  btn.textContent = '⏳ Parsing...';
  setStatus(status, 'loading', '⏳ Parsing commentary...');

  try {
    let data;
    const inningsNum = parseInt(document.getElementById('innings').value) || 1;

    if (uploadedFile) {
      const form = new FormData();
      form.append('file', uploadedFile);
      form.append('team1', document.getElementById('team1').value || 'Team A');
      form.append('team2', document.getElementById('team2').value || 'Team B');
      form.append('innings', String(inningsNum));
      const target = document.getElementById('target').value;
      if (target) form.append('target', target);
      const res = await fetch(`${API}/api/parse/upload`, { method: 'POST', body: form });
      const text = await res.text();
      try { data = JSON.parse(text); } catch { throw new Error(text.substring(0, 300)); }
      if (!res.ok) throw new Error(data.detail || 'Parse failed');
    } else {
      const res = await fetch(`${API}/api/parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          commentary,
          team1: document.getElementById('team1').value || 'Team A',
          team2: document.getElementById('team2').value || 'Team B',
          innings: inningsNum,
          target: parseInt(document.getElementById('target').value) || null,
        }),
      });
      const text = await res.text();
      try { data = JSON.parse(text); } catch { throw new Error(text.substring(0, 300)); }
      if (!res.ok) throw new Error(data.detail || 'Parse failed');
    }

    // Use same match_id across innings if already set
    if (!currentMatchId) currentMatchId = data.match_id;

    // Store innings data
    const inn = data.innings;
    allInningsData[inn] = data;
    inningsAvailable = data.innings_available || [inn];
    currentInnings = inn;

    // Fetch analytics for this innings
    const aRes = await fetch(`${API}/api/analytics/${currentMatchId}?innings=${inn}`);
    const analytics = await aRes.json();
    allAnalyticsData[inn] = analytics;

    // Reset file state
    uploadedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('commentary-input').placeholder = 'Paste commentary here...';

    setStatus(status, 'success',
      `✅ Innings ${inn}: ${data.events_parsed} deliveries parsed in ${data.parse_time_ms}ms — ID: ${currentMatchId}`);

    document.getElementById('match-id-label').textContent = `Match: ${currentMatchId}`;
    document.getElementById('match-badge').classList.remove('hidden');

    renderInningsSwitcher();
    renderScorecard(data);
    renderAnalytics(analytics);
    showToast(`Innings ${inn}: ${data.events_parsed} balls parsed`, 'success');
    document.querySelector('[data-tab="scorecard"]').click();

  } catch (e) {
    setStatus(status, 'error', `❌ ${e.message}`);
    showToast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Parse Commentary →';
  }
});

function setStatus(el, type, msg) {
  el.className = `status ${type}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ── Innings switcher ────────────────────────────────────
function renderInningsSwitcher() {
  const sw = document.getElementById('innings-switcher');
  if (inningsAvailable.length <= 1) { sw.classList.add('hidden'); return; }
  sw.classList.remove('hidden');
  sw.innerHTML = inningsAvailable.map(i => `
    <button class="innings-btn ${i === currentInnings ? 'active' : ''}" data-innings="${i}">
      Innings ${i}
    </button>`).join('');
  sw.querySelectorAll('.innings-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = parseInt(btn.dataset.innings);
      if (i === currentInnings) return;
      currentInnings = i;
      renderInningsSwitcher();
      if (allInningsData[i]) renderScorecard(allInningsData[i]);
      if (allAnalyticsData[i]) renderAnalytics(allAnalyticsData[i]);
    });
  });
}

// ── Scorecard rendering ─────────────────────────────────
function renderScorecard(data) {
  const state = data.match_state;

  // Header
  const rrr = state.required_run_rate ? ` | RRR: ${state.required_run_rate}` : '';
  const target = state.target ? `<span>Target: ${state.target}</span>` : '';
  document.getElementById('scorecard-header').innerHTML = `
    <div>
      <div class="score-main">${state.total_runs}/${state.wickets}</div>
      <div class="score-meta">
        <span>${state.overs}.${state.balls_in_over} overs</span>
        <span>${state.batting_team} vs ${state.bowling_team}</span>
        ${target}
      </div>
    </div>
    <div>
      <div class="score-rr">RR: ${state.run_rate}${rrr}</div>
      <div class="phase-pills">
        ${Object.entries(state.phase_scores || {}).map(([ph, runs]) =>
          `<div class="phase-pill"><div class="ph-label">${ph}</div><div class="ph-score">${runs}/${state.phase_wickets?.[ph] || 0}</div></div>`
        ).join('')}
      </div>
    </div>`;

  // Batting — sort by balls faced desc so active batsmen show first
  const batters = Object.values(state.batting_stats || {})
    .sort((a, b) => b.balls_faced - a.balls_faced);
  const batTbody = document.querySelector('#batting-table tbody');
  batTbody.innerHTML = '';
  if (!batters.length) {
    batTbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted)">No batting data</td></tr>';
  }
  batters.forEach(p => {
    const tr = document.createElement('tr');
    const badge = p.is_out
      ? `<span class="out-badge">OUT</span>`
      : `<span class="not-out-badge">*</span>`;
    tr.innerHTML = `
      <td><strong>${p.player}</strong></td>
      <td><strong>${p.runs}</strong></td>
      <td>${p.balls_faced}</td>
      <td>${p.fours}</td>
      <td>${p.sixes}</td>
      <td>${p.strike_rate}</td>
      <td>${badge}</td>`;
    batTbody.appendChild(tr);
  });

  // Bowling — sort by balls bowled desc
  const bowlers = Object.values(state.bowling_stats || {})
    .sort((a, b) => b.balls_bowled - a.balls_bowled);
  const bowlTbody = document.querySelector('#bowling-table tbody');
  bowlTbody.innerHTML = '';
  if (!bowlers.length) {
    bowlTbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted)">No bowling data</td></tr>';
  }
  bowlers.forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${p.player}</strong></td>
      <td>${p.overs_bowled}</td>
      <td>${p.runs_conceded}</td>
      <td><strong>${p.wickets}</strong></td>
      <td>${p.economy}</td>
      <td>${p.dot_balls}</td>`;
    bowlTbody.appendChild(tr);
  });

  // Partnerships — only show valid ones (no "Unknown" placeholders)
  const pList = document.getElementById('partnerships-list');
  pList.innerHTML = '';
  const validPartnerships = (state.partnerships || []).filter(
    p => p.batsman1 !== 'Unknown' && p.batsman2 !== 'Unknown' && p.balls > 0
  );
  if (!validPartnerships.length) {
    pList.innerHTML = '<span style="color:var(--muted)">No completed partnerships yet</span>';
  } else {
    validPartnerships.forEach(p => {
      pList.innerHTML += `
        <div class="partnership-item">
          <span class="p-wkt">Wkt ${p.wicket_number}</span>
          <span class="p-names">${p.batsman1} & ${p.batsman2}</span>
          <span class="p-runs">${p.runs}</span>
          <span class="p-balls">(${p.balls}b)</span>
        </div>`;
    });
  }

  // Current partnership
  const cp = state.current_partnership;
  if (cp && cp.batsman1 !== 'Unknown' && cp.batsman2 !== 'Unknown') {
    pList.innerHTML += `
      <div class="partnership-item" style="border-color:var(--accent)">
        <span class="p-wkt" style="color:var(--accent)">Current</span>
        <span class="p-names">${cp.batsman1} & ${cp.batsman2}</span>
        <span class="p-runs">${cp.runs}</span>
        <span class="p-balls">(${cp.balls}b)</span>
      </div>`;
  }

  // Over history
  const ohDiv = document.getElementById('over-history');
  ohDiv.innerHTML = '';
  (state.over_history || []).forEach(ov => {
    const balls = (ov.balls || []).map(b => {
      let cls = '';
      if (b === 'W') cls = 'wicket';
      else if (b === '4') cls = 'four';
      else if (b === '6') cls = 'six';
      else if (b === 'Wd') cls = 'wide';
      else if (b === 'Nb') cls = 'nb';
      return `<div class="ball-chip ${cls}">${b}</div>`;
    }).join('');
    ohDiv.innerHTML += `
      <div class="over-row">
        <span class="over-num">Ov ${ov.over_number + 1}</span>
        <div class="over-balls">${balls}</div>
        <span class="over-runs">${ov.runs}</span>
        <span class="over-bowler">${ov.bowler || ''}</span>
      </div>`;
  });
  if (!state.over_history?.length) {
    ohDiv.innerHTML = '<span style="color:var(--muted)">No overs completed yet</span>';
  }

  // Ball by ball
  const bbbDiv = document.getElementById('ball-by-ball');
  bbbDiv.innerHTML = '';
  (data.events || []).forEach(e => {
    let cls = '';
    if (e.is_wicket) cls = 'wicket';
    else if (e.boundary_type === '6') cls = 'six';
    else if (e.is_boundary) cls = 'boundary';
    const evtLabel = e.event_type.replace(/_/g, ' ');
    const commentary = (e.raw_commentary || '').replace(/"/g, '&quot;');
    bbbDiv.innerHTML += `
      <div class="bbb-item ${cls}" title="${commentary}">
        <div class="bbb-over">${e.over_ball}</div>
        <div class="bbb-event ${e.event_type}">${evtLabel}</div>
        <div style="color:var(--muted);font-size:11px">${e.batsman || ''} ${e.bowler ? '/ ' + e.bowler : ''}</div>
      </div>`;
  });
}

// ── Analytics rendering ─────────────────────────────────
function renderAnalytics(analytics) {
  const timeline = analytics.momentum_timeline || [];
  const labels   = timeline.map(p => p.over_ball);
  const momData  = timeline.map(p => p.momentum);
  const runsData = timeline.map(p => p.cumulative_runs);

  if (momentumChart) momentumChart.destroy();
  const mCtx = document.getElementById('momentum-chart').getContext('2d');
  momentumChart = new Chart(mCtx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Momentum', data: momData, borderColor: '#4f8ef7',
          backgroundColor: 'rgba(79,142,247,0.1)', fill: true, tension: 0.4,
          pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
        { label: 'Cumulative Runs', data: runsData, borderColor: '#22c55e',
          backgroundColor: 'transparent', tension: 0.3, pointRadius: 0,
          borderWidth: 1.5, borderDash: [4,4], yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
      scales: {
        x:  { ticks: { color: '#64748b', maxTicksLimit: 12, font: { size: 10 } }, grid: { color: '#1e2235' } },
        y:  { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1e2235' },
              title: { display: true, text: 'Momentum', color: '#64748b', font: { size: 10 } } },
        y1: { position: 'right', ticks: { color: '#64748b', font: { size: 10 } },
              grid: { drawOnChartArea: false },
              title: { display: true, text: 'Runs', color: '#64748b', font: { size: 10 } } },
      },
    },
  });

  const phaseScores = analytics.phase_scores || {};
  const phaseWkts   = analytics.phase_wickets || {};
  if (phaseChart) phaseChart.destroy();
  const pCtx = document.getElementById('phase-chart').getContext('2d');
  phaseChart = new Chart(pCtx, {
    type: 'bar',
    data: {
      labels: Object.keys(phaseScores),
      datasets: [
        { label: 'Runs', data: Object.values(phaseScores),
          backgroundColor: ['rgba(79,142,247,0.7)','rgba(124,92,252,0.7)','rgba(239,68,68,0.7)'],
          borderRadius: 6 },
        { label: 'Wickets',
          data: Object.keys(phaseScores).map(k => phaseWkts[k] || 0),
          backgroundColor: 'rgba(245,158,11,0.5)', borderRadius: 6, yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
      scales: {
        x:  { ticks: { color: '#64748b' }, grid: { color: '#1e2235' } },
        y:  { ticks: { color: '#64748b' }, grid: { color: '#1e2235' } },
        y1: { position: 'right', ticks: { color: '#64748b' }, grid: { drawOnChartArea: false } },
      },
    },
  });

  const tpList = document.getElementById('turning-points-list');
  tpList.innerHTML = '';
  const tps = analytics.turning_points || [];
  if (!tps.length) {
    tpList.innerHTML = '<p style="color:var(--muted)">No significant turning points detected. Try a longer innings.</p>';
    return;
  }
  tps.forEach(tp => {
    const evidence = (tp.evidence || []).map(e => `<li>${e}</li>`).join('');
    tpList.innerHTML += `
      <div class="tp-item">
        <div class="tp-header">
          <span class="tp-over">Over ${tp.over_ball}</span>
          <span class="tp-title">${tp.title}</span>
          <span class="impact-badge ${tp.impact_level}">${(tp.impact_level||'').toUpperCase()} ${tp.impact_score}%</span>
        </div>
        <p class="tp-desc">${tp.description}</p>
        <ul class="tp-evidence">${evidence}</ul>
      </div>`;
  });
}

// ── Q&A ─────────────────────────────────────────────────
async function askQuestion(question) {
  if (!currentMatchId) { showToast('Parse a match first', 'error'); return; }
  if (!question.trim()) return;

  const history = document.getElementById('qa-history');
  const entry = document.createElement('div');
  entry.className = 'qa-entry';
  entry.innerHTML = `<div class="qa-q">Q: ${question}</div><div class="qa-a">⏳ Thinking...</div>`;
  history.prepend(entry);

  try {
    const res = await fetch(`${API}/api/qa`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ match_id: currentMatchId, question, top_k: 5, innings: currentInnings }),
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { throw new Error(text.substring(0, 200)); }
    if (!res.ok) throw new Error(data.detail || 'Q&A failed');

    const confPct = Math.round((data.confidence || 0) * 100);
    entry.querySelector('.qa-a').textContent = data.answer;
    entry.innerHTML += `<div class="qa-meta">
      <span class="conf-bar" style="width:${confPct}px"></span>
      Confidence: ${confPct}% | ${data.grounded ? '✅ Grounded' : '⚠️ Ungrounded'}
    </div>`;
  } catch (e) {
    entry.querySelector('.qa-a').textContent = `Error: ${e.message}`;
  }
}

document.getElementById('qa-btn').addEventListener('click', () => {
  const q = document.getElementById('qa-input').value.trim();
  if (!q) return;
  askQuestion(q);
  document.getElementById('qa-input').value = '';
});

document.getElementById('qa-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('qa-btn').click();
});

document.querySelectorAll('.suggestion').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelector('[data-tab="qa"]').click();
    askQuestion(btn.dataset.q);
  });
});

// ── Summary ─────────────────────────────────────────────
document.querySelectorAll('.summary-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!currentMatchId) { showToast('Parse a match first', 'error'); return; }
    const type = btn.dataset.type;
    const output = document.getElementById('summary-output');
    output.textContent = '⏳ Generating summary...';
    try {
      const res = await fetch(`${API}/api/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ match_id: currentMatchId, summary_type: type, innings: currentInnings }),
      });
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); } catch { throw new Error(text.substring(0, 200)); }
      if (!res.ok) throw new Error(data.detail || 'Summary failed');
      output.textContent = data.text;
    } catch (e) {
      output.textContent = `Error: ${e.message}`;
    }
  });
});

// ── Toast ────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type}`;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 3500);
}
