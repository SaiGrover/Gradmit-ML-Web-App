/**
 * GradAdmit — main.js
 * Graduate Admission Prediction — JIIT Noida Capstone
 * Handles: navigation, predictor API calls, analytics charts, code viewer, dataset table
 */

(function () {
  'use strict';

  /* ─────────────────────────────────────────────────
     CONFIG
  ───────────────────────────────────────────────── */
  var API = {
    predict: '/api/predict',
    dataset: '/api/dataset',
    metrics: '/api/metrics',
  };

  var FEATURES = ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA','Research'];
  var FEATURE_LABELS = ['GRE Score','TOEFL Score','Univ. Rating','SOP','LOR','CGPA','Research'];

  /* ─────────────────────────────────────────────────
     NAVIGATION
  ───────────────────────────────────────────────── */
  var chartsReady = false, tableReady = false, codeReady = false;

  function showTab(id) {
    document.querySelectorAll('.section').forEach(function (s) { s.classList.remove('active'); });
    document.querySelectorAll('.nav-tab').forEach(function (t) { t.classList.remove('active'); });
    var section = document.getElementById('section-' + id);
    if (section) section.classList.add('active');
    var tab = document.querySelector('[data-tab="' + id + '"]');
    if (tab) tab.classList.add('active');
    if (id === 'analytics' && !chartsReady) { buildCharts(); chartsReady = true; }
    if (id === 'data'      && !tableReady)  { buildTable();  tableReady  = true; }
    if (id === 'code'      && !codeReady)   { buildCode();   codeReady   = true; }
  }

  document.querySelectorAll('.nav-tab').forEach(function (btn) {
    btn.addEventListener('click', function () { showTab(this.getAttribute('data-tab')); });
  });

  var goPredictor = document.getElementById('btn-go-predictor');
  var goAnalytics = document.getElementById('btn-go-analytics');
  if (goPredictor) goPredictor.addEventListener('click', function () { showTab('predictor'); });
  if (goAnalytics) goAnalytics.addEventListener('click', function () { showTab('analytics'); });

  /* ─────────────────────────────────────────────────
     TOAST NOTIFICATIONS
  ───────────────────────────────────────────────── */
  function toast(msg, type) {
    type = type || 'success';
    var el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(function () {
      el.style.opacity = '0';
      el.style.transition = 'opacity .4s';
      setTimeout(function () { el.remove(); }, 400);
    }, 3000);
  }

  /* ─────────────────────────────────────────────────
     PREDICTOR
  ───────────────────────────────────────────────── */
  var resVal = 1;
  var predictTimeout = null;

  function readSliders() {
    return {
      gre:               parseFloat(document.getElementById('gre').value),
      toefl:             parseFloat(document.getElementById('toefl').value),
      university_rating: parseFloat(document.getElementById('urating').value),
      sop:               parseFloat(document.getElementById('sop').value),
      lor:               parseFloat(document.getElementById('lor').value),
      cgpa:              parseFloat(document.getElementById('cgpa').value),
      research:          resVal,
    };
  }

  function updateBadges() {
    document.getElementById('gre-val').textContent    = document.getElementById('gre').value;
    document.getElementById('toefl-val').textContent  = document.getElementById('toefl').value;
    document.getElementById('urating-val').textContent = document.getElementById('urating').value;
    document.getElementById('sop-val').textContent    = document.getElementById('sop').value;
    document.getElementById('lor-val').textContent    = document.getElementById('lor').value;
    document.getElementById('cgpa-val').textContent   = parseFloat(document.getElementById('cgpa').value).toFixed(2);
  }

  function runPredictor() {
    updateBadges();
    clearTimeout(predictTimeout);
    predictTimeout = setTimeout(callPredictAPI, 120); // debounce
  }

  function callPredictAPI() {
    var payload = readSliders();
    fetch(API.predict, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) { toast('Prediction error: ' + data.error, 'error'); return; }
        renderResult(data);
      })
      .catch(function () { toast('Could not reach server.', 'error'); });
  }

  function renderResult(data) {
    var pct     = data.probability;
    var admitted = data.admitted;

    document.getElementById('prob-display').textContent = pct + '%';
    document.getElementById('prob-display').style.color = admitted ? 'var(--admitted)' : 'var(--rejected)';

    var v = document.getElementById('verdict');
    v.textContent = admitted ? '✓ LIKELY ADMITTED' : '✗ LIKELY REJECTED';
    v.className   = 'result-verdict ' + (admitted ? 'admitted' : 'rejected');

    var bar = document.getElementById('prob-bar');
    bar.style.width      = pct + '%';
    bar.style.background = admitted
      ? 'linear-gradient(90deg, var(--accent), var(--accent2))'
      : 'linear-gradient(90deg, var(--accentr), #fb7185)';

    // Factor bars
    var contribs = data.contributions;
    var maxC = Math.max.apply(null, Object.values(contribs).map(Math.abs));
    var fl = document.getElementById('factor-list');
    fl.innerHTML = '';
    FEATURES.forEach(function (feat, i) {
      var raw = contribs[feat] || 0;
      var w   = maxC > 0 ? Math.max(2, Math.round((Math.abs(raw) / maxC) * 100)) : 2;
      var div = document.createElement('div');
      div.className = 'factor-item';
      div.innerHTML =
        '<div class="factor-lbl">' + FEATURE_LABELS[i] + '</div>' +
        '<div class="factor-bar-bg"><div class="factor-bar-fill" style="width:' + w + '%"></div></div>' +
        '<div class="factor-val">' + raw.toFixed(2) + '</div>';
      fl.appendChild(div);
    });

    // Tips
    document.getElementById('advice-box').textContent =
      '💡 ' + (data.tips && data.tips.length ? data.tips.join(' · ') : 'Keep up the strong profile!');
  }

  // Slider & toggle listeners
  ['gre', 'toefl', 'urating', 'sop', 'lor', 'cgpa'].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('input', runPredictor);
  });

  var resNo  = document.getElementById('res-no');
  var resYes = document.getElementById('res-yes');
  if (resNo) resNo.addEventListener('click', function () {
    resVal = 0;
    resNo.classList.add('active');
    resYes.classList.remove('active');
    runPredictor();
  });
  if (resYes) resYes.addEventListener('click', function () {
    resVal = 1;
    resYes.classList.add('active');
    resNo.classList.remove('active');
    runPredictor();
  });

  // Initial run
  if (document.getElementById('gre')) runPredictor();

  /* ─────────────────────────────────────────────────
     ANALYTICS — CHART.JS DASHBOARD
  ───────────────────────────────────────────────── */
  // Filter chips toggle
  document.querySelectorAll('.filter-chip').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.filter-chip').forEach(function (b) { b.classList.remove('active'); });
      this.classList.add('active');
      var f = this.getAttribute('data-filter');
      document.querySelectorAll('[class*="filter-group-"]').forEach(function (el) { el.style.display = 'none'; });
      document.querySelectorAll('.filter-group-' + f).forEach(function (el) { el.style.display = ''; });
    });
  });

  function buildCharts() {
    // Fetch live metrics from Flask
    fetch(API.metrics)
      .then(function (r) { return r.json(); })
      .then(function (m) { renderCharts(m); })
      .catch(function () {
        // Fallback to hardcoded metrics if server unavailable
        renderCharts(window.__METRICS__ || {});
      });
  }

  function renderCharts(m) {
    var F  = "'Plus Jakarta Sans',sans-serif";
    var MO = "'JetBrains Mono',monospace";
    var tc = '#5e8099';
    var gr = { color: 'rgba(30,52,72,.6)' };
    var ax = {
      x: { grid: gr, ticks: { color: tc, font: { family: MO, size: 10 } } },
      y: { grid: gr, ticks: { color: tc, font: { family: MO, size: 10 } } },
    };
    var leg = { labels: { color: tc, font: { family: F, size: 11 } } };

    var roc  = m.roc  || { fpr: [0,0,.5,1], tpr: [0,.9,.95,1] };
    var cm   = m.confusion_matrix || [[67,4],[7,47]];
    var coef = m.coefficients || {};
    var gs   = m.group_stats || {};

    // ROC Curve
    var rocEl = document.getElementById('roc-chart');
    if (rocEl) new Chart(rocEl, {
      type: 'line',
      data: {
        labels: roc.fpr,
        datasets: [
          { label: 'ROC Curve (AUC=' + (m.auc || 0.975) + ')', data: roc.tpr,
            borderColor: 'var(--accent)', backgroundColor: 'rgba(0,212,170,.08)',
            fill: true, tension: .3, pointRadius: 0, borderWidth: 2.5 },
          { label: 'Random Classifier', data: roc.fpr,
            borderColor: 'rgba(94,128,153,.5)', borderDash: [5,5], pointRadius: 0, borderWidth: 1.5 },
        ],
      },
      options: { responsive: true, plugins: { legend: leg }, scales: ax },
    });

    // Confusion Matrix
    var cmEl = document.getElementById('cm-chart');
    var cmFlat = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]];
    if (cmEl) new Chart(cmEl, {
      type: 'bar',
      data: {
        labels: [
          'True Neg (' + cmFlat[0] + ')', 'False Pos (' + cmFlat[1] + ')',
          'False Neg (' + cmFlat[2] + ')', 'True Pos (' + cmFlat[3] + ')',
        ],
        datasets: [{
          data: cmFlat, borderRadius: 6,
          backgroundColor: ['rgba(0,212,170,.7)', 'rgba(244,63,94,.6)', 'rgba(244,63,94,.6)', 'rgba(0,212,170,.7)'],
          borderColor:     ['var(--accent)', 'var(--accentr)', 'var(--accentr)', 'var(--accent)'],
          borderWidth: 1.5,
        }],
      },
      options: { responsive: true, plugins: { legend: { display: false } }, scales: ax },
    });

    // Feature Coefficients
    var coefEl = document.getElementById('coef-chart');
    var coefLabels = Object.keys(coef).map(function (k) { return k.replace('_', ' '); });
    var coefVals   = Object.values(coef);
    var coefColors = ['rgba(0,212,170,.8)','rgba(14,165,233,.8)','rgba(245,158,11,.8)',
                      'rgba(167,139,250,.8)','rgba(244,63,94,.7)','rgba(0,212,170,.5)','rgba(14,165,233,.5)'];
    if (coefEl) new Chart(coefEl, {
      type: 'bar',
      data: {
        labels: coefLabels,
        datasets: [{ label: 'Coefficient', data: coefVals, borderRadius: 5, backgroundColor: coefColors, borderWidth: 0 }],
      },
      options: { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } }, scales: ax },
    });

    // Compare chart (CGPA & GRE)
    var cmpEl = document.getElementById('compare-chart');
    if (cmpEl) new Chart(cmpEl, {
      type: 'bar',
      data: {
        labels: ['Not Admitted', 'Admitted'],
        datasets: [
          { label: 'Avg CGPA', data: [gs.cgpa_rejected || 8.18, gs.cgpa_admitted || 9.09],
            backgroundColor: ['rgba(244,63,94,.7)', 'rgba(0,212,170,.7)'], borderRadius: 5, borderWidth: 0 },
          { label: 'GRE ÷ 10', data: [(gs.gre_rejected || 309.5) / 10, (gs.gre_admitted || 325.5) / 10],
            backgroundColor: ['rgba(244,63,94,.3)', 'rgba(0,212,170,.3)'], borderRadius: 5, borderWidth: 0 },
        ],
      },
      options: { responsive: true, plugins: { legend: leg }, scales: ax },
    });

    // Class distribution doughnut
    var distEl = document.getElementById('dist-chart');
    var cc = m.class_counts || { admitted: 218, not_admitted: 282 };
    if (distEl) new Chart(distEl, {
      type: 'doughnut',
      data: {
        labels: ['Not Admitted (' + cc.not_admitted + ')', 'Admitted (' + cc.admitted + ')'],
        datasets: [{
          data: [cc.not_admitted, cc.admitted],
          backgroundColor: ['rgba(244,63,94,.75)', 'rgba(0,212,170,.75)'],
          borderColor: ['var(--accentr)', 'var(--accent)'],
          borderWidth: 2, hoverOffset: 6,
        }],
      },
      options: { responsive: true, cutout: '65%', plugins: { legend: leg } },
    });

    // Research impact
    var resEl = document.getElementById('research-chart');
    if (resEl) new Chart(resEl, {
      type: 'bar',
      data: {
        labels: ['No Research', 'Has Research'],
        datasets: [{
          label: '% Admitted',
          data: [gs.res_rejected_rate || 34, gs.res_admitted_rate || 84],
          backgroundColor: ['rgba(244,63,94,.7)', 'rgba(0,212,170,.7)'],
          borderRadius: 6, borderWidth: 0,
        }],
      },
      options: {
        responsive: true, plugins: { legend: { display: false } },
        scales: { x: ax.x, y: { grid: gr, ticks: { color: tc, font: { family: MO, size: 10 } }, max: 100 } },
      },
    });
  }

  /* ─────────────────────────────────────────────────
     JUPYTER CODE VIEWER
  ───────────────────────────────────────────────── */
  var CELLS = [
    { type: 'md', title: 'Project Overview', content: '<h1>🎓 Capstone – Graduate Admission Prediction</h1><h2>1. Problem Statement</h2><p>Predict whether a student will be admitted to a graduate program based on academic scores and profile details using logistic regression, helping students assess their admission chances with data-driven insights.</p><h2>2. Objective</h2><p>Build a logistic regression model that classifies whether a student will be admitted based on GRE, TOEFL, university rating, SOP, LOR, CGPA, and research experience.</p>' },
    { type: 'code', title: '1. Import Libraries', code: '<span class="kw">import</span> pandas <span class="kw">as</span> pd\n<span class="kw">import</span> numpy <span class="kw">as</span> np\n<span class="kw">import</span> matplotlib.pyplot <span class="kw">as</span> plt\n<span class="kw">import</span> seaborn <span class="kw">as</span> sns\n<span class="kw">from</span> sklearn.model_selection <span class="kw">import</span> train_test_split\n<span class="kw">from</span> sklearn.linear_model <span class="kw">import</span> LogisticRegression\n<span class="kw">from</span> sklearn.tree <span class="kw">import</span> DecisionTreeClassifier\n<span class="kw">from</span> sklearn.neighbors <span class="kw">import</span> KNeighborsClassifier\n<span class="kw">from</span> sklearn.metrics <span class="kw">import</span> (\n    accuracy_score, precision_score, recall_score, f1_score,\n    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc\n)', output: 'Libraries imported successfully.' },
    { type: 'md', title: 'Dataset Overview', content: '<h2>3. Dataset Overview</h2><p>The <strong>Graduate Admission Prediction Dataset</strong> from Kaggle includes <strong>500 student profiles</strong>.</p><h3>Features: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research, Chance of Admit</h3>' },
    { type: 'code', title: '2. Load &amp; Preprocess', code: 'df = pd.<span class="fn">read_csv</span>(<span class="st">\'Admission_Predict.csv\'</span>)\ndf.columns = df.columns.str.<span class="fn">strip</span>().<span class="fn">str.replace</span>(<span class="st">\' \'</span>, <span class="st">\'_\'</span>)\n\n<span class="kw">if</span> <span class="st">\'Serial_No.\'</span> <span class="kw">in</span> df.columns:\n    df.<span class="fn">drop</span>(columns=[<span class="st">\'Serial_No.\'</span>], inplace=<span class="kw">True</span>)\n\n<span class="cm"># Create binary target variable</span>\ndf[<span class="st">\'Admitted\'</span>] = (df[<span class="st">\'Chance_of_Admit\'</span>] >= <span class="nu">0.75</span>).<span class="fn">astype</span>(<span class="kw">int</span>)\ndf.<span class="fn">drop</span>(columns=[<span class="st">\'Chance_of_Admit\'</span>], inplace=<span class="kw">True</span>)\n\n<span class="fn">print</span>(<span class="st">"Missing Values:"</span>)\n<span class="fn">print</span>(df.<span class="fn">isnull</span>().<span class="fn">sum</span>())\n<span class="fn">print</span>(<span class="st">"Duplicates:"</span>, df.<span class="fn">duplicated</span>().<span class="fn">sum</span>())', output: 'Missing Values:\nGRE_Score           0\nTOEFL_Score         0\nUniversity_Rating   0\nSOP                 0\nLOR                 0\nCGPA                0\nResearch            0\nAdmitted            0\ndtype: int64\n\nDuplicates: 0' },
    { type: 'code', title: '3. EDA — Correlation Heatmap', code: 'plt.figure(figsize=(<span class="nu">10</span>, <span class="nu">8</span>))\nsns.<span class="fn">heatmap</span>(df.<span class="fn">corr</span>(), annot=<span class="kw">True</span>, cmap=<span class="st">\'coolwarm\'</span>, fmt=<span class="st">".2f"</span>)\nplt.<span class="fn">title</span>(<span class="st">\'Feature Correlation Heatmap\'</span>)\nplt.<span class="fn">show</span>()\n\nsns.<span class="fn">boxplot</span>(x=<span class="st">\'Admitted\'</span>, y=<span class="st">\'CGPA\'</span>, data=df)\nplt.<span class="fn">title</span>(<span class="st">\'CGPA by Admission Status\'</span>)\nplt.<span class="fn">show</span>()', output: '[Heatmap displayed — CGPA has highest correlation 0.87 with admission]\n[Boxplot displayed — clear separation at CGPA ≈ 8.5]' },
    { type: 'code', title: '4. Train Logistic Regression', code: 'X = df.<span class="fn">drop</span>(<span class="st">"Admitted"</span>, axis=<span class="nu">1</span>)\ny = df[<span class="st">"Admitted"</span>]\n\nX_train, X_test, y_train, y_test = <span class="fn">train_test_split</span>(\n    X, y, test_size=<span class="nu">0.25</span>, random_state=<span class="nu">42</span>, stratify=y\n)\n\nmodel = <span class="fn">LogisticRegression</span>(max_iter=<span class="nu">2000</span>)\nmodel.<span class="fn">fit</span>(X_train, y_train)\n\ny_pred  = model.<span class="fn">predict</span>(X_test)\ny_proba = model.<span class="fn">predict_proba</span>(X_test)[:, <span class="nu">1</span>]', output: 'Model trained successfully.\nTrain size: 375  |  Test size: 125' },
    { type: 'code', title: '5. Evaluation Metrics', code: '<span class="fn">print</span>(<span class="fn">classification_report</span>(y_test, y_pred))\n<span class="fn">print</span>(<span class="st">"Accuracy :"</span>, <span class="fn">accuracy_score</span>(y_test, y_pred))\n<span class="fn">print</span>(<span class="st">"Precision:"</span>, <span class="fn">precision_score</span>(y_test, y_pred))\n<span class="fn">print</span>(<span class="st">"Recall   :"</span>, <span class="fn">recall_score</span>(y_test, y_pred))\n<span class="fn">print</span>(<span class="st">"F1 Score :"</span>, <span class="fn">f1_score</span>(y_test, y_pred))\n<span class="fn">print</span>(<span class="st">"AUC      :"</span>, <span class="fn">roc_auc_score</span>(y_test, y_proba))', output: 'Accuracy : 0.912\nPrecision: 0.9216\nRecall   : 0.8704\nF1 Score : 0.8952\nAUC      : 0.9757' },
    { type: 'code', title: '6. ROC Curve', code: 'fpr, tpr, _ = <span class="fn">roc_curve</span>(y_test, y_proba)\nroc_auc = <span class="fn">auc</span>(fpr, tpr)\n\nplt.<span class="fn">figure</span>(figsize=(<span class="nu">8</span>, <span class="nu">6</span>))\nplt.<span class="fn">plot</span>(fpr, tpr, label=<span class="fn">f</span><span class="st">"ROC Curve (AUC = {roc_auc:.2f})"</span>, color=<span class="st">\'darkorange\'</span>, lw=<span class="nu">2</span>)\nplt.<span class="fn">plot</span>([<span class="nu">0</span>,<span class="nu">1</span>], [<span class="nu">0</span>,<span class="nu">1</span>], <span class="st">\'k--\'</span>, label=<span class="st">"Random Classifier"</span>)\nplt.<span class="fn">xlabel</span>(<span class="st">"False Positive Rate"</span>)\nplt.<span class="fn">ylabel</span>(<span class="st">"True Positive Rate"</span>)\nplt.<span class="fn">title</span>(<span class="st">"Receiver Operating Characteristic"</span>)\nplt.<span class="fn">legend</span>(loc=<span class="st">"lower right"</span>)\nplt.<span class="fn">show</span>()', output: '[ROC curve plotted — AUC = 0.98, excellent discrimination]' },
    { type: 'md', title: 'Failed Models', content: '<h2>8. Failed Models</h2><h3>Decision Tree</h3><p>Prone to overfitting, lower recall, unstable structure, no natural probability output.</p><h3>KNN</h3><p>High computational cost, sensitive to mixed-scale features, performance varied widely with k.</p>' },
    { type: 'code', title: '7. Decision Tree Comparison', code: 'results = []\n<span class="kw">for</span> depth <span class="kw">in</span> <span class="fn">range</span>(<span class="nu">1</span>, <span class="nu">11</span>):\n    tree = <span class="fn">DecisionTreeClassifier</span>(max_depth=depth, random_state=<span class="nu">42</span>)\n    tree.<span class="fn">fit</span>(X_train, y_train)\n    y_pred_t  = tree.<span class="fn">predict</span>(X_test)\n    y_proba_t = tree.<span class="fn">predict_proba</span>(X_test)[:, <span class="nu">1</span>]\n    results.<span class="fn">append</span>({\n        <span class="st">\'Model\'</span>: <span class="fn">f</span><span class="st">\'Decision Tree (depth={depth})\'</span>,\n        <span class="st">\'Accuracy\'</span>: <span class="fn">accuracy_score</span>(y_test, y_pred_t),\n        <span class="st">\'AUC\'</span>: <span class="fn">roc_auc_score</span>(y_test, y_proba_t)\n    })\npd.<span class="fn">DataFrame</span>(results)', output: 'Best Decision Tree: 0.864 accuracy — below Logistic Regression 0.912' },
  ];

  var codeFilter = 'all';

  function buildCode() {
    renderNotebook();
    document.querySelectorAll('.code-tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.code-tab').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        codeFilter = this.getAttribute('data-ctab');
        renderNotebook();
      });
    });
  }

  function renderNotebook() {
    var cells = CELLS.filter(function (c) {
      return codeFilter === 'all' ||
        (codeFilter === 'code' && c.type === 'code') ||
        (codeFilter === 'md'   && c.type === 'md');
    });
    var nb = document.getElementById('notebook');
    if (!nb) return;
    nb.innerHTML = '';
    cells.forEach(function (cell, idx) {
      var div = document.createElement('div');
      div.className = 'nb-cell';

      var hd = document.createElement('div');
      hd.className = 'nb-cell-hd';
      hd.innerHTML =
        '<span class="nb-cell-type ' + cell.type + '">' + (cell.type === 'code' ? 'Code' : 'Markdown') + '</span>' +
        '<span class="nb-cell-num">[' + (idx + 1) + ']</span>' +
        '<span class="nb-cell-title">' + cell.title + '</span>' +
        '<span class="nb-cell-toggle">▶</span>';

      var body = document.createElement('div');
      body.className = 'nb-cell-body';

      if (cell.type === 'code') {
        body.innerHTML = '<pre>' + cell.code + '</pre>' +
          (cell.output ? '<div class="nb-output">Out[' + (idx + 1) + ']:\n' + cell.output + '</div>' : '');
      } else {
        body.innerHTML = '<div class="nb-md">' + cell.content + '</div>';
      }

      hd.addEventListener('click', function () {
        var tog = hd.querySelector('.nb-cell-toggle');
        if (body.classList.contains('open')) {
          body.classList.remove('open'); tog.classList.remove('open');
        } else {
          body.classList.add('open'); tog.classList.add('open');
        }
      });

      div.appendChild(hd);
      div.appendChild(body);
      nb.appendChild(div);
    });
  }

  /* ─────────────────────────────────────────────────
     DATASET TABLE
  ───────────────────────────────────────────────── */
  var PAGE = 15, curPage = 1;

  function buildTable() {
    var si = document.getElementById('search-input');
    var af = document.getElementById('admit-filter');
    var rf = document.getElementById('res-filter');
    if (si) si.addEventListener('input',  doFilter);
    if (af) af.addEventListener('change', doFilter);
    if (rf) rf.addEventListener('change', doFilter);
    fetchPage(1);
  }

  function doFilter() { fetchPage(1); }

  function fetchPage(page) {
    curPage = page;
    var q  = (document.getElementById('search-input')  || {}).value || '';
    var af = (document.getElementById('admit-filter')   || {}).value || '';
    var rf = (document.getElementById('res-filter')     || {}).value || '';

    var url = API.dataset +
      '?page=' + page + '&per_page=' + PAGE +
      '&search=' + encodeURIComponent(q) +
      '&admitted=' + af + '&research=' + rf;

    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) { drawTable(data); })
      .catch(function () { toast('Could not load dataset.', 'error'); });
  }

  function drawTable(data) {
    var tbody = document.getElementById('data-tbody');
    if (!tbody) return;
    tbody.innerHTML = data.records.map(function (r, i) {
      var rowNum = (curPage - 1) * PAGE + i + 1;
      return '<tr>' +
        '<td style="color:var(--muted)">' + rowNum + '</td>' +
        '<td>' + r.GRE_Score + '</td>' +
        '<td>' + r.TOEFL_Score + '</td>' +
        '<td>' + r.University_Rating + '</td>' +
        '<td>' + r.SOP + '</td>' +
        '<td>' + r.LOR + '</td>' +
        '<td>' + r.CGPA + '</td>' +
        '<td>' + (r.Research ? '✓' : '–') + '</td>' +
        '<td><span class="badge ' + (r.Admitted ? 'badge-admitted' : 'badge-rejected') + '">' +
          (r.Admitted ? 'Admitted' : 'Rejected') + '</span></td>' +
        '</tr>';
    }).join('');

    var rc = document.getElementById('row-count');
    if (rc) rc.textContent = 'Showing ' + data.total + ' records';

    drawPages(data.total);
  }

  function drawPages(total) {
    var pages = Math.ceil(total / PAGE);
    var pg = document.getElementById('pagination');
    if (!pg) return;
    pg.innerHTML = '';
    for (var i = 1; i <= Math.min(pages, 10); i++) {
      var btn = document.createElement('button');
      btn.className = 'page-btn' + (i === curPage ? ' active' : '');
      btn.textContent = i;
      btn.setAttribute('data-p', i);
      btn.addEventListener('click', function () { fetchPage(parseInt(this.getAttribute('data-p'), 10)); });
      pg.appendChild(btn);
    }
    if (pages > 10) {
      var sp = document.createElement('span');
      sp.style.cssText = 'color:var(--muted);font-size:.7rem;padding:6px';
      sp.textContent = '… ' + pages + ' pages';
      pg.appendChild(sp);
    }
  }

})();
