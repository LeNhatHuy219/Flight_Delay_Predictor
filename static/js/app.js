/**
 * Flight Delay Predictor — Frontend JS
 * Handles: tab switching, metadata, prediction, feature importance chart, history.
 */
(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────
  const $origin    = document.getElementById("origin");
  const $dest      = document.getElementById("dest");
  const $depDate   = document.getElementById("dep-date");
  const $depTime   = document.getElementById("dep-time");
  const $airline   = document.getElementById("airline");
  const $btn       = document.getElementById("predict-btn");
  const $result    = document.getElementById("result-area");
  const $history   = document.getElementById("history-area");
  const $mapBox    = document.getElementById("map-container");
  const $topMetric = document.getElementById("top-metric");
  const $metricGrid  = document.getElementById("metric-grid");
  const $leaderboard = document.getElementById("leaderboard-card");
  const $stageCard   = document.getElementById("stage-card");
  const $fiCard      = document.getElementById("fi-card");

  let meta = {};
  let predictions = [];   // session history
  let currentMode = "single";

  // ── Mode toggle ────────────────────────────────
  document.querySelectorAll(".mode-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".mode-btn").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      currentMode = btn.dataset.mode;
      var airlineSection = document.getElementById("airline-section");
      airlineSection.style.display = currentMode === "single" ? "" : "none";
      $result.innerHTML = "";
    });
  });

  // ── Tabs ───────────────────────────────────────
  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      document.querySelectorAll(".tab").forEach(function (t) { t.classList.remove("active"); });
      document.querySelectorAll(".tab-pane").forEach(function (p) { p.classList.remove("active"); });
      tab.classList.add("active");
      document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
    });
  });

  // ── Populate a <select> ────────────────────────
  function fillSelect(el, items, nameMap) {
    el.innerHTML = "";
    items.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v;
      opt.textContent = nameMap && nameMap[v] ? v + " — " + nameMap[v] : v;
      el.appendChild(opt);
    });
  }

  // ── Keep origin ≠ dest ─────────────────────────
  function syncAirports() {
    var ov = $origin.value;
    if ($dest.value === ov) {
      var alt = Array.from($dest.options).find(function (o) { return o.value !== ov; });
      if (alt) $dest.value = alt.value;
    }
  }

  function showFieldError(msg) {
    $result.innerHTML = '<div class="note" style="color:var(--red);border-left-color:var(--red)">' + msg + '</div>';
  }

  // ── Populate departure time dropdown ───────────
  function fillTimeSelect() {
    $depTime.innerHTML = "";
    for (var h = 0; h < 24; h++) {
      for (var m = 0; m < 60; m += 15) {
        var hh = String(h).padStart(2, "0");
        var mm = String(m).padStart(2, "0");
        var opt = document.createElement("option");
        opt.value = hh + ":" + mm;
        opt.textContent = hh + ":" + mm;
        if (h === 8 && m === 0) opt.selected = true;
        $depTime.appendChild(opt);
      }
    }
  }

  // ── Load metadata ──────────────────────────────
  function loadMeta() {
    fetch("/api/meta")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        meta = d;
        fillSelect($origin, d.origins || []);
        fillSelect($dest, d.dests || []);
        fillSelect($airline, d.airlines || [], d.airline_names || {});
        fillTimeSelect();

        if ($dest.options.length > 1) $dest.selectedIndex = 1;
        $origin.addEventListener("change", syncAirports);
        $dest.addEventListener("change", function () {
          if ($dest.value === $origin.value) syncAirports();
        });

        $topMetric.innerHTML =
          "Model: <strong>" + (d.model_name || "—") + "</strong>" +
          (d.source_rows ? " &nbsp;·&nbsp; " + Number(d.source_rows).toLocaleString() + " rows" : "");

        renderMetrics(d);
        loadFeatureImportance();
      })
      .catch(function (err) {
        console.error("Meta load error:", err);
        $topMetric.textContent = "Không thể tải metadata";
      });
  }

  // ── Feature Importance ─────────────────────────
  function loadFeatureImportance() {
    fetch("/api/feature_importance")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error || !d.features) return;
        renderFeatureImportance(d);
      })
      .catch(function () {});
  }

  function renderFeatureImportance(d) {
    var features   = d.features;
    var importances = d.importances;
    var maxVal = importances[0] || 1;

    // Friendly label map
    var labelMap = {
      "dep_seconds":        "Giờ cất cánh (giây)",
      "dep_hour_sin":       "Giờ bay (sin)",
      "dep_hour_cos":       "Giờ bay (cos)",
      "dep_dayofweek":      "Ngày trong tuần",
      "scheduled_time":     "Thời lượng bay",
      "tavg":               "Nhiệt độ TB",
      "wspd":               "Tốc độ gió",
      "prcp":               "Lượng mưa",
      "snow":               "Lượng tuyết",
      "prev_avg_delay_3":   "Delay TB 3 chuyến trước",
      "prev_max_delay_3":   "Delay max 3 chuyến trước",
      "AIRLINE":            "Hãng hàng không",
      "ORIGIN_AIRPORT":     "Sân bay xuất phát",
      "DESTINATION_AIRPORT":"Sân bay đến",
    };

    // Colour tiers
    function barColor(pct) {
      if (pct >= 60) return "var(--cyan)";
      if (pct >= 30) return "var(--blue)";
      return "var(--indigo)";
    }

    var html = '<div class="card-title">Feature Importance — Top ' + features.length + '</div>';
    html += '<div class="fi-list">';

    features.forEach(function (f, i) {
      var imp  = importances[i];
      var pct  = (imp / maxVal * 100).toFixed(1);
      var label = labelMap[f] || f;
      var color = barColor(parseFloat(pct));
      html +=
        '<div class="fi-row">' +
          '<div class="fi-label">' + label + '</div>' +
          '<div class="fi-bar-wrap">' +
            '<div class="fi-bar" style="width:' + pct + '%;background:' + color + '"></div>' +
          '</div>' +
          '<div class="fi-val">' + imp.toFixed(1) + '%</div>' +
        '</div>';
    });
    html += '</div>';
    $fiCard.innerHTML = html;
    $fiCard.style.display = "";
  }

  // ── Metrics tab ────────────────────────────────
  function renderMetrics(d) {
    var m = d.metrics || {};
    var cards = "";
    var pairs = [
      ["RMSE", m.rmse || m.RMSE, "grad"],
      ["MAE",  m.mae  || m.MAE,  "teal"],
      ["R²",   m.r2   || m.R2,   "amber"],
      ["Training rows", d.source_rows, "grad"]
    ];
    pairs.forEach(function (p) {
      if (p[1] == null) return;
      var val = typeof p[1] === "number" ? (p[1] % 1 ? p[1].toFixed(4) : Number(p[1]).toLocaleString()) : p[1];
      cards += '<div class="metric-card"><div class="metric-label">' + p[0] +
               '</div><div class="metric-value ' + p[2] + '">' + val + '</div></div>';
    });
    $metricGrid.innerHTML = cards;
    renderLeaderboard(d.ml_leaderboard || d.combined_leaderboard, $leaderboard, "Bảng xếp hạng mô hình");
    renderLeaderboard(d.stage_leaderboard, $stageCard, "Leaderboard theo giai đoạn");
  }

  function renderLeaderboard(lb, container, title) {
    if (!lb || !lb.length) { container.style.display = "none"; return; }
    container.style.display = "";
    var keys = Object.keys(lb[0]);
    var html = '<div class="card-title">' + title + '</div><table class="lb-table"><thead><tr>';
    keys.forEach(function (k) { html += "<th>" + k + "</th>"; });
    html += "</tr></thead><tbody>";
    lb.forEach(function (row) {
      html += "<tr>";
      keys.forEach(function (k) {
        var v = row[k];
        html += "<td>" + (typeof v === "number" ? (v % 1 ? v.toFixed(4) : v) : (v || "")) + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  // ── Predict ────────────────────────────────────
  $btn.addEventListener("click", function () {
    if ($origin.value === $dest.value) {
      showFieldError(" Sân bay xuất phát và sân bay đến không được trùng nhau.");
      return;
    }
    $btn.disabled = true;
    $btn.textContent = "Đang tính…";

    var payload = {
      origin: $origin.value,
      dest: $dest.value,
      date: $depDate.value,
      time: $depTime.value,
      mode: currentMode,
      airline: $airline.value,
      prev_avg_delay_3: 0,
      prev_max_delay_3: 0,
      tavg: 0, wspd: 0, prcp: 0, snow: 0
    };

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) {
          $result.innerHTML = '<div class="note" style="color:var(--red)">' + d.error + '</div>';
          renderMapEmpty();
        } else if (d.mode === "all") {
          renderAllResult(d);
          addHistory({
            origin:  $origin.value,
            dest:    $dest.value,
            airline: "ALL",
            date:    $depDate.value,
            time:    $depTime.value,
            delay:   d.avg_delay,
            status_key:  d.status_key,
            status_text: d.status_text,
          });
        } else {
          renderSingleResult(d);
          addHistory({
            origin:  $origin.value,
            dest:    $dest.value,
            airline: $airline.value,
            date:    $depDate.value,
            time:    $depTime.value,
            delay:   d.delay,
            status_key:  d.status_key,
            status_text: d.status_text,
          });
        }
      })
      .catch(function (err) {
        $result.innerHTML = '<div class="note" style="color:var(--red)">Lỗi kết nối: ' + err + '</div>';
        renderMapEmpty();
      })
      .finally(function () {
        $btn.disabled = false;
        $btn.textContent = "Dự đoán";
      });
  });

  // ── Render single result ───────────────────────
  function renderSingleResult(d) {
    var name = (meta.airline_names || {})[d.airline] || d.airline;
    var html =
      '<div class="result-card">' +
        '<div class="result-tag">Dự đoán delay</div>' +
        '<div class="delay-num ' + d.status_key + '">' + d.delay + '</div>' +
        '<div class="delay-unit">phút</div>' +
        '<span class="status ' + d.status_key + '">' + d.status_text + '</span>' +
        '<div class="pill-row">' +
          '<div class="pill"><div class="pill-label">Hãng</div><div class="pill-value">' + name + '</div></div>' +
          '<div class="pill"><div class="pill-label">Xuất phát</div><div class="pill-value">' + d.origin + '</div></div>' +
          '<div class="pill"><div class="pill-label">Đến</div><div class="pill-value">' + d.dest + '</div></div>' +
          '<div class="pill"><div class="pill-label">Giờ</div><div class="pill-value">' + d.time + '</div></div>' +
          (d.scheduled_minutes ? '<div class="pill"><div class="pill-label">TG bay ước tính</div><div class="pill-value">' + d.scheduled_minutes + ' ph</div></div>' : '') +
        '</div>' +
        '<div class="note">Kết quả ước lượng dựa trên dữ liệu lịch sử năm 2015.</div>' +
      '</div>';
    $result.innerHTML = html;
    renderMap(d.map_html);
  }

  // ── Render all-airlines result ─────────────────
  function renderAllResult(d) {
    var rows = d.rankings || [];
    var avg  = d.avg_delay;
    var sk   = d.status_key;

    var tableRows = rows.map(function (r, i) {
      var medal = i === 0
        ? '<i class="fa-solid fa-trophy" style="color:#f59e0b;margin-right:.35rem"></i>'
        : i === 1
        ? '<i class="fa-solid fa-medal" style="color:#94a3b8;margin-right:.35rem"></i>'
        : i === 2
        ? '<i class="fa-solid fa-medal" style="color:#c2773a;margin-right:.35rem"></i>'
        : '<span style="color:var(--muted);font-size:.82rem;margin-right:.35rem">' + (i + 1) + '.</span>';
      var delayClass = r.delay <= 5 ? "on-time" : r.delay <= 20 ? "slight" : "delayed";
      return '<tr' + (i === 0 ? ' class="best-row"' : '') + '>' +
        '<td>' + medal + '<strong>' + r.airline + '</strong></td>' +
        '<td style="color:var(--muted);font-size:.82rem">' + (r.name || "") + '</td>' +
        '<td class="hist-delay ' + delayClass + '" style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;text-align:right">' + r.delay + ' ph</td>' +
      '</tr>';
    }).join("");

    var html =
      '<div class="rankings-card">' +
        '<div class="result-tag" style="text-align:center">Delay trung bình tất cả hãng</div>' +
        '<div class="avg-row">' +
          '<span class="avg-num ' + sk + '">' + avg + '</span>' +
          '<span class="avg-label">phút</span>' +
        '</div>' +
        '<div style="text-align:center;margin-bottom:1rem">' +
          '<span class="status ' + sk + '">' + d.status_text + '</span>' +
        '</div>' +
        '<div class="card-title">Xếp hạng theo delay dự báo</div>' +
        '<table class="rankings-table">' +
          '<thead><tr><th>Hãng</th><th>Tên đầy đủ</th><th style="text-align:right">Delay</th></tr></thead>' +
          '<tbody>' + tableRows + '</tbody>' +
        '</table>' +
      '</div>';

    $result.innerHTML = html;
    renderMap(d.map_html);
  }

  // ── History ────────────────────────────────────
  function addHistory(entry) {
    predictions.unshift(entry);   // newest first
    renderHistory();
  }

  function renderHistory() {
    if (!predictions.length) { $history.innerHTML = ""; return; }

    var html =
      '<div class="history-card">' +
        '<div class="history-header">' +
          '<span class="card-title" style="margin:0">Lịch sử dự đoán</span>' +
          '<button class="history-clear" id="clear-history">Xoá</button>' +
        '</div>' +
        '<div class="history-scroll">' +
          '<table class="history-table">' +
            '<thead><tr>' +
              '<th>#</th><th>Route</th><th>Hãng</th><th>Ngày</th><th>Giờ</th><th>Delay</th><th>Trạng thái</th>' +
            '</tr></thead><tbody>';

    predictions.forEach(function (p, i) {
      html +=
        '<tr>' +
          '<td class="hist-num">' + (i + 1) + '</td>' +
          '<td><span class="route-badge">' + p.origin + '</span> → <span class="route-badge">' + p.dest + '</span></td>' +
          '<td>' + p.airline + '</td>' +
          '<td>' + p.date + '</td>' +
          '<td>' + p.time + '</td>' +
          '<td class="hist-delay ' + p.status_key + '">' + p.delay + ' ph</td>' +
          '<td><span class="status-dot ' + p.status_key + '"></span>' + p.status_text + '</td>' +
        '</tr>';
    });

    html += '</tbody></table></div></div>';
    $history.innerHTML = html;

    document.getElementById("clear-history").addEventListener("click", function () {
      predictions = [];
      $history.innerHTML = "";
    });
  }

  // ── Map ────────────────────────────────────────
  function renderMap(mapHtml) {
    if (mapHtml) {
      var blob = new Blob([mapHtml], { type: "text/html" });
      var url  = URL.createObjectURL(blob);
      var iframe = document.createElement("iframe");
      iframe.style.cssText = "width:100%;min-height:345px;border:none;display:block;";
      iframe.onload = function () { URL.revokeObjectURL(url); };
      iframe.src = url;
      $mapBox.innerHTML = "";
      $mapBox.appendChild(iframe);
    } else {
      renderMapEmpty();
    }
  }

  function renderMapEmpty() {
    $mapBox.innerHTML =
      '<div class="map-empty"><div>' +
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;margin-bottom:.35rem">Lộ trình sẽ hiển thị sau khi dự đoán</div>' +
        '<div style="font-size:.82rem">Bản đồ dùng tọa độ từ data/airports.csv.</div>' +
      '</div></div>';
  }

  // ── Init ───────────────────────────────────────
  loadMeta();

})();
