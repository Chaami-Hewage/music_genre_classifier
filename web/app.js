const $ = (id) => document.getElementById(id);

const statusBadge = $("statusBadge");
const lyricsEl = $("lyrics");
const btnClassify = $("btnClassify");
const btnClear = $("btnClear");
const btnExample = $("btnExample");
const predictedEl = $("predicted");
const latencyPill = $("latencyPill");
const charsPill = $("charsPill");
const tableBody = $("table").querySelector("tbody");

const exampleLyrics =
  "This is a short example lyric.\\n" +
  "Replace this with any real song lyrics for best results.\\n" +
  "The model works best when you paste 2-3 verses or more.";

function setBadge(state, text) {
  statusBadge.classList.remove("ok", "bad");
  if (state) statusBadge.classList.add(state);
  statusBadge.querySelector(".text").textContent = text;
}

function fmtPct(x) {
  if (!Number.isFinite(x)) return "—";
  return (x * 100).toFixed(2) + "%";
}

function setBusy(busy) {
  btnClassify.disabled = busy;
  btnClassify.textContent = busy ? "Classifying…" : "Classify";
}

let chart;
function ensureChart() {
  if (chart) return chart;
  const ctx = $("chart");
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Score",
          data: [],
          borderWidth: 1,
          borderColor: "rgba(110,168,254,0.8)",
          backgroundColor: "rgba(110,168,254,0.25)",
          borderRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => " " + fmtPct(ctx.raw),
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "rgba(231,236,255,0.82)" },
          grid: { color: "rgba(255,255,255,0.08)" },
        },
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            color: "rgba(231,236,255,0.82)",
            callback: (v) => (v * 100).toFixed(0) + "%",
          },
          grid: { color: "rgba(255,255,255,0.08)" },
        },
      },
    },
  });
  return chart;
}

function renderResult(result, latencyMs, textLen) {
  predictedEl.textContent = result?.predicted ?? "—";
  latencyPill.textContent = Number.isFinite(latencyMs) ? `${latencyMs}ms` : "—";
  charsPill.textContent = Number.isFinite(textLen) ? `${textLen} chars` : "—";

  const scores = Array.isArray(result?.scores) ? result.scores : [];
  const labels = scores.map((s) => s.label);
  const values = scores.map((s) => s.score);

  const c = ensureChart();
  c.data.labels = labels;
  c.data.datasets[0].data = values;
  c.update();

  tableBody.innerHTML = "";
  for (const item of scores) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    const td2 = document.createElement("td");
    td1.textContent = item.label;
    td2.textContent = fmtPct(item.score);
    td2.className = "num";
    tr.appendChild(td1);
    tr.appendChild(td2);
    tableBody.appendChild(tr);
  }
}

async function healthCheck() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (!res.ok) throw new Error("bad status");
    setBadge("ok", "Backend online");
    return true;
  } catch {
    setBadge("bad", "Backend offline");
    return false;
  }
}

async function classify() {
  const text = (lyricsEl.value || "").trim();
  if (!text) return;

  setBusy(true);
  const started = performance.now();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "text/plain; charset=utf-8" },
      body: text,
    });
    const latency = Math.round(performance.now() - started);
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || `HTTP ${res.status}`);
    }
    const json = await res.json();
    renderResult(json, latency, text.length);
  } catch (e) {
    predictedEl.textContent = "—";
    latencyPill.textContent = "—";
    charsPill.textContent = "—";
    tableBody.innerHTML =
      `<tr><td colspan="2">Error: ${String(e?.message || e)}</td></tr>`;
  } finally {
    setBusy(false);
  }
}

btnClear.addEventListener("click", () => {
  lyricsEl.value = "";
  lyricsEl.focus();
});

btnExample.addEventListener("click", () => {
  lyricsEl.value = exampleLyrics;
  lyricsEl.focus();
});

btnClassify.addEventListener("click", classify);

lyricsEl.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") classify();
});

(async function init() {
  setBadge(null, "Checking…");
  await healthCheck();
  setInterval(healthCheck, 2500);
  ensureChart();
})();

