// =============================================================================
// SHAKTI + FPGA AI INFERENCE SIMULATOR ENGINE (app.js)
// =============================================================================

const PRESETS = {
  mnist: {
    A: [[2, 1, 0, 3], [1, 4, 2, 1], [3, 0, 1, 2], [2, 2, 3, 1]],
    B: [[1, -1, 2, 0], [2, 3, -1, 1], [0, 2, 1, -2], [1, 0, 3, 2]],
    bias: [1, 0, -2, 2]
  },
  edge: {
    A: [[3, 2, 1, 0], [0, 2, 4, 1], [1, 3, 2, 2], [4, 1, 0, 2]],
    B: [[2, 0, -1, 1], [1, 2, 1, 0], [-1, 1, 3, 2], [0, -2, 1, 3]],
    bias: [0, 2, 1, -1]
  },
  identity: {
    A: [[5, -2, 3, 1], [4, 0, -3, 2], [1, 6, 2, -4], [2, 3, 5, 7]],
    B: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    bias: [0, 0, 0, 0]
  }
};

const STAGES = [
  { id: 0, short: "0. Idle", desc: "System initialized. Edit Matrix A/B or click Step Phase.", cline: 1, hwCycles: 0 },
  { id: 1, short: "1. SW Quant", desc: "SHAKTI CPU packs INT8 activations & weights into 32-bit words.", cline: 1, hwCycles: 4 },
  { id: 2, short: "2. AXI Write A", desc: "SHAKTI writes 4 packed rows of Matrix A to 0x40020010..1C.", cline: 2, hwCycles: 12 },
  { id: 3, short: "3. AXI Write B", desc: "SHAKTI writes 4 packed rows of Matrix B to 0x40020020..2C.", cline: 3, hwCycles: 20 },
  { id: 4, short: "4. Trigger HW", desc: "SHAKTI writes 0x01 to CTRL_REG (0x40020000) to start FPGA.", cline: 4, hwCycles: 22 },
  { id: 5, short: "5. PE Cycle k=0", desc: "FPGA 16 PEs compute C[i][j] += A[i][0] * B[0][j] in parallel.", cline: 5, hwCycles: 23, k: 0 },
  { id: 6, short: "6. PE Cycle k=1", desc: "FPGA 16 PEs compute C[i][j] += A[i][1] * B[1][j] in parallel.", cline: 5, hwCycles: 24, k: 1 },
  { id: 7, short: "7. PE Cycle k=2", desc: "FPGA 16 PEs compute C[i][j] += A[i][2] * B[2][j] in parallel.", cline: 5, hwCycles: 25, k: 2 },
  { id: 8, short: "8. PE Cycle k=3", desc: "FPGA 16 PEs compute C[i][j] += A[i][3] * B[3][j] & assert DONE.", cline: 5, hwCycles: 26, k: 3 },
  { id: 9, short: "9. AXI Read C", desc: "SHAKTI reads 16 accumulated 32-bit outputs from 0x40020030..6C.", cline: 6, hwCycles: 44 },
  { id: 10, short: "10. SW ReLU", desc: "SHAKTI applies Bias + ReLU(x) = max(0,x) and Argmax classification.", cline: 7, hwCycles: 68 }
];

let matA = JSON.parse(JSON.stringify(PRESETS.mnist.A));
let matB = JSON.parse(JSON.stringify(PRESETS.mnist.B));
let biasVec = [...PRESETS.mnist.bias];
let peAccum = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
let currentStage = 0;
let autoPlayTimer = null;
let benchmarkChartInstance = null;

function packRowHex(row) {
  let val = 0;
  for (let i = 0; i < 4; i++) {
    const byteVal = (row[i] & 0xFF) >>> 0;
    val = (val | (byteVal << (i * 8))) >>> 0;
  }
  return "0x" + val.toString(16).toUpperCase().padStart(8, "0");
}

function logEvent(msg, highlight = false) {
  const box = document.getElementById("simConsole");
  const div = document.createElement("div");
  div.className = highlight ? "text-emerald-400 font-semibold" : "text-slate-300";
  div.textContent = `[Stage ${currentStage}] ${msg}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function renderMatrices() {
  const aGrid = document.getElementById("matrixAGrid");
  const bGrid = document.getElementById("matrixBGrid");
  aGrid.innerHTML = "";
  bGrid.innerHTML = "";

  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const inpA = document.createElement("input");
      inpA.type = "number";
      inpA.value = matA[r][c];
      inpA.className = "w-full bg-slate-900 border border-slate-700 rounded text-center text-xs py-1 text-sky-300 mono focus:outline-none focus:border-sky-400";
      inpA.addEventListener("change", (e) => {
        matA[r][c] = Math.max(-128, Math.min(127, parseInt(e.target.value, 10) || 0));
        resetSimulation();
      });
      aGrid.appendChild(inpA);

      const inpB = document.createElement("input");
      inpB.type = "number";
      inpB.value = matB[r][c];
      inpB.className = "w-full bg-slate-900 border border-slate-700 rounded text-center text-xs py-1 text-amber-300 mono focus:outline-none focus:border-amber-400";
      inpB.addEventListener("change", (e) => {
        matB[r][c] = Math.max(-128, Math.min(127, parseInt(e.target.value, 10) || 0));
        resetSimulation();
      });
      bGrid.appendChild(inpB);
    }
  }
}

function renderPipelineBar() {
  const container = document.getElementById("pipelineSteps");
  container.innerHTML = "";
  for (let i = 1; i <= 10; i++) {
    const st = STAGES[i];
    const active = currentStage === i;
    const done = currentStage > i;
    const div = document.createElement("div");
    div.className = `px-2 py-1.5 rounded-lg border text-center font-medium transition ${
      active ? "bg-sky-600/30 border-sky-400 text-white font-bold" :
      done ? "bg-emerald-950/50 border-emerald-700/50 text-emerald-300" :
      "bg-slate-950 border-slate-800 text-slate-500"
    }`;
    div.textContent = st.short;
    container.appendChild(div);
  }
}

function renderSystolicGrid(activeK = null) {
  const grid = document.getElementById("systolicGrid");
  grid.innerHTML = "";
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const cell = document.createElement("div");
      const isComputing = activeK !== null;
      cell.className = `p-2.5 rounded-xl border border-slate-800 bg-slate-950 transition flex flex-col justify-between ${
        isComputing ? "pe-active" : ""
      }`;
      const prod = isComputing ? (matA[r][activeK] * matB[activeK][c]) : null;
      const exprText = isComputing
        ? `${matA[r][activeK]}&times;${matB[activeK][c]} = ${prod >= 0 ? "+" + prod : prod}`
        : "MAC Ready";
      cell.innerHTML = `
        <div class="flex items-center justify-between text-[10px] text-slate-400 mono">
          <span>PE[${r},${c}]</span>
          <span class="${isComputing ? 'text-sky-300 font-bold' : 'text-slate-500'}">${exprText}</span>
        </div>
        <div class="text-center my-1">
          <span class="text-lg font-extrabold mono ${peAccum[r][c] !== 0 ? 'text-white' : 'text-slate-500'}">${peAccum[r][c]}</span>
        </div>
        <div class="text-[9px] text-slate-500 mono text-right">32-bit Acc</div>
      `;
      grid.appendChild(cell);
    }
  }
}

function renderRegisterTable() {
  const tbody = document.getElementById("regTableBody");
  const ctrlVal = currentStage === 4 ? "0x00000001" : "0x00000000";
  let statusVal = "0x00000000";
  if (currentStage >= 5 && currentStage <= 7) {
    const cyc = (currentStage - 4) << 8;
    statusVal = "0x" + (cyc | 0x01).toString(16).toUpperCase().padStart(8, "0");
  } else if (currentStage >= 8) {
    statusVal = "0x00040002";
  }

  const rows = [
    { off: "0x00", name: "CTRL_REG", val: ctrlVal, hi: currentStage === 4 },
    { off: "0x04", name: "STATUS_REG", val: statusVal, hi: currentStage >= 5 && currentStage <= 8 },
    { off: "0x10", name: "MAT_A_ROW0", val: currentStage >= 2 ? packRowHex(matA[0]) : "0x00000000", hi: currentStage === 2 },
    { off: "0x14", name: "MAT_A_ROW1", val: currentStage >= 2 ? packRowHex(matA[1]) : "0x00000000", hi: currentStage === 2 },
    { off: "0x20", name: "MAT_B_ROW0", val: currentStage >= 3 ? packRowHex(matB[0]) : "0x00000000", hi: currentStage === 3 },
    { off: "0x24", name: "MAT_B_ROW1", val: currentStage >= 3 ? packRowHex(matB[1]) : "0x00000000", hi: currentStage === 3 },
    { off: "0x30", name: "MAT_C[0,0]", val: currentStage >= 8 ? "0x" + (peAccum[0][0] >>> 0).toString(16).toUpperCase().padStart(8, "0") : "0x00000000", hi: currentStage === 9 },
    { off: "0x6C", name: "MAT_C[3,3]", val: currentStage >= 8 ? "0x" + (peAccum[3][3] >>> 0).toString(16).toUpperCase().padStart(8, "0") : "0x00000000", hi: currentStage === 9 }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr class="${r.hi ? 'bg-emerald-950/60 text-emerald-300 font-bold' : 'text-slate-300'}">
      <td class="py-1.5 px-2.5 text-slate-400">${r.off}</td>
      <td class="py-1.5 px-2.5">${r.name}</td>
      <td class="py-1.5 px-2.5 text-right">${r.val}</td>
    </tr>
  `).join("");
}

function renderReLUOutput() {
  const reluGrid = document.getElementById("reluGrid");
  const badge = document.getElementById("predBadge");
  reluGrid.innerHTML = "";

  let colSums = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const raw = peAccum[r][c] + biasVec[c];
      const reluVal = currentStage === 10 ? Math.max(0, raw) : 0;
      colSums[c] += reluVal;
      const cell = document.createElement("div");
      cell.className = `py-1.5 rounded border ${
        currentStage === 10
          ? (reluVal > 0 ? "bg-indigo-600/30 border-indigo-400 text-white font-bold" : "bg-slate-900 border-slate-800 text-slate-500")
          : "bg-slate-950 border-slate-800 text-slate-600"
      }`;
      cell.textContent = currentStage === 10 ? reluVal : "-";
      reluGrid.appendChild(cell);
    }
  }

  if (currentStage === 10) {
    let bestClass = 0;
    for (let j = 1; j < 4; j++) {
      if (colSums[j] > colSums[bestClass]) bestClass = j;
    }
    badge.textContent = `Class #${bestClass} (Score: ${colSums[bestClass]})`;
    badge.className = "text-[11px] mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold";
  } else {
    badge.textContent = "Awaiting Stage 10";
    badge.className = "text-[11px] mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-200";
  }
}

function updateUIForStage() {
  const st = STAGES[currentStage];
  document.getElementById("stageBadge").textContent = `${st.id} / 10 : ${st.short.toUpperCase()}`;
  document.getElementById("stageExplain").textContent = st.desc;
  document.getElementById("hwCyclesDisplay").textContent = `${st.hwCycles} / 68 cycles`;

  document.querySelectorAll(".c-line").forEach(el => {
    const lineNum = parseInt(el.getAttribute("data-cline"), 10);
    if (lineNum === st.cline && currentStage > 0) {
      el.className = "c-line px-2 py-1 rounded bg-indigo-600/40 border border-indigo-400 text-white font-semibold";
    } else {
      el.className = "c-line px-2 py-1 rounded text-slate-400";
    }
  });

  const busBox = document.getElementById("axiBusBox");
  const busLabel = document.getElementById("axiStateLabel");
  const busText = document.getElementById("axiPacketText");

  if ([2, 3, 4, 9].includes(currentStage)) {
    busBox.classList.add("bus-pulse", "border-emerald-500");
    if (currentStage === 2) {
      busLabel.textContent = "AXI WRITE BURST (MAT_A)";
      busText.textContent = `AWADDR=0x40020010 WDATA=${packRowHex(matA[0])} [OKAY]`;
    } else if (currentStage === 3) {
      busLabel.textContent = "AXI WRITE BURST (MAT_B)";
      busText.textContent = `AWADDR=0x40020020 WDATA=${packRowHex(matB[0])} [OKAY]`;
    } else if (currentStage === 4) {
      busLabel.textContent = "AXI WRITE CTRL_REG";
      busText.textContent = "AWADDR=0x40020000 WDATA=0x00000001 (START=1)";
    } else if (currentStage === 9) {
      busLabel.textContent = "AXI READ BURST (MAT_C)";
      busText.textContent = `ARADDR=0x40020030 RDATA=${peAccum[0][0]} (16 words)`;
    }
  } else {
    busBox.classList.remove("bus-pulse", "border-emerald-500");
    busLabel.textContent = currentStage >= 5 && currentStage <= 8 ? "POLLING STATUS (0x04)" : "IDLE";
    busText.textContent = currentStage >= 5 && currentStage <= 8
      ? `ARADDR=0x40020004 BUSY=${currentStage < 8 ? 1 : 0} DONE=${currentStage === 8 ? 1 : 0}`
      : "No active AXI bus transaction";
  }

  const activeK = st.k !== undefined ? st.k : null;
  document.getElementById("peStepIndicator").textContent = activeK !== null
    ? `k_step = ${activeK} (16 PEs Active)`
    : (currentStage >= 8 ? "DONE (4 Cycles)" : "k_step: IDLE");

  renderPipelineBar();
  renderSystolicGrid(activeK);
  renderRegisterTable();
  renderReLUOutput();
}

function stepSimulation() {
  if (currentStage >= 10) {
    resetSimulation();
    return;
  }
  currentStage++;
  const st = STAGES[currentStage];
  if (st.k !== undefined) {
    const k = st.k;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        peAccum[r][c] += matA[r][k] * matB[k][c];
      }
    }
  }
  logEvent(st.desc, currentStage === 8 || currentStage === 10);
  updateUIForStage();
}

function resetSimulation() {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer);
    autoPlayTimer = null;
    document.getElementById("btnAutoPlay").textContent = "Auto-Run Simulation";
  }
  currentStage = 0;
  peAccum = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
  document.getElementById("simConsole").innerHTML = "";
  logEvent("Simulation reset. Ready to offload 4x4 matrix multiplication from SHAKTI to FPGA.");
  updateUIForStage();
}

function toggleAutoPlay() {
  const btn = document.getElementById("btnAutoPlay");
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer);
    autoPlayTimer = null;
    btn.textContent = "Auto-Run Simulation";
    return;
  }
  if (currentStage >= 10) resetSimulation();
  btn.textContent = "Pause Auto-Run";
  autoPlayTimer = setInterval(() => {
    if (currentStage < 10) {
      stepSimulation();
    } else {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      btn.textContent = "Auto-Run Simulation";
    }
  }, 950);
}

function initBenchmarkChart() {
  const ctx = document.getElementById("benchmarkChart");
  if (!ctx || benchmarkChartInstance) return;
  benchmarkChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["4x4 Matrix Layer", "8x8 Matrix (4 Tiles)", "16x16 Matrix (16 Tiles)"],
      datasets: [
        {
          label: "SHAKTI Software-Only (Cycles)",
          data: [612, 4752, 37440],
          backgroundColor: "rgba(244, 63, 94, 0.8)",
          borderRadius: 6
        },
        {
          label: "SHAKTI + FPGA Co-Design (Cycles)",
          data: [68, 272, 1088],
          backgroundColor: "rgba(16, 185, 129, 0.85)",
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          type: "logarithmic",
          grid: { color: "rgba(148, 163, 184, 0.15)" },
          ticks: { color: "#cbd5e1" },
          title: { display: true, text: "Clock Cycles (Log Scale)", color: "#94a3b8" }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#cbd5e1" }
        }
      },
      plugins: {
        legend: { labels: { color: "#e2e8f0" } }
      }
    }
  });
}

// Attach Event Listeners on DOM Load
window.addEventListener("DOMContentLoaded", () => {
  renderMatrices();
  resetSimulation();

  document.getElementById("btnStep").addEventListener("click", stepSimulation);
  document.getElementById("btnAutoPlay").addEventListener("click", toggleAutoPlay);
  document.getElementById("btnReset").addEventListener("click", resetSimulation);

  document.getElementById("presetSelect").addEventListener("change", (e) => {
    const val = e.target.value;
    if (val === "random") {
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          matA[r][c] = Math.floor(Math.random() * 7) - 2;
          matB[r][c] = Math.floor(Math.random() * 7) - 3;
        }
      }
    } else if (PRESETS[val]) {
      matA = JSON.parse(JSON.stringify(PRESETS[val].A));
      matB = JSON.parse(JSON.stringify(PRESETS[val].B));
      biasVec = [...PRESETS[val].bias];
    }
    renderMatrices();
    resetSimulation();
  });

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      ["simView", "benchView", "codeView"].forEach(id => {
        document.getElementById(id).classList.toggle("hidden", id !== target);
      });
      document.querySelectorAll(".tab-btn").forEach(b => {
        const active = b === btn;
        b.className = `tab-btn px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold transition ${
          active ? "bg-sky-600 text-white" : "text-slate-400 hover:text-white"
        }`;
      });
      if (target === "benchView") initBenchmarkChart();
    });
  });

  document.querySelectorAll(".code-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-code");
      document.querySelectorAll(".code-panel").forEach(p => {
        p.classList.toggle("hidden", p.id !== target);
      });
      document.querySelectorAll(".code-tab-btn").forEach(b => {
        const active = b === btn;
        b.className = `code-tab-btn px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
          active ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300 hover:text-white"
        }`;
      });
    });
  });

  document.getElementById("btnCopyCode").addEventListener("click", () => {
    const visiblePre = Array.from(document.querySelectorAll(".code-panel")).find(p => !p.classList.contains("hidden"));
    if (!visiblePre) return;
    const temp = document.createElement("textarea");
    temp.value = visiblePre.textContent;
    document.body.appendChild(temp);
    temp.select();
    document.execCommand("copy");
    document.body.removeChild(temp);
    const btn = document.getElementById("btnCopyCode");
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = "Copy Active Code"; }, 1500);
  });
});
