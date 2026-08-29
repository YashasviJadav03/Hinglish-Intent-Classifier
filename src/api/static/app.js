/**
 * Hinglish Intent Classifier Web Application
 * Frontend JavaScript Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const utteranceInput = document.getElementById("utterance-input");
  const charCounter = document.getElementById("char-counter");
  const clearInputBtn = document.getElementById("clear-input-btn");
  const classifyBtn = document.getElementById("classify-btn");
  const samplePillsContainer = document.getElementById("sample-pills-container");
  const systemStatusText = document.getElementById("system-status-text");
  const latencyBadge = document.getElementById("latency-badge");
  const latencyText = document.getElementById("latency-text");
  const topIntentName = document.getElementById("top-intent-name");
  const intentAction = document.getElementById("intent-action");
  const circleBar = document.getElementById("circle-bar");
  const confidencePct = document.getElementById("confidence-pct");
  const distBarsContainer = document.getElementById("dist-bars-container");
  const diffCleaned = document.getElementById("diff-cleaned");

  // Intent Action Protocols mapping for voice agents
  const INTENT_ACTIONS = {
    price_negotiation: "🎯 <strong>Voice Agent Trigger:</strong> Offer standard tier discount coupon or initiate pricing rebuttal protocol.",
    complaint: "⚠️ <strong>Voice Agent Trigger:</strong> Escalate immediately to Tier-2 Support Lead and initiate ticket resolution workflow.",
    purchase_inquiry: "🔍 <strong>Voice Agent Trigger:</strong> Send product specifications brochure and offer complimentary live product demo.",
    callback_request: "📞 <strong>Voice Agent Trigger:</strong> Record preferred callback timestamp and queue automated dialer retry in CRM.",
    not_interested: "🚫 <strong>Voice Agent Trigger:</strong> Tag as cold lead in sales pipeline, log DND preferences, and end call politely.",
    positive_confirmation: "✅ <strong>Voice Agent Trigger:</strong> Trigger instant payment link via SMS/WhatsApp and mark lead as WON in CRM."
  };

  const ALL_INTENTS = [
    "price_negotiation",
    "complaint",
    "purchase_inquiry",
    "callback_request",
    "not_interested",
    "positive_confirmation"
  ];

  // 1. Check API Health on Startup
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        systemStatusText.textContent = `Live LoRA Model Active (${data.device.toUpperCase()})`;
      } else {
        systemStatusText.textContent = "API Degraded";
      }
    } catch (err) {
      systemStatusText.textContent = "Connected (Local Mode)";
    }
  }

  // 2. Update Character Counter
  function updateCharCount() {
    const len = utteranceInput.value.length;
    charCounter.textContent = `${len} / 250`;
  }

  utteranceInput.addEventListener("input", updateCharCount);
  updateCharCount();

  // 3. Clear Input
  clearInputBtn.addEventListener("click", () => {
    utteranceInput.value = "";
    updateCharCount();
    utteranceInput.focus();
  });

  // 4. Sample Pill Click Handling
  samplePillsContainer.addEventListener("click", (e) => {
    const btn = e.target.closest(".pill-btn");
    if (!btn) return;

    // Toggle active state
    document.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");

    const text = btn.getAttribute("data-text");
    utteranceInput.value = text;
    updateCharCount();

    // Trigger classification immediately
    runClassification();
  });

  // 5. Main Classification Handler
  async function runClassification() {
    const text = utteranceInput.value.trim();
    if (!text) {
      alert("Please enter a customer utterance to classify.");
      return;
    }

    // Start loading state & timer
    classifyBtn.classList.add("loading");
    const startTime = performance.now();

    try {
      const response = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });

      const elapsedMs = Math.round(performance.now() - startTime);
      latencyText.textContent = `~${elapsedMs}ms`;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);
    } catch (error) {
      console.error("Classification error:", error);
      alert("Failed to reach classification API. Please check your network or server status.");
    } finally {
      classifyBtn.classList.remove("loading");
    }
  }

  classifyBtn.addEventListener("click", runClassification);

  // Allow Ctrl+Enter or Cmd+Enter to classify
  utteranceInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runClassification();
    }
  });

  // 6. Render Results to UI
  function renderResults(data) {
    const { intent, confidence, cleaned_text, all_scores } = data;

    // Top Intent Name
    topIntentName.textContent = intent.replace(/_/g, " ");

    // Intent Action Recommendation
    intentAction.innerHTML = INTENT_ACTIONS[intent] || "🎯 <strong>Voice Agent Action:</strong> Route utterance to conversational sales agent.";

    // Circular Confidence Chart (0 to 100)
    const confPercent = Math.round(confidence * 100);
    confidencePct.textContent = `${confPercent}%`;
    circleBar.setAttribute("stroke-dasharray", `${confPercent}, 100`);

    // Cleaned Text Diff
    diffCleaned.textContent = cleaned_text || utteranceInput.value;

    // Distribution Bars
    distBarsContainer.innerHTML = "";

    // Sort scores descending
    const sortedIntents = Object.entries(all_scores).sort((a, b) => b[1] - a[1]);

    sortedIntents.forEach(([intentKey, score]) => {
      const isTop = intentKey === intent;
      const pct = (score * 100).toFixed(1);

      const row = document.createElement("div");
      row.className = `bar-row ${isTop ? "top" : ""}`;

      row.innerHTML = `
        <div class="bar-label-row">
          <span class="bar-intent-name">${intentKey.replace(/_/g, " ")}</span>
          <span class="bar-score-val">${pct}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%"></div>
        </div>
      `;

      distBarsContainer.appendChild(row);
    });
  }

  // Initial Run on Page Load
  checkHealth();
  runClassification();
});
