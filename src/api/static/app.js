/**
 * Hinglish Intent Classifier Web Application
 * Frontend JavaScript Logic with Stats Animation, Batch Demo, Confidence Fallback
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements — Single Classifier
  const utteranceInput = document.getElementById("utterance-input");
  const charCounter = document.getElementById("char-counter");
  const clearInputBtn = document.getElementById("clear-input-btn");
  const classifyBtn = document.getElementById("classify-btn");
  const samplePillsContainer = document.getElementById("sample-pills-container");
  const systemStatusText = document.getElementById("system-status-text");
  const latencyText = document.getElementById("latency-text");
  const topIntentName = document.getElementById("top-intent-name");
  const intentAction = document.getElementById("intent-action");
  const circleBar = document.getElementById("circle-bar");
  const confidencePct = document.getElementById("confidence-pct");
  const distBarsContainer = document.getElementById("dist-bars-container");
  const diffCleaned = document.getElementById("diff-cleaned");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");
  const fallbackBanner = document.getElementById("fallback-banner");
  const fallbackSecondaryIntent = document.getElementById("fallback-secondary-intent");
  const secondaryIntentBadge = document.getElementById("secondary-intent-badge");
  const secondaryIntentName = document.getElementById("secondary-intent-name");
  const secondaryIntentConf = document.getElementById("secondary-intent-conf");

  // DOM Elements — Batch Demo
  const batchDemoBtn = document.getElementById("batch-demo-btn");
  const batchGrid = document.getElementById("batch-grid");

  // Intent Action Protocols mapping for voice agents
  const INTENT_ACTIONS = {
    price_negotiation: "🎯 <strong>Voice Agent Trigger:</strong> Offer standard tier discount coupon or initiate pricing rebuttal protocol.",
    complaint: "⚠️ <strong>Voice Agent Trigger:</strong> Escalate immediately to Tier-2 Support Lead and initiate ticket resolution workflow.",
    purchase_inquiry: "🔍 <strong>Voice Agent Trigger:</strong> Send product specifications brochure and offer complimentary live product demo.",
    callback_request: "📞 <strong>Voice Agent Trigger:</strong> Record preferred callback timestamp and queue automated dialer retry in CRM.",
    not_interested: "🚫 <strong>Voice Agent Trigger:</strong> Tag as cold lead in sales pipeline, log DND preferences, and end call politely.",
    positive_confirmation: "✅ <strong>Voice Agent Trigger:</strong> Trigger instant payment link via SMS/WhatsApp and mark lead as WON in CRM."
  };

  const INTENT_EMOJIS = {
    price_negotiation: "🏷️",
    complaint: "⚠️",
    purchase_inquiry: "🔍",
    callback_request: "📞",
    not_interested: "🚫",
    positive_confirmation: "✅"
  };

  // ==========================================================================
  // 1. Health Check
  // ==========================================================================
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

  // ==========================================================================
  // 2. Threshold Slider
  // ==========================================================================
  if (thresholdSlider && thresholdVal) {
    thresholdSlider.addEventListener("input", () => {
      thresholdVal.textContent = `${thresholdSlider.value}%`;
      runClassification();
    });
  }

  // ==========================================================================
  // 3. Character Counter
  // ==========================================================================
  function updateCharCount() {
    const len = utteranceInput.value.length;
    charCounter.textContent = `${len} / 250`;
  }
  utteranceInput.addEventListener("input", updateCharCount);
  updateCharCount();

  // ==========================================================================
  // 4. Clear Input
  // ==========================================================================
  clearInputBtn.addEventListener("click", () => {
    utteranceInput.value = "";
    updateCharCount();
    utteranceInput.focus();
  });

  // ==========================================================================
  // 5. Sample Pills
  // ==========================================================================
  samplePillsContainer.addEventListener("click", (e) => {
    const btn = e.target.closest(".pill-btn");
    if (!btn) return;
    document.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    utteranceInput.value = btn.getAttribute("data-text");
    updateCharCount();
    runClassification();
  });

  // ==========================================================================
  // 6. Single Classification
  // ==========================================================================
  async function runClassification() {
    const text = utteranceInput.value.trim();
    if (!text) return;

    const thresholdDecimal = thresholdSlider ? parseFloat(thresholdSlider.value) / 100.0 : 0.60;
    classifyBtn.classList.add("loading");
    const startTime = performance.now();

    try {
      const response = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, confidence_threshold: thresholdDecimal })
      });

      const elapsedMs = Math.round(performance.now() - startTime);
      latencyText.textContent = `~${elapsedMs}ms`;

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      renderResults(data);
    } catch (error) {
      console.error("Classification error:", error);
    } finally {
      classifyBtn.classList.remove("loading");
    }
  }

  classifyBtn.addEventListener("click", runClassification);
  utteranceInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runClassification();
    }
  });

  // ==========================================================================
  // 7. Render Single Results
  // ==========================================================================
  function renderResults(data) {
    const { intent, confidence, is_uncertain, fallback, secondary_intent, secondary_confidence, cleaned_text, all_scores } = data;

    topIntentName.textContent = intent.replace(/_/g, " ");
    intentAction.innerHTML = INTENT_ACTIONS[intent] || "🎯 <strong>Voice Agent Action:</strong> Route utterance to conversational sales agent.";

    const confPercent = Math.round(confidence * 100);
    confidencePct.textContent = `${confPercent}%`;
    circleBar.setAttribute("stroke-dasharray", `${confPercent}, 100`);

    if (fallback && fallbackBanner) {
      fallbackBanner.style.display = "flex";
      if (fallbackSecondaryIntent && secondary_intent) {
        fallbackSecondaryIntent.textContent = `${secondary_intent.replace(/_/g, " ")} (${Math.round((secondary_confidence || 0) * 100)}%)`;
      }
    } else if (fallbackBanner) {
      fallbackBanner.style.display = "none";
    }

    if (secondary_intent && secondaryIntentBadge) {
      secondaryIntentBadge.style.display = "inline-flex";
      secondaryIntentName.textContent = secondary_intent.replace(/_/g, " ");
      secondaryIntentConf.textContent = `${Math.round((secondary_confidence || 0) * 100)}%`;
    } else if (secondaryIntentBadge) {
      secondaryIntentBadge.style.display = "none";
    }

    diffCleaned.textContent = cleaned_text || utteranceInput.value;

    distBarsContainer.innerHTML = "";
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

  // ==========================================================================
  // 8. Stats Counter Animation
  // ==========================================================================
  function animateCounters() {
    const statNumbers = document.querySelectorAll(".stat-number");
    statNumbers.forEach((el, i) => {
      const target = parseFloat(el.dataset.target);
      const suffix = el.dataset.suffix || "";
      const isDecimal = target % 1 !== 0;
      const duration = 1200;
      const startTime = performance.now();

      // Stagger reveal
      setTimeout(() => {
        el.closest(".stat-item").classList.add("visible");
      }, i * 100);

      function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // ease-out
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = eased * target;

        if (isDecimal) {
          el.textContent = current.toFixed(1) + suffix;
        } else {
          el.textContent = Math.round(current).toLocaleString() + suffix;
        }

        if (progress < 1) {
          requestAnimationFrame(tick);
        }
      }

      setTimeout(() => requestAnimationFrame(tick), i * 100);
    });
  }

  // Observe stats row
  const statsRow = document.getElementById("stats-row");
  if (statsRow) {
    let statsAnimated = false;
    const statsObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !statsAnimated) {
          statsAnimated = true;
          animateCounters();
        }
      });
    }, { threshold: 0.3 });
    statsObserver.observe(statsRow);
  }

  // ==========================================================================
  // 9. Batch Demo
  // ==========================================================================
  const BATCH_UTTERANCES = [
    "Thoda discount de do na, price bohot zyada lag raha hai",
    "Mera order 5 din se aaya nahi, kya scam hai ye?",
    "Is product ki warranty kitni hai aur kya features hain?",
    "Abhi meeting chal raha hai, kal shaam ko call karna",
    "Bilkul bhi interest nahi hai, please do not call again",
    "Haanji pakka confirm hai, payment link bhej dijiye abhi"
  ];

  // Show placeholder initially
  if (batchGrid) {
    batchGrid.innerHTML = `
      <div class="batch-placeholder">
        <span class="batch-placeholder-icon">🚀</span>
        Hit <strong>"Run Batch Demo"</strong> to classify 6 diverse utterances simultaneously
      </div>
    `;
  }

  if (batchDemoBtn) {
    batchDemoBtn.addEventListener("click", async () => {
      batchDemoBtn.classList.add("loading");
      batchGrid.innerHTML = "";

      try {
        const thresholdDecimal = thresholdSlider ? parseFloat(thresholdSlider.value) / 100.0 : 0.60;

        const response = await fetch("/classify/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            texts: BATCH_UTTERANCES,
            confidence_threshold: thresholdDecimal
          })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        renderBatchResults(data.results, BATCH_UTTERANCES);
      } catch (error) {
        console.error("Batch demo error:", error);
        batchGrid.innerHTML = `<div class="batch-placeholder">❌ Batch request failed. Make sure the API server is running.</div>`;
      } finally {
        batchDemoBtn.classList.remove("loading");
      }
    });
  }

  function renderBatchResults(results, utterances) {
    batchGrid.innerHTML = "";
    results.forEach((r, i) => {
      const card = document.createElement("div");
      const isFallback = r.fallback;
      card.className = `batch-card${isFallback ? " fallback-card" : ""}`;

      const confPct = Math.round(r.confidence * 100);
      const emoji = INTENT_EMOJIS[r.intent] || "🎯";
      const action = INTENT_ACTIONS[r.intent] || "";
      // Strip HTML tags for batch action display
      const actionText = action.replace(/<[^>]*>/g, "");

      card.innerHTML = `
        <div class="batch-card-header">
          <span class="batch-intent-tag${isFallback ? " uncertain" : ""}">${emoji} ${r.intent.replace(/_/g, " ")}</span>
          <span class="batch-confidence${confPct < 60 ? " low" : ""}">${confPct}%</span>
        </div>
        <div class="batch-utterance">"${utterances[i]}"</div>
        <div class="batch-action">${actionText}</div>
        ${r.secondary_intent ? `<div class="batch-secondary">↳ Secondary: ${r.secondary_intent.replace(/_/g, " ")} (${Math.round((r.secondary_confidence || 0) * 100)}%)</div>` : ""}
      `;

      batchGrid.appendChild(card);

      // Staggered reveal animation
      setTimeout(() => card.classList.add("revealed"), 150 * i);
    });
  }

  // ==========================================================================
  // Init
  // ==========================================================================
  checkHealth();
  runClassification();
});
