// Extension "Achat Éco-Responsable" — popup
//
// Si tu changes de domaine (ex: tu passes sur Render, ou tu ajoutes un
// domaine personnalisé), mets à jour cette constante ET la liste
// "host_permissions" dans manifest.json (les deux doivent correspondre).
const API_BASE = "https://web-production-671e46.up.railway.app";

const currentUrlEl = document.getElementById("currentUrl");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const scoreBadgeEl = document.getElementById("scoreBadge");
const scoreNumEl = document.getElementById("scoreNum");
const scoreTitleEl = document.getElementById("scoreTitle");
const fMaterialsEl = document.getElementById("fMaterials");
const fCo2eEl = document.getElementById("fCo2e");
const fRecycEl = document.getElementById("fRecyc");
const compareLinkEl = document.getElementById("compareLink");

let currentTabUrl = "";

async function getActiveTabUrl() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return (tab && tab.url) || "";
}

function scoreClass(score) {
  if (score >= 60) return "high";
  if (score >= 40) return "mid";
  return "low";
}

(async function init() {
  currentTabUrl = await getActiveTabUrl();
  currentUrlEl.textContent = currentTabUrl || "(page inconnue)";
  compareLinkEl.href = `${API_BASE}/?url=${encodeURIComponent(currentTabUrl)}`;

  const estUnePageWeb = currentTabUrl.startsWith("http://") || currentTabUrl.startsWith("https://");
  if (!estUnePageWeb) {
    analyzeBtn.disabled = true;
    statusEl.textContent = "Ouvre une page produit d'un site web pour l'analyser.";
  }
})();

analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  resultEl.style.display = "none";
  statusEl.textContent = "Analyse en cours… (peut prendre 10-20 secondes) ⏳";

  try {
    const res = await fetch(`${API_BASE}/api/analyze-one`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentTabUrl }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Erreur inconnue");

    const f = data.features || {};
    scoreNumEl.textContent = data.eco_score;
    scoreTitleEl.textContent = data.title || "";
    scoreBadgeEl.className = "score-badge " + scoreClass(data.eco_score);
    fMaterialsEl.textContent = f.materials || "—";
    fCo2eEl.textContent = (f.co2e_kg != null) ? `${f.co2e_kg} kg` : "—";
    fRecycEl.textContent = f.recyclability || "inconnu";

    resultEl.style.display = "block";
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Erreur : " + err.message;
  } finally {
    analyzeBtn.disabled = false;
  }
});
