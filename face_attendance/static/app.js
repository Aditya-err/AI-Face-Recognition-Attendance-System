/* Shared UI helpers for the attendance app. */
"use strict";

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

function esc(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function timeStr(iso) {
  return (iso || "").slice(11, 19); // HH:MM:SS part of YYYY-MM-DDTHH:MM:SS
}

function dateStr(iso) {
  return (iso || "").slice(0, 10);
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  let data = null;
  try { data = await res.json(); } catch (_e) { /* non-JSON body */ }
  if (!res.ok) {
    const message = (data && data.error) ||
      `Request failed (${res.status})`;
    const err = new Error(message);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

/* ------------------------------ toasts ------------------------------ */
function toast(message, kind) {
  kind = kind || "info";
  let box = $("#toasts");
  if (!box) {
    box = document.createElement("div");
    box.id = "toasts";
    document.body.appendChild(box);
  }
  const el = document.createElement("div");
  el.className = "toast " + (kind === "success" ? "ok"
    : kind === "error" ? "err" : "info");
  el.textContent = message;
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s";
    setTimeout(() => el.remove(), 320); }, 3400);
}

/* ------------------------------ nav status ------------------------------ */
function badgeForKind(kind) {
  const map = { recognized: ["green", "Recognized"], attendance: ["info", "Attendance"],
    uncertain: ["amber", "Uncertain"], unknown: ["red", "Unknown"] };
  const [cls, text] = map[kind] || ["gray", kind];
  return `<span class="badge ${cls}">${text}</span>`;
}

function dotClassForCamera(state) {
  if (state === "running") return "green";
  if (state === "error") return "red";
  return "amber";
}

async function refreshNavStatus() {
  try {
    const data = await fetchJSON("/api/camera/status");
    const el = $("#nav-cam");
    if (el) {
      el.innerHTML =
        `<span class="dot ${dotClassForCamera(data.state)}"></span>Camera: ${esc(data.state)}`;
    }
    const me = $("#nav-models");
    if (me) {
      me.innerHTML = `<span class="dot ${data.models_available ? "green" : "red"}"></span>` +
        `Face models: ${data.models_available ? "ready" : "missing"}`;
    }
  } catch (_e) { /* server briefly unavailable - ignore */ }
}
