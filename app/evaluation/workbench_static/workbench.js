"use strict";

const state = { cases: [], selectedId: null, filter: "all", detail: null, busy: false };
const csrf = document.querySelector('meta[name="workbench-csrf"]').content;
const caseList = document.querySelector("#case-list");
const detailContent = document.querySelector("#detail-content");
const emptyState = document.querySelector("#empty-state");
const toast = document.querySelector("#toast");
const finalizeDialog = document.querySelector("#finalize-dialog");

document.querySelector("#refresh-button").addEventListener("click", loadWorkspace);
document.querySelectorAll(".filter").forEach((button) => {
  button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll(".filter").forEach((item) => {
      const active = item === button;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    renderCases();
  });
});
document.querySelector("#confirm-finalize").addEventListener("click", (event) => {
  event.preventDefault();
  finalizeSelected();
});

loadWorkspace();

async function loadWorkspace() {
  if (state.busy) return;
  setBusy(true);
  try {
    const [summary, casesPayload] = await Promise.all([api("/api/summary"), api("/api/cases")]);
    state.cases = casesPayload.cases;
    renderSummary(summary);
    renderCases();
    if (state.selectedId && state.cases.some((item) => item.case_id === state.selectedId)) {
      state.detail = await api(`/api/cases/${encodeURIComponent(state.selectedId)}`);
      renderDetail(state.detail);
    } else if (state.cases.length) {
      state.selectedId = state.cases[0].case_id;
      renderCases();
      state.detail = await api(`/api/cases/${encodeURIComponent(state.selectedId)}`);
      renderDetail(state.detail);
    } else {
      showEmpty("Локальных кейсов пока нет", "Создайте bounded export через CLI. Интерфейс не обращается к production БД.");
    }
  } catch (error) {
    showToast(readError(error), true);
    showEmpty("Не удалось открыть выборку", "Проверьте private root и dataset, затем обновите страницу.");
  } finally {
    setBusy(false);
  }
}

function renderSummary(summary) {
  const items = [
    ["Всего кейсов", summary.total_cases],
    ["Финализировано", summary.finalized_cases],
    ["Принято", summary.accepted_cases],
    ["Отклонено", summary.rejected_cases],
    ["На проверке", summary.pending_cases],
    ["Покрытие review", formatPercent(summary.human_review_coverage)],
  ];
  const grid = document.querySelector("#summary-grid");
  grid.replaceChildren(...items.map(([label, value]) => {
    const item = el("div", "summary-item");
    item.append(el("dt", "", label), el("dd", "", String(value)));
    return item;
  }));
}

function renderCases() {
  const visible = state.cases.filter(caseMatchesFilter);
  if (!visible.length) {
    caseList.replaceChildren(el("p", "empty-list", "В этом состоянии кейсов нет."));
    return;
  }
  caseList.replaceChildren(...visible.map((item) => {
    const button = el("button", "case-row");
    button.type = "button";
    button.classList.toggle("is-selected", item.case_id === state.selectedId);
    button.setAttribute("aria-current", item.case_id === state.selectedId ? "true" : "false");
    const top = el("span", "case-row-top");
    top.append(el("span", "case-name", item.case_id), status(item.review_state === "pending" ? item.integrity_state : item.review_state));
    const meta = [item.distributor_code || "unknown", formatDate(item.run_created_at)].join(" · ");
    button.append(top, el("p", "case-meta", meta));
    button.addEventListener("click", () => selectCase(item.case_id));
    return button;
  }));
}

function caseMatchesFilter(item) {
  if (state.filter === "all") return true;
  if (state.filter === "pending") return item.review_state === "pending" && item.integrity_state === "verified";
  if (state.filter === "blocked") return ["blocked", "invalid"].includes(item.integrity_state);
  if (state.filter === "finalized") return ["accepted", "rejected"].includes(item.review_state);
  return true;
}

async function selectCase(caseId, focus = true) {
  if (state.busy) return;
  state.selectedId = caseId;
  renderCases();
  setBusy(true);
  try {
    state.detail = await api(`/api/cases/${encodeURIComponent(caseId)}`);
    renderDetail(state.detail);
    if (focus) document.querySelector("#case-detail").focus();
  } catch (error) {
    showToast(readError(error), true);
    showEmpty("Кейс недоступен", "Сервер отказал в чтении. Raw evidence не было показано.");
  } finally {
    setBusy(false);
  }
}

function renderDetail(detail) {
  emptyState.hidden = true;
  detailContent.hidden = false;
  const header = el("header", "detail-header");
  const identity = el("div");
  identity.append(el("p", "eyebrow", `Кейс · MatchRun ${detail.match_run_id}`), el("h2", "", detail.case_id));
  header.append(identity, status(detail.review_state === "pending" ? detail.integrity_state : detail.review_state));
  if (detail.integrity_state !== "verified") {
    const denial = el("section", "denial-panel");
    denial.append(el("p", "eyebrow", "Запись запрещена"), el("h2", "", "Evidence не прошло проверку целостности"));
    denial.append(el("p", "", "Источник и результат не открыты. Восстановите bundle повторным append-only export под новым case id."));
    denial.append(list(detail.integrity_errors));
    detailContent.replaceChildren(header, denial);
    return;
  }

  const layout = el("div", "detail-layout");
  layout.append(renderEvidence(detail), renderReviewSurface(detail));
  detailContent.replaceChildren(header, layout);
}

function renderEvidence(detail) {
  const rail = el("aside", "evidence-rail");
  rail.setAttribute("aria-label", "Evidence и lineage");
  const inner = el("div", "evidence-rail-inner");
  inner.append(
    evidenceBlock("Целостность", [
      ["Состояние", "verified"],
      ["Eligible", detail.golden_review_eligible ? "yes" : "no"],
      ["Blockers", String(detail.blockers.length)],
    ]),
    evidenceBlock("Pipeline", Object.entries(detail.stages)),
    evidenceBlock("Artifacts", Object.entries(detail.artifact_hashes).map(([key, value]) => [key, value.slice(0, 12)])),
    evidenceBlock("Матрица", [
      ["Режим", detail.matrix_evidence.mode],
      ["Сверки", String(detail.matrix_evidence.matched_diagnostics.length)],
      ["Расхождения", String(detail.matrix_evidence.mismatches.length)],
    ]),
    evidenceBlock("Контекст", [
      ["Commit", detail.production_code_revision.slice(0, 12)],
      ["Model", detail.model_id],
      ["Категорий", String(detail.category_count)],
    ]),
  );
  if (detail.blockers.length) {
    const block = el("section", "evidence-block");
    block.append(el("p", "eyebrow", "Blockers"), list(detail.blockers));
    inner.append(block);
  }
  rail.append(inner);
  return rail;
}

function evidenceBlock(title, rows) {
  const block = el("section", "evidence-block");
  block.append(el("p", "eyebrow", title));
  const entries = el("ul", "evidence-list");
  rows.forEach(([key, value]) => {
    const row = el("li");
    row.append(el("span", "", humanize(key)), el("code", "", String(value ?? "—")));
    entries.append(row);
  });
  block.append(entries);
  return block;
}

function renderReviewSurface(detail) {
  const surface = el("div", "review-surface");
  const quote = detail.quote || {};
  surface.append(contentSection("Запрос", sourceBlock(detail.source)));
  surface.append(contentSection("Результат", quoteBlock(quote)));

  const facts = el("div", "fact-columns");
  facts.append(factBox("Инженерные проверки", quote.engineer_checks), factBox("Пробелы закупки", quote.procurement_gaps));
  facts.append(factBox("Допущения", quote.assumptions), factBox("Отклонения", quote.key_deviations));
  surface.append(contentSection("Проверяемые факты", facts));

  if (detail.validation.errors.length || detail.validation.warnings.length) {
    const validation = el("div", "fact-columns");
    validation.append(factBox("Ошибки", detail.validation.errors), factBox("Предупреждения", detail.validation.warnings));
    surface.append(contentSection("Validation", validation));
  }
  surface.append(reviewForm(detail));
  return surface;
}

function contentSection(title, content) {
  const section = el("section", "content-section");
  section.append(el("h3", "", title), content);
  return section;
}

function sourceBlock(source) {
  return el("div", "source-text", source?.source_text || "Источник не содержит текста.");
}

function quoteBlock(quote) {
  const block = el("div");
  const intro = el("div", "quote-intro");
  const copy = el("div");
  copy.append(el("h3", "", quote.title || "Без заголовка"), el("p", "", quote.client_summary || "Нет клиентского summary."));
  if (quote.coverage_summary) copy.append(el("p", "", quote.coverage_summary));
  const total = el("div", "quote-total");
  total.append(el("span", "eyebrow", "Итого"), el("strong", "", formatMoney(quote.total_price_value, quote.total_price_currency)));
  intro.append(copy, total);
  block.append(intro);
  const lines = el("div", "quote-lines");
  (quote.lines || []).forEach((line) => {
    const row = el("article", "quote-line");
    row.append(
      el("p", "part", line.part_number || line.line_id || "—"),
      el("p", "", line.item_name || line.role || "—"),
      el("p", "", `× ${line.quantity ?? "—"}`),
      el("p", "", formatMoney(line.line_total_value, line.line_total_currency)),
    );
    if (line.reason) row.append(el("p", "reason", line.reason));
    lines.append(row);
  });
  block.append(lines);
  return block;
}

function factBox(title, items) {
  const box = el("section", "fact-box");
  box.append(el("h3", "", title), list(items || [], "Нет записей"));
  return box;
}

function reviewForm(detail) {
  const section = el("section", "content-section review-form");
  section.append(el("p", "eyebrow", "Human review"), el("h3", "", "Решение инженера"));
  const finalized = ["accepted", "rejected"].includes(detail.review_state);
  if (finalized) {
    section.append(el("p", "", `Решение финализировано: ${statusLabel(detail.review_state)}. Дальнейшее редактирование закрыто.`));
  }
  const form = el("form");
  form.id = "review-form";
  const review = detail.review;
  const grid = el("div", "form-grid");
  grid.append(
    selectField("decision", "Решение", [
      ["pending", "Не принято"], ["accept", "Принять"], ["reject", "Отклонить"],
    ], review.decision, finalized, !detail.golden_review_eligible ? ["accept"] : []),
    inputField("reviewer_role", "Роль проверяющего", "text", review.reviewer_role, finalized, "Например, solution-engineer"),
    inputField("semantic_score", "Семантическая оценка", "number", review.semantic_score, finalized, "0.00–1.00", { min: "0", max: "1", step: "0.01" }),
    inputField("unsupported_material_claim_count", "Неподтверждённые утверждения", "number", review.unsupported_material_claim_count, finalized, "Только существенные утверждения", { min: "0", step: "1" }),
    inputField("business_weighted_loss", "Business-weighted loss", "number", review.business_weighted_loss, finalized, "0 означает отсутствие потери", { min: "0", step: "0.01" }),
    textareaField("critical_error_codes", "Критические коды", (review.critical_error_codes || []).join("\n"), finalized, "Один код на строку"),
  );
  form.append(grid);

  const criteriaTitle = el("h3", "", "Атомарные критерии");
  criteriaTitle.className = "field-label";
  form.append(criteriaTitle);
  const criteria = el("div", "criteria-list");
  review.atomic_criteria.forEach((criterion) => criteria.append(criterionField(criterion, finalized)));
  form.append(criteria);

  const actions = el("div", "action-bar");
  const saveState = el("p", "field-hint", finalized ? "Receipt создан. Поля заблокированы." : "Изменения остаются только в private bundle.");
  saveState.id = "save-state";
  const buttons = el("div", "action-buttons");
  const save = el("button", "button button-quiet", "Сохранить черновик");
  save.type = "submit";
  save.disabled = finalized;
  const finalize = el("button", "button", "Проверить и финализировать");
  finalize.type = "button";
  finalize.disabled = finalized;
  finalize.addEventListener("click", prepareFinalize);
  buttons.append(save, finalize);
  actions.append(saveState, buttons);
  form.append(actions);
  form.addEventListener("submit", saveReview);
  section.append(form);
  return section;
}

function criterionField(criterion, disabled) {
  const fieldset = el("fieldset", "criterion");
  fieldset.dataset.criterionId = criterion.criterion_id;
  fieldset.disabled = disabled;
  fieldset.append(el("legend", "", humanize(criterion.criterion_id)));
  const radios = el("div", "radio-row");
  [["pending", "Не решено"], ["pass", "Пройдено"], ["fail", "Не пройдено"], ["not_applicable", "Н/П"]].forEach(([value, label]) => {
    const choice = el("label", "radio-choice");
    const input = el("input");
    input.type = "radio";
    input.name = `criterion-${criterion.criterion_id}`;
    input.value = value;
    input.checked = criterion.status === value;
    choice.append(input, document.createTextNode(label));
    radios.append(choice);
  });
  const refs = el("textarea");
  refs.className = "criterion-evidence";
  refs.value = (criterion.evidence_refs || []).join("\n");
  refs.placeholder = "Evidence refs, по одной на строку";
  refs.setAttribute("aria-label", `Evidence refs: ${humanize(criterion.criterion_id)}`);
  fieldset.append(radios, refs);
  return fieldset;
}

async function saveReview(event) {
  event.preventDefault();
  if (state.busy) return;
  setBusy(true);
  try {
    const result = await api(`/api/cases/${encodeURIComponent(state.selectedId)}/review`, {
      method: "POST", body: reviewPayload(),
    });
    state.detail.review = result.review;
    document.querySelector("#save-state").textContent = `Сохранено ${new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}`;
    showToast("Черновик сохранён локально.");
  } catch (error) {
    showToast(readError(error), true);
  } finally {
    setBusy(false);
  }
}

async function prepareFinalize() {
  if (state.busy) return;
  setBusy(true);
  try {
    const result = await api(`/api/cases/${encodeURIComponent(state.selectedId)}/review`, {
      method: "POST", body: reviewPayload(),
    });
    state.detail.review = result.review;
    document.querySelector("#finalize-copy").textContent = `Кейс ${state.selectedId}. Решение: ${statusLabel(result.review.decision)}. После финализации редактирование будет закрыто.`;
    finalizeDialog.showModal();
  } catch (error) {
    showToast(readError(error), true);
  } finally {
    setBusy(false);
  }
}

async function finalizeSelected() {
  if (state.busy) return;
  setBusy(true);
  try {
    await api(`/api/cases/${encodeURIComponent(state.selectedId)}/finalize`, {
      method: "POST", body: { confirm_case_id: state.selectedId },
    });
    finalizeDialog.close();
    showToast("Решение финализировано, receipt создан.");
    setBusy(false);
    await loadWorkspace();
  } catch (error) {
    showToast(readError(error), true);
  } finally {
    setBusy(false);
  }
}

function reviewPayload() {
  const form = document.querySelector("#review-form");
  return {
    decision: form.elements.decision.value,
    reviewer_role: emptyToNull(form.elements.reviewer_role.value),
    semantic_score: numberOrNull(form.elements.semantic_score.value),
    unsupported_material_claim_count: integerOrNull(form.elements.unsupported_material_claim_count.value),
    critical_error_codes: lines(form.elements.critical_error_codes.value),
    business_weighted_loss: numberOrNull(form.elements.business_weighted_loss.value),
    atomic_criteria: [...form.querySelectorAll(".criterion")].map((field) => ({
      criterion_id: field.dataset.criterionId,
      status: field.querySelector("input:checked").value,
      evidence_refs: lines(field.querySelector(".criterion-evidence").value),
    })),
  };
}

function selectField(name, label, options, value, disabled, disabledValues = []) {
  const field = fieldShell(label);
  const select = el("select");
  select.name = name;
  select.disabled = disabled;
  options.forEach(([optionValue, copy]) => {
    const option = el("option", "", copy);
    option.value = optionValue;
    option.selected = optionValue === value;
    option.disabled = disabledValues.includes(optionValue);
    select.append(option);
  });
  field.querySelector("label").after(select);
  field.querySelector("label").htmlFor = name;
  select.id = name;
  return field;
}

function inputField(name, label, type, value, disabled, hint, attrs = {}) {
  const field = fieldShell(label, hint);
  const input = el("input");
  input.id = name;
  input.name = name;
  input.type = type;
  input.value = value ?? "";
  input.disabled = disabled;
  Object.entries(attrs).forEach(([key, attrValue]) => input.setAttribute(key, attrValue));
  field.querySelector("label").after(input);
  field.querySelector("label").htmlFor = name;
  return field;
}

function textareaField(name, label, value, disabled, hint) {
  const field = fieldShell(label, hint);
  field.classList.add("field-wide");
  const textarea = el("textarea");
  textarea.id = name;
  textarea.name = name;
  textarea.value = value;
  textarea.disabled = disabled;
  field.querySelector("label").after(textarea);
  field.querySelector("label").htmlFor = name;
  return field;
}

function fieldShell(label, hint) {
  const field = el("div", "field");
  const labelElement = el("label", "", label);
  field.append(labelElement);
  if (hint) field.append(el("span", "field-hint", hint));
  return field;
}

async function api(url, options = {}) {
  const request = { headers: { Accept: "application/json" }, ...options };
  if (options.body !== undefined) {
    request.headers = { ...request.headers, "Content-Type": "application/json", "X-Workbench-CSRF": csrf };
    request.body = JSON.stringify(options.body);
  }
  const response = await fetch(url, request);
  const payload = await response.json().catch(() => ({ error: "response.invalid_json" }));
  if (!response.ok) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    error.details = payload.details || [];
    throw error;
  }
  return payload;
}

function setBusy(value) {
  state.busy = value;
  document.body.setAttribute("aria-busy", String(value));
  document.querySelectorAll("button").forEach((button) => {
    if (value && !button.disabled) button.dataset.busyDisabled = "true";
    if (button.dataset.busyDisabled === "true") button.disabled = value;
    if (!value) delete button.dataset.busyDisabled;
  });
}

function showEmpty(title, copy) {
  detailContent.hidden = true;
  emptyState.hidden = false;
  emptyState.querySelector("h2").textContent = title;
  emptyState.querySelector("p:last-child").textContent = copy;
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("is-error", isError);
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 4200);
}

function status(value) {
  return el("span", `status status-${value}`, statusLabel(value));
}

function statusLabel(value) {
  return ({ verified: "Целостно", blocked: "Blocked", invalid: "Invalid", pending: "На проверке", accepted: "Принято", rejected: "Отклонено", accept: "Принять", reject: "Отклонить" })[value] || value;
}

function list(items, emptyCopy = "Нет записей") {
  const ul = el("ul", "plain-list");
  if (!items?.length) ul.append(el("li", "", emptyCopy));
  else items.forEach((item) => ul.append(el("li", "", String(item))));
  return ul;
}

function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

function humanize(value) { return String(value).replaceAll("_", " ").replaceAll("-", " "); }
function lines(value) { return String(value || "").split("\n").map((item) => item.trim()).filter(Boolean); }
function emptyToNull(value) { const cleaned = String(value || "").trim(); return cleaned || null; }
function numberOrNull(value) { return value === "" ? null : Number(value); }
function integerOrNull(value) { return value === "" ? null : Number.parseInt(value, 10); }
function formatPercent(value) { return `${Math.round(Number(value || 0) * 100)}%`; }
function formatDate(value) { return value ? new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "short", year: "numeric" }).format(new Date(value)) : "—"; }
function formatMoney(value, currency) { return value === null || value === undefined || value === "" ? "—" : `${value} ${currency || ""}`.trim(); }
function readError(error) { return [error.message, ...(error.details || [])].join(": "); }
