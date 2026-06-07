// AI 美甲试戴 v3 - Chatbot + Gallery 流程

const API_BASE = '';  // 同源或 proxy
const API_TRYON = '/api/tryon';
const API_RECOMMEND = '/api/recommend';
const API_RECOMMEND_TRYON = '/api/recommend_tryon';
const API_CHAT = '/api/chat';
const API_STYLES = '/api/styles';

const USE_MOCK = false;
const STEP_INTERVAL_MS = 2500;

// ==================== 状态 ====================
const state = {
  selectedStyleId: null,
  selectedFile: null,
  history: [],   // chat history: [{role, content}]
  styles: [],
};

// ==================== DOM ====================
const $ = (id) => document.getElementById(id);

const gallery = $('gallery');
const dropzone = $('dropzone');
const fileInput = $('fileInput');
const preview = $('preview');
const tryOnBtn = $('tryOnBtn');
const whitePreviewBtn = $('whitePreviewBtn');
const tryOnHint = $('tryOnHint');
const resetBtn = $('resetBtn');
const result = $('result');
const resultOriginal = $('resultOriginal');
const resultGenerated = $('resultGenerated');
const styleBadge = $('styleBadge');
const styleName = $('styleName');
const nailCount = $('nailCount');
const saveBtn = $('saveBtn');
const loader = $('loader');
const recommendBtn = $('recommendBtn');
const recommendReason = $('recommendReason');
const styleHint = $('styleHint');
const resultHint = $('resultHint');
const styleAdvice = $('styleAdvice');
const fitScore = $('fitScore');
const fitText = $('fitText');
const trendText = $('trendText');
const widthScale = $('widthScale');
const lengthScale = $('lengthScale');
const offsetScale = $('offsetScale');
const opacityScale = $('opacityScale');
const widthValue = $('widthValue');
const lengthValue = $('lengthValue');
const offsetValue = $('offsetValue');
const opacityValue = $('opacityValue');

const chatMessages = $('chatMessages');
const chatInput = $('chatInput');
const chatSend = $('chatSend');

const stepEls = Array.from(document.querySelectorAll('.loader__step'));

// ==================== 加载款式库 ====================
async function loadStyles() {
  try {
    const r = await fetch(API_STYLES);
    const data = await r.json();
    state.styles = data.styles || [];
    state.stylesTarget = data.target || state.styles.length;
    // 动态提示文本: "14 款中选 1 款 (评估 25 款 中前 14)"
    if (state.stylesTarget > state.styles.length) {
      styleHint.textContent = `${state.styles.length} 款中选 1 款 (目标 ${state.stylesTarget} 款)`;
    } else {
      styleHint.textContent = `${state.styles.length} 款中选 1 款`;
    }
    renderGallery();
  } catch (e) {
    console.error('loadStyles failed', e);
    gallery.innerHTML = '<div style="grid-column:1/-1; color: var(--ink-faint); text-align:center; padding: 20px;">加载款式失败</div>';
  }
}

function renderGallery() {
  gallery.innerHTML = '';
  state.styles.forEach(style => {
    const card = document.createElement('div');
    card.className = 'style-card';
    // 没有真实资产时才标记为补充色卡。
    if (!style.image && style.id > 14) {
      card.classList.add('style-card--supplement');
      card.title = '补充色卡 (原评估图未提供)';
    }
    card.dataset.styleId = style.id;
    card.style.setProperty('--swatch-color', style.color_hex);
    card.innerHTML = `
      <span class="style-card__num">#${style.id}</span>
      <div class="style-card__swatch" ${style.image ? `style="background-image:url('${style.image}')"` : ''}></div>
      <div class="style-card__name">${style.name}</div>
    `;
    card.addEventListener('click', () => selectStyle(style.id));
    gallery.appendChild(card);
  });
}

function selectStyle(id) {
  state.selectedStyleId = id;
  document.querySelectorAll('.style-card').forEach(el => {
    el.classList.toggle('is-selected', parseInt(el.dataset.styleId) === id);
  });
  const style = state.styles.find(s => s.id === id);
  if (style) {
    styleHint.textContent = `已选 #${id} ${style.name}`;
  }
  updateTryOnButton();
}

// ==================== 上传 ====================
dropzone.addEventListener('click', (e) => {
  if (dropzone.classList.contains('has-image')) return;
  fileInput.click();
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('请上传图片文件');
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    alert('文件超过 5MB');
    return;
  }
  state.selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  dropzone.classList.add('has-image');
  updateTryOnButton();
}

function updateTryOnButton() {
  const hasStyle = state.selectedStyleId != null;
  const hasFile = state.selectedFile != null;
  tryOnBtn.disabled = !(hasStyle && hasFile);
  whitePreviewBtn.disabled = !(hasStyle && hasFile);
  if (!hasStyle && !hasFile) {
    tryOnHint.textContent = '选款式 + 上传照片';
  } else if (!hasStyle) {
    tryOnHint.textContent = '请选款式';
  } else if (!hasFile) {
    tryOnHint.textContent = '请上传照片';
  } else {
    const style = state.styles.find(s => s.id === state.selectedStyleId);
    tryOnHint.textContent = `将用 #${state.selectedStyleId} ${style.name} 渲染`;
  }
}

// ==================== 试戴 ====================
tryOnBtn.addEventListener('click', async () => {
  if (!state.selectedFile || !state.selectedStyleId) return;
  await runTryOn(false);
});

whitePreviewBtn.addEventListener('click', async () => {
  if (!state.selectedFile || !state.selectedStyleId) return;
  await runTryOn(true);
});

async function runTryOn(whiteMode) {
  result.hidden = true;
  loader.hidden = false;
  tryOnBtn.disabled = true;
  whitePreviewBtn.disabled = true;
  resetSteps();
  const advancer = startStepAdvancer();

  try {
    const data = USE_MOCK
      ? await mockTryOn()
      : await callRealAPI(state.selectedFile, state.selectedStyleId, whiteMode);

    clearInterval(advancer);
    completeAllSteps();
    await sleep(400);
    loader.hidden = true;
    showResult(data);
  } catch (err) {
    clearInterval(advancer);
    loader.hidden = true;
    handleError(err);
  } finally {
    tryOnBtn.disabled = false;
    whitePreviewBtn.disabled = false;
  }
}

async function callRealAPI(file, styleId, whiteMode = false) {
  const form = new FormData();
  form.append('hand_image', file);
  form.append('style_id', String(styleId));
  appendTuning(form);
  if (whiteMode) form.append('white_mode', '1');
  const r = await fetch(API_TRYON, { method: 'POST', body: form });
  if (!r.ok) {
    throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  }
  return await r.json();
}

function showResult(data) {
  resultOriginal.src = preview.src;
  resultGenerated.src = data.result_image;
  styleBadge.textContent = data.style_id;
  const style = state.styles.find(s => s.id === data.style_id) || data.style;
  styleName.textContent = style ? style.name : `#${data.style_id}`;
  nailCount.textContent = `检测指甲 ${data.nails_detected} 个`;
  resultHint.textContent = data.debug_white_mode ? '白模模式' : '已渲染';
  if (data.fit_text || data.trend_text || data.fit_score) {
    fitScore.textContent = data.fit_score ? `${data.fit_score}/100` : '-';
    fitText.textContent = data.fit_text || '';
    trendText.textContent = data.trend_text || '';
    styleAdvice.hidden = false;
  } else {
    styleAdvice.hidden = true;
  }
  result.hidden = false;
  saveBtn.hidden = false;
  resetBtn.hidden = false;

  setTimeout(() => {
    result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

// ==================== 步骤进度 ====================
function resetSteps() {
  stepEls.forEach(el => el.classList.remove('is-active', 'is-done'));
}
function activateStep(idx) {
  stepEls.forEach((el, i) => {
    el.classList.remove('is-active');
    if (i < idx) el.classList.add('is-done');
    else el.classList.remove('is-done');
  });
  if (stepEls[idx]) stepEls[idx].classList.add('is-active');
}
function completeAllSteps() {
  stepEls.forEach(el => {
    el.classList.remove('is-active');
    el.classList.add('is-done');
  });
}
function startStepAdvancer() {
  let i = 0;
  activateStep(0);
  return setInterval(() => {
    i++;
    if (i >= stepEls.length) return;
    activateStep(i);
  }, STEP_INTERVAL_MS);
}

// ==================== 重置 ====================
resetBtn.addEventListener('click', () => {
  state.selectedFile = null;
  preview.src = '';
  dropzone.classList.remove('has-image');
  fileInput.value = '';
  updateTryOnButton();
  resetBtn.hidden = true;
  result.hidden = true;
  saveBtn.hidden = true;
});

// ==================== 保存结果图 ====================
saveBtn.addEventListener('click', () => {
  const img = resultGenerated;
  if (!img || !img.src) return;
  const a = document.createElement('a');
  a.href = img.src;
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.download = `nail-tryon-${ts}.jpg`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// ==================== 自动推荐 ====================
const aiAnalysis = $('aiAnalysis');
const analysisClose = $('analysisClose');

recommendBtn.addEventListener('click', async () => {
  if (!state.selectedFile) {
    alert('请先上传手部照片');
    return;
  }
  recommendBtn.disabled = true;
  recommendBtn.textContent = '分析中…';
  try {
    const form = new FormData();
    form.append('hand_image', state.selectedFile);
    appendTuning(form);
    const r = await fetch(API_RECOMMEND_TRYON, { method: 'POST', body: form });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderAnalysis(data);
    if (data.style_id) selectStyle(data.style_id);
    if (data.result_image) showResult(data);
  } catch (e) {
    recommendReason.textContent = '分析失败';
    console.error(e);
  } finally {
    recommendBtn.disabled = false;
    recommendBtn.textContent = '不知道选哪款？让 AI 分析推荐';
  }
});

function renderAnalysis(data) {
  // analysis
  const a = data.analysis || {};
  $('tagSkin').textContent = a.skin_tone || '-';
  $('tagHand').textContent = a.hand_type || '-';
  $('tagNail').textContent = a.nail_status || '-';
  $('tagScene').textContent = a.scene || '-';
  
  // 3 款推荐
  const recs = data.recommendations || [];
  const list = $('recommendList');
  list.innerHTML = '';
  recs.forEach((r, idx) => {
    const style = r.style || state.styles.find(s => s.id === r.style_id);
    if (!style) return;
    const item = document.createElement('div');
    item.className = 'ai-rec-item';
    item.innerHTML = `
      <div class="ai-rec-item__swatch" style="background:${style.color_hex}"></div>
      <div class="ai-rec-item__num">#${style.id}</div>
      <div class="ai-rec-item__main">
        <div class="ai-rec-item__name">${style.name}</div>
        <div class="ai-rec-item__reason">${r.reason}</div>
      </div>
      <div class="ai-rec-item__pick">${idx === 0 ? '首选' : '备选'}</div>
    `;
    item.addEventListener('click', () => {
      selectStyle(style.id);
      aiAnalysis.hidden = true;
    });
    list.appendChild(item);
  });
  
  aiAnalysis.hidden = false;
  recommendReason.textContent = '';
}

analysisClose.addEventListener('click', () => {
  aiAnalysis.hidden = true;
});

// ==================== Chatbot ====================
chatInput.addEventListener('input', () => {
  chatSend.disabled = chatInput.value.trim().length === 0;
});
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !chatSend.disabled) {
    sendChat();
  }
});
chatSend.addEventListener('click', sendChat);

async function sendChat() {
  const text = chatInput.value.trim();
  if (!text) return;
  
  // 渲染用户消息
  appendMessage('user', text);
  chatInput.value = '';
  chatSend.disabled = true;
  state.history.push({ role: 'user', content: text });

  // 渲染"思考中"
  const thinkingMsg = appendMessage('bot', '思考中…');

  try {
    let handB64 = null;
    if (state.selectedFile) {
      handB64 = await fileToDataURL(state.selectedFile);
    }
    
    const r = await fetch(API_CHAT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        style_id: state.selectedStyleId || 0,
        history: state.history.slice(-6),
        hand_image: handB64,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    
    // 替换"思考中"为实际回复
    thinkingMsg.querySelector('.msg__bubble').textContent = data.reply;
    state.history.push({ role: 'assistant', content: data.reply });
  } catch (e) {
    thinkingMsg.querySelector('.msg__bubble').textContent = '抱歉，服务暂时不可用';
    console.error(e);
  }
}

function appendMessage(role, text) {
  const msg = document.createElement('div');
  msg.className = `msg msg--${role === 'user' ? 'user' : 'bot'}`;
  msg.innerHTML = `<div class="msg__bubble">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msg;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// ==================== 工具 ====================
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
function fileToDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

function appendTuning(form) {
  form.append('width_scale', widthScale.value);
  form.append('length_scale', lengthScale.value);
  form.append('offset_scale', offsetScale.value);
  form.append('opacity', opacityScale.value);
}

function updateTuningLabels() {
  widthValue.textContent = `${Math.round(parseFloat(widthScale.value) * 100)}%`;
  lengthValue.textContent = `${Math.round(parseFloat(lengthScale.value) * 100)}%`;
  const offset = parseFloat(offsetScale.value);
  offsetValue.textContent = offset === 0 ? '0' : (offset < 0 ? '向指尖' : '向手心');
  opacityValue.textContent = `${Math.round(parseFloat(opacityScale.value) * 100)}%`;
}
[widthScale, lengthScale, offsetScale, opacityScale].forEach(input => {
  input.addEventListener('input', updateTuningLabels);
});
updateTuningLabels();
function handleError(err) {
  let msg = err.message || '未知错误';
  if (msg.includes('Failed to fetch')) msg = '无法连接到后端服务';
  else if (msg.includes('NetworkError')) msg = '网络错误';
  else if (msg.includes('500')) msg = '服务器内部错误';
  alert(msg);
  console.error(err);
}

// ==================== Mock ====================
async function mockTryOn() {
  await sleep(STEP_INTERVAL_MS);
  activateStep(1);
  await sleep(STEP_INTERVAL_MS);
  activateStep(2);
  await sleep(800);
  const dataUrl = await fileToDataURL(state.selectedFile);
  return {
    result_image: dataUrl,
    style_id: state.selectedStyleId,
    nails_detected: 5,
  };
}

// ==================== Init ====================
loadStyles();
