/**
 * DriveStream AI — Chat Application Logic
 * Handles user input, API calls, message rendering, and UI state.
 */

// ── State ───────────────────────────────────────────────────────────

let currentMode = 'ask';  // 'ask' | 'search' | 'explain'
let isLoading = false;

const API_BASE = '';  // Same origin

// ── DOM Refs ────────────────────────────────────────────────────────

const messagesEl = document.getElementById('messages');
const inputForm = document.getElementById('input-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const mainTitle = document.getElementById('main-title');
const modelBadge = document.getElementById('model-name');
const vectorCount = document.getElementById('vector-count');
const healthDot = document.querySelector('.health-dot');
const healthText = document.querySelector('.health-text');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');

// ── Initialize ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkHealth();
    setInterval(checkHealth, 10000); // Poll every 10 seconds
    autoResizeTextarea();
});


function setupEventListeners() {
    // Form submission
    inputForm.addEventListener('submit', handleSubmit);

    // Textarea auto-resize + enable/disable send button
    userInput.addEventListener('input', () => {
        autoResizeTextarea();
        sendBtn.disabled = !userInput.value.trim() || isLoading;
    });

    // Enter to submit (Shift+Enter for newline)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (userInput.value.trim() && !isLoading) {
                inputForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => switchMode(btn.dataset.mode));
    });

    // Sidebar toggle (mobile)
    sidebarToggle.addEventListener('click', toggleSidebar);
}


// ── Mode Switching ──────────────────────────────────────────────────

function switchMode(mode) {
    currentMode = mode;

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    const titles = {
        ask: 'Ask about DriveStream',
        search: 'Search Code',
        explain: 'Explain a Class',
    };
    const placeholders = {
        ask: 'Ask a question about DriveStream...',
        search: 'Search for code (e.g., "partition routing", "offset commit")...',
        explain: 'Enter a class name (e.g., EventStreamEngine)...',
    };

    mainTitle.textContent = titles[mode];
    userInput.placeholder = placeholders[mode];
    userInput.focus();
}


// ── Submit Handler ──────────────────────────────────────────────────

async function handleSubmit(e) {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text || isLoading) return;

    // Add user message
    addMessage('user', text);
    userInput.value = '';
    autoResizeTextarea();
    sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = addTypingIndicator();
    isLoading = true;

    try {
        let data;

        if (currentMode === 'ask') {
            data = await apiCall('/api/ask', { question: text });
            removeTypingIndicator(typingEl);
            addAssistantMessage(data.answer, data.sources);

        } else if (currentMode === 'search') {
            data = await apiCall('/api/search', { query: text });
            removeTypingIndicator(typingEl);
            addSearchResults(data.results);

        } else if (currentMode === 'explain') {
            data = await apiCall('/api/explain', { class_name: text });
            removeTypingIndicator(typingEl);
            addAssistantMessage(data.explanation, data.sources);
        }

    } catch (err) {
        removeTypingIndicator(typingEl);
        addErrorMessage(err.message || 'Something went wrong. Is the server running?');
    } finally {
        isLoading = false;
        sendBtn.disabled = !userInput.value.trim();
    }
}


// ── API Calls ───────────────────────────────────────────────────────

async function apiCall(endpoint, body) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    return response.json();
}


async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        if (data.llm_ready) {
            healthDot.classList.add('online');
            healthDot.classList.remove('error');
            healthText.textContent = `GPU Online · ${data.vectors} vectors`;
            modelBadge.textContent = data.model.split('/').pop();
            vectorCount.textContent = data.vectors;
        } else {
            healthDot.classList.add('error');
            healthDot.classList.remove('online');
            healthText.textContent = 'Admin is Busy at work or Gaming';
            modelBadge.textContent = 'GPU Offline';
        }
    } catch {
        healthDot.classList.add('error');
        healthDot.classList.remove('online');
        healthText.textContent = 'API Offline';
        modelBadge.textContent = 'Disconnected';
    }
}


// ── Message Rendering ───────────────────────────────────────────────

function addMessage(role, text) {
    const avatar = role === 'user' ? '👤' : '⚡';
    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;
    msgEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
        </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
}


function addAssistantMessage(text, sources) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const tags = sources.map(s => {
            const label = s.method
                ? `${s.class}.${s.method}()`
                : s.class || s.file.split('/').pop();
            const relevance = Math.round(s.relevance * 100);
            return `<span class="source-tag">${escapeHtml(label)} <span class="source-relevance">${relevance}%</span></span>`;
        }).join('');

        sourcesHtml = `
            <div class="sources">
                <div class="sources-header">Sources</div>
                ${tags}
            </div>
        `;
    }

    msgEl.innerHTML = `
        <div class="message-avatar">⚡</div>
        <div class="message-content">
            <div class="message-text">${formatMarkdown(text)}</div>
            ${sourcesHtml}
        </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
}


function addSearchResults(results) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';

    let resultsHtml = `<h3>Found ${results.length} result${results.length !== 1 ? 's' : ''}</h3>`;

    results.forEach((r, i) => {
        const relevance = Math.round(r.relevance * 100);
        const title = r.method
            ? `${r.class}.${r.method}()`
            : r.class || r.id;
        const file = r.file.split('/').pop();

        resultsHtml += `
            <div style="margin: 12px 0; padding: 12px; background: var(--bg-tertiary); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <strong style="color: var(--accent-cyan); font-family: var(--font-mono); font-size: 13px;">${escapeHtml(title)}</strong>
                    <span class="source-relevance">${relevance}% match</span>
                </div>
                <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 8px;">${escapeHtml(file)} · ${r.type}</div>
                <pre><code>${escapeHtml(r.content.substring(0, 300))}${r.content.length > 300 ? '...' : ''}</code></pre>
            </div>
        `;
    });

    msgEl.innerHTML = `
        <div class="message-avatar">🔍</div>
        <div class="message-content">
            <div class="message-text">${resultsHtml}</div>
        </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
}


function addErrorMessage(text) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.innerHTML = `
        <div class="message-avatar">⚠️</div>
        <div class="message-content">
            <div class="message-text">
                <div class="error-text">${escapeHtml(text)}</div>
            </div>
        </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
}


function addTypingIndicator() {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant typing';
    msgEl.innerHTML = `
        <div class="message-avatar">⚡</div>
        <div class="message-content">
            <div class="message-text">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        </div>
    `;
    messagesEl.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
}


function removeTypingIndicator(el) {
    if (el && el.parentNode) {
        el.remove();
    }
}


// ── Suggestion & Quick Actions ──────────────────────────────────────

function askSuggestion(btn) {
    const text = btn.textContent.trim();
    userInput.value = text;
    autoResizeTextarea();
    sendBtn.disabled = false;
    inputForm.dispatchEvent(new Event('submit'));
}

function explainClass(className) {
    switchMode('explain');
    userInput.value = className;
    autoResizeTextarea();
    sendBtn.disabled = false;
    inputForm.dispatchEvent(new Event('submit'));
}


// ── Utilities ───────────────────────────────────────────────────────

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Simple markdown formatting for LLM responses
    let html = escapeHtml(text);

    // Code blocks (```...```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code (`...`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold (**...**)
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic (*...*)
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headers (### ...)
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');

    // Lists (- ...)
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Numbered lists (1. ...)
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Paragraphs (double newlines)
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[34]>)/g, '$1');
    html = html.replace(/(<\/h[34]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');

    return html;
}


function toggleSidebar() {
    sidebar.classList.toggle('open');

    // Create/remove backdrop
    let backdrop = document.querySelector('.sidebar-backdrop');
    if (sidebar.classList.contains('open')) {
        if (!backdrop) {
            backdrop = document.createElement('div');
            backdrop.className = 'sidebar-backdrop visible';
            backdrop.addEventListener('click', toggleSidebar);
            document.body.appendChild(backdrop);
        } else {
            backdrop.classList.add('visible');
        }
    } else if (backdrop) {
        backdrop.remove();
    }
}
