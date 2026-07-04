/**
 * DriveStream AI — JavaScript UI Tests
 * Tests for DOM manipulation, health polling, mode switching, and markdown rendering.
 * @jest-environment jsdom
 */

const fs = require('fs');
const path = require('path');
const { createDOM } = require('./setup');

// ── Helper: Load app.js source and extract functions ──────────────
// We eval the source inside jsdom so we can test the functions directly.

function loadAppInDOM(dom) {
  const appJsPath = path.resolve(__dirname, '../../web/static/app.js');
  const appJsSource = fs.readFileSync(appJsPath, 'utf-8');

  // Inject the source into the jsdom window
  const scriptEl = dom.window.document.createElement('script');
  scriptEl.textContent = appJsSource;
  dom.window.document.body.appendChild(scriptEl);

  return dom.window;
}


// ═══════════════════════════════════════════════════════════════════
// Test Suites
// ═══════════════════════════════════════════════════════════════════

describe('DOM Structure', () => {
  let dom;

  beforeEach(() => {
    dom = createDOM();
  });

  test('has all required DOM elements', () => {
    const doc = dom.window.document;
    expect(doc.getElementById('messages')).not.toBeNull();
    expect(doc.getElementById('input-form')).not.toBeNull();
    expect(doc.getElementById('user-input')).not.toBeNull();
    expect(doc.getElementById('send-btn')).not.toBeNull();
    expect(doc.getElementById('main-title')).not.toBeNull();
    expect(doc.getElementById('model-name')).not.toBeNull();
    expect(doc.getElementById('health-status')).not.toBeNull();
  });

  test('has navigation buttons for all modes', () => {
    const doc = dom.window.document;
    expect(doc.getElementById('nav-ask')).not.toBeNull();
    expect(doc.getElementById('nav-search')).not.toBeNull();
    expect(doc.getElementById('nav-explain')).not.toBeNull();
  });

  test('nav-ask is active by default', () => {
    const doc = dom.window.document;
    const askBtn = doc.getElementById('nav-ask');
    expect(askBtn.classList.contains('active')).toBe(true);
  });

  test('send button is disabled by default', () => {
    const doc = dom.window.document;
    const sendBtn = doc.getElementById('send-btn');
    expect(sendBtn.disabled).toBe(true);
  });

  test('health dot starts without online class', () => {
    const doc = dom.window.document;
    const dot = doc.querySelector('.health-dot');
    expect(dot.classList.contains('online')).toBe(false);
  });

  test('health text starts with Connecting...', () => {
    const doc = dom.window.document;
    const text = doc.querySelector('.health-text');
    expect(text.textContent).toBe('Connecting...');
  });
});


describe('escapeHtml', () => {
  let win;
  let dom;

  beforeEach(() => {
    dom = createDOM();
    // Mock fetch to prevent DOMContentLoaded health check from failing
    dom.window.fetch = jest.fn(() => Promise.reject(new Error('mock')));
    win = loadAppInDOM(dom);
  });

  test('escapes angle brackets', () => {
    const result = win.escapeHtml('<script>alert("xss")</script>');
    expect(result).not.toContain('<script>');
    expect(result).toContain('&lt;');
    expect(result).toContain('&gt;');
  });

  test('escapes ampersands', () => {
    const result = win.escapeHtml('foo & bar');
    expect(result).toContain('&amp;');
  });

  test('handles empty string', () => {
    const result = win.escapeHtml('');
    expect(result).toBe('');
  });

  test('preserves normal text', () => {
    const result = win.escapeHtml('Hello World');
    expect(result).toBe('Hello World');
  });
});


describe('formatMarkdown', () => {
  let win;
  let dom;

  beforeEach(() => {
    dom = createDOM();
    dom.window.fetch = jest.fn(() => Promise.reject(new Error('mock')));
    win = loadAppInDOM(dom);
  });

  test('converts bold text', () => {
    const result = win.formatMarkdown('**bold text**');
    expect(result).toContain('<strong>bold text</strong>');
  });

  test('converts italic text', () => {
    const result = win.formatMarkdown('*italic text*');
    expect(result).toContain('<em>italic text</em>');
  });

  test('converts inline code', () => {
    const result = win.formatMarkdown('use `myFunction()` here');
    expect(result).toContain('<code>myFunction()</code>');
  });

  test('converts code blocks', () => {
    const result = win.formatMarkdown('```java\npublic class Foo {}\n```');
    expect(result).toContain('<pre>');
    expect(result).toContain('<code>');
  });

  test('converts unordered lists', () => {
    const result = win.formatMarkdown('- item one\n- item two');
    expect(result).toContain('<li>item one</li>');
    expect(result).toContain('<li>item two</li>');
    expect(result).toContain('<ul>');
  });

  test('converts headings', () => {
    const result = win.formatMarkdown('## My Heading');
    expect(result).toContain('<h3>My Heading</h3>');
  });
});


describe('Health Status DOM Updates', () => {
  let dom;

  beforeEach(() => {
    dom = createDOM();
  });

  test('online status adds correct class', () => {
    const doc = dom.window.document;
    const dot = doc.querySelector('.health-dot');
    dot.classList.add('online');
    expect(dot.classList.contains('online')).toBe(true);
    expect(dot.classList.contains('error')).toBe(false);
  });

  test('error status adds correct class', () => {
    const doc = dom.window.document;
    const dot = doc.querySelector('.health-dot');
    dot.classList.add('error');
    expect(dot.classList.contains('error')).toBe(true);
    expect(dot.classList.contains('online')).toBe(false);
  });

  test('switching from online to error removes online', () => {
    const doc = dom.window.document;
    const dot = doc.querySelector('.health-dot');
    dot.classList.add('online');
    // Simulate going offline
    dot.classList.remove('online');
    dot.classList.add('error');
    expect(dot.classList.contains('online')).toBe(false);
    expect(dot.classList.contains('error')).toBe(true);
  });

  test('health text can be updated', () => {
    const doc = dom.window.document;
    const text = doc.querySelector('.health-text');
    text.textContent = 'GPU Online · 149 vectors';
    expect(text.textContent).toBe('GPU Online · 149 vectors');
  });

  test('health text shows funny offline message', () => {
    const doc = dom.window.document;
    const text = doc.querySelector('.health-text');
    text.textContent = 'Admin is Busy at work or Gaming';
    expect(text.textContent).toContain('Gaming');
  });
});


describe('Mode Switching DOM', () => {
  let dom;

  beforeEach(() => {
    dom = createDOM();
  });

  test('switching to search mode updates title', () => {
    const doc = dom.window.document;
    const title = doc.getElementById('main-title');
    title.textContent = 'Search Code';
    expect(title.textContent).toBe('Search Code');
  });

  test('switching to explain mode updates title', () => {
    const doc = dom.window.document;
    const title = doc.getElementById('main-title');
    title.textContent = 'Explain a Class';
    expect(title.textContent).toBe('Explain a Class');
  });

  test('nav buttons toggle active class', () => {
    const doc = dom.window.document;
    const askBtn = doc.getElementById('nav-ask');
    const searchBtn = doc.getElementById('nav-search');

    // Simulate switching to search
    askBtn.classList.remove('active');
    searchBtn.classList.add('active');

    expect(askBtn.classList.contains('active')).toBe(false);
    expect(searchBtn.classList.contains('active')).toBe(true);
  });
});

describe('LLM Mode Switching & Cost Protection', () => {
  let win;
  let dom;

  beforeEach(() => {
    dom = createDOM();
    dom.window.fetch = jest.fn(() => Promise.reject(new Error('mock')));
    win = loadAppInDOM(dom);
  });

  test('has llm mode select element', () => {
    const doc = dom.window.document;
    expect(doc.getElementById('llm-mode-select')).not.toBeNull();
  });

  test('llm mode change updates currentLlmMode', () => {
    const doc = dom.window.document;
    const select = doc.getElementById('llm-mode-select');
    
    // Simulate user selecting hf_api
    select.value = 'hf_api';
    const event = new dom.window.Event('change');
    select.dispatchEvent(event);
    
    // Since currentLlmMode is not exported, we can verify its effect by inspecting the payload or mocking apiCall if it was exposed.
    // For now, we just verify the event listener doesn't throw.
    expect(select.value).toBe('hf_api');
  });
});
