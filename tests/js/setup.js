/**
 * jsdom setup — mocks the HTML structure from index.html
 * so that app.js can find DOM elements during testing.
 */

const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

const { JSDOM } = require('jsdom');

const html = `
<!DOCTYPE html>
<html lang="en">
<head><title>DriveStream AI Test</title></head>
<body>
  <div id="app">
    <aside id="sidebar" class="sidebar">
      <div class="sidebar-header">
        <div class="logo">
          <span class="logo-icon">⚡</span>
          <span class="logo-text">DriveStream<span class="logo-ai">AI</span></span>
        </div>
      </div>
      <nav class="sidebar-nav">
        <button class="nav-btn active" data-mode="ask" id="nav-ask">
          <span class="nav-icon">💬</span><span>Ask Codebase</span>
        </button>
        <button class="nav-btn" data-mode="search" id="nav-search">
          <span class="nav-icon">🔍</span><span>Search Code</span>
        </button>
        <button class="nav-btn" data-mode="explain" id="nav-explain">
          <span class="nav-icon">📖</span><span>Explain Class</span>
        </button>
      </nav>
      <div class="sidebar-info">
        <span id="vector-count">0</span>
      </div>
      <div class="sidebar-footer">
        <div class="llm-mode-switch" id="llm-mode-container">
          <label for="llm-mode-select">API Mode:</label>
          <select id="llm-mode-select">
            <option value="remote">Remote API</option>
            <option value="hf_api">HuggingFace API</option>
          </select>
        </div>
        <div id="health-status" class="health-indicator">
          <span class="health-dot"></span>
          <span class="health-text">Connecting...</span>
        </div>
      </div>
    </aside>
    <main id="main-content" class="main">
      <header class="main-header">
        <button id="sidebar-toggle" class="sidebar-toggle">☰</button>
        <h1 id="main-title">Ask about DriveStream</h1>
        <div class="header-badge" id="model-badge">
          <span class="badge-dot"></span>
          <span id="model-name">Loading...</span>
        </div>
      </header>
      <div id="messages" class="messages"></div>
      <form id="input-form" class="input-area">
        <textarea id="user-input" placeholder="Ask a question..."></textarea>
        <button id="send-btn" type="submit" disabled>Send</button>
      </form>
    </main>
  </div>
</body>
</html>
`;

function createDOM() {
  const dom = new JSDOM(html, {
    url: 'http://localhost:8000',
    pretendToBeVisual: true,
    runScripts: 'dangerously',
  });
  return dom;
}

module.exports = { createDOM, html };
