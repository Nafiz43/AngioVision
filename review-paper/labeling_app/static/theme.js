// Applied immediately (before first paint) to avoid a flash of the wrong theme.
// Default is light unless the user has explicitly toggled to dark before.
document.documentElement.setAttribute('data-theme', localStorage.getItem('dsa_theme') || 'light');

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('dsa_theme', theme);
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  const current = document.documentElement.getAttribute('data-theme');
  btn.textContent = current === 'dark' ? '☀️ Light' : '🌙 Dark';
  btn.addEventListener('click', () => {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });
});
