// ===== Mobile Menu & Sticky Header =====
const header = document.getElementById('header');
const toggle = document.getElementById('menuToggle');
const mobileMenu = document.getElementById('mobileMenu');

// Sticky header
window.addEventListener('scroll', () => {
  header.classList.toggle('scrolled', window.scrollY > 80);
});

// Mobile toggle
if (toggle) {
  toggle.addEventListener('click', () => {
    mobileMenu.classList.toggle('open');
  });
}

// Close mobile on link click
document.querySelectorAll('.mobile-menu a').forEach(link => {
  link.addEventListener('click', () => {
    mobileMenu.classList.remove('open');
  });
});

// Close on outside click
document.addEventListener('click', (e) => {
  if (!e.target.closest('header') && !e.target.closest('.mobile-menu')) {
    mobileMenu?.classList.remove('open');
  }
});