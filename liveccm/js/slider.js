// ===== Swiper.js Sliders =====
document.addEventListener('DOMContentLoaded', function() {
  // Hero slider (if needed) - using Swiper
  if (document.querySelector('.hero-slider')) {
    new Swiper('.hero-slider', {
      loop: true,
      autoplay: { delay: 5000 },
      pagination: { el: '.swiper-pagination', clickable: true },
      effect: 'fade'
    });
  }
  
  // Testimonial slider
  if (document.querySelector('.testimonial-slider')) {
    new Swiper('.testimonial-slider', {
      loop: true,
      autoplay: { delay: 6000 },
      pagination: { el: '.swiper-pagination', clickable: true },
      effect: 'fade'
    });
  }
});