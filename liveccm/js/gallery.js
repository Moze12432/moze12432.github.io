// ===== Gallery: Filter Tabs =====
document.addEventListener('DOMContentLoaded', function() {
  const tabs = document.querySelectorAll('.filter-tabs button');
  
  tabs.forEach(tab => {
    tab.addEventListener('click', function() {
      tabs.forEach(t => t.classList.remove('active'));
      this.classList.add('active');
      
      // In a real implementation, filter masonry items here
      console.log('Filter: ' + this.textContent);
    });
  });
  
  // Lightbox placeholder (would be expanded)
  document.querySelectorAll('.masonry-placeholder, .gallery-item').forEach(item => {
    item.addEventListener('click', function() {
      // Placeholder for lightbox
      alert('Lightbox: ' + this.textContent || 'Image');
    });
  });
});