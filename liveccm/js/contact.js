// ===== Contact Form Validation =====
document.addEventListener('DOMContentLoaded', function() {
  const forms = document.querySelectorAll('form[data-netlify="true"]');
  
  forms.forEach(form => {
    form.addEventListener('submit', function(e) {
      const name = this.querySelector('input[name="name"]');
      const email = this.querySelector('input[name="email"]');
      const message = this.querySelector('textarea[name="message"]');
      
      // Basic validation
      let valid = true;
      
      if (name && name.value.trim() === '') {
        valid = false;
        name.style.borderColor = 'red';
      } else if (name) {
        name.style.borderColor = '#ddd';
      }
      
      if (email && email.value.trim() === '') {
        valid = false;
        email.style.borderColor = 'red';
      } else if (email) {
        email.style.borderColor = '#ddd';
      }
      
      if (message && message.value.trim() === '') {
        valid = false;
        message.style.borderColor = 'red';
      } else if (message) {
        message.style.borderColor = '#ddd';
      }
      
      if (!valid) {
        e.preventDefault();
        alert('Please fill in all required fields.');
      }
    });
  });
});