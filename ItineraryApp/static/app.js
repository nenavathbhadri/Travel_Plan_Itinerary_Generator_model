/* ============================================
   APP.JS — Travel Plan Itinerary Generator
   Core JavaScript: Theme, Validation, UX
   ============================================ */

(function () {
  'use strict';

  // =============================================
  //  DARK / LIGHT MODE TOGGLE
  // =============================================
  const ThemeManager = {
    STORAGE_KEY: 'travelplan-theme',

    init() {
      const saved = localStorage.getItem(this.STORAGE_KEY);
      if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
      } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
      }

      document.querySelectorAll('.theme-toggle').forEach(btn => {
        btn.addEventListener('click', () => this.toggle());
      });
    },

    toggle() {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem(this.STORAGE_KEY, next);
    }
  };

  // =============================================
  //  NAVBAR SCROLL BEHAVIOR
  // =============================================
  const NavbarManager = {
    init() {
      const navbar = document.querySelector('.navbar');
      if (!navbar) return;

      window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
          navbar.classList.add('scrolled');
        } else {
          navbar.classList.remove('scrolled');
        }
      }, { passive: true });

      // Mobile menu toggle
      const menuBtn = document.querySelector('.mobile-menu-btn');
      const navLinks = document.querySelector('.nav-links');
      if (menuBtn && navLinks) {
        menuBtn.addEventListener('click', () => {
          navLinks.classList.toggle('open');
          const isOpen = navLinks.classList.contains('open');
          menuBtn.innerHTML = isOpen ? '✕' : '☰';
          menuBtn.setAttribute('aria-expanded', isOpen);
        });

        // Close menu on nav link click
        navLinks.querySelectorAll('a').forEach(link => {
          link.addEventListener('click', () => {
            navLinks.classList.remove('open');
            menuBtn.innerHTML = '☰';
          });
        });
      }
    }
  };

  // =============================================
  //  FORM VALIDATION (Modern inline)
  // =============================================
  const FormValidator = {
    init() {
      document.querySelectorAll('form[data-validate]').forEach(form => {
        form.addEventListener('submit', (e) => {
          if (!this.validateForm(form)) {
            e.preventDefault();
          } else {
            // Show loading state
            const submitBtn = form.querySelector('[type="submit"]');
            if (submitBtn && form.dataset.loading !== 'false') {
              submitBtn.classList.add('loading');
              submitBtn.disabled = true;
              this.showLoadingOverlay(form.dataset.loadingMessage || 'Processing...');
            }
          }
        });

        // Live validation on blur
        form.querySelectorAll('.form-input[required]').forEach(input => {
          input.addEventListener('blur', () => this.validateField(input));
          input.addEventListener('input', () => {
            if (input.classList.contains('error')) {
              this.validateField(input);
            }
          });
        });
      });
    },

    validateForm(form) {
      let isValid = true;
      form.querySelectorAll('.form-input[required]').forEach(input => {
        if (!this.validateField(input)) {
          isValid = false;
        }
      });
      return isValid;
    },

    validateField(input) {
      const value = input.value.trim();
      let errorMsg = '';

      // Required check
      if (!value) {
        errorMsg = `Please enter ${input.dataset.label || 'this field'}`;
      }
      // Email check
      else if (input.type === 'email') {
        const emailRegex = /^[a-zA-Z0-9._-]+@[a-z]+\.(com|in|org|net|edu)$/;
        if (!emailRegex.test(value)) {
          errorMsg = 'Please enter a valid email address';
        }
      }
      // Number check
      else if (input.type === 'number' || input.dataset.type === 'number') {
        if (isNaN(value)) {
          errorMsg = 'Please enter a valid number';
        }
      }
      // Min length
      else if (input.minLength > 0 && value.length < input.minLength) {
        errorMsg = `Must be at least ${input.minLength} characters`;
      }

      // Show/hide error
      const errorEl = input.parentElement.querySelector('.form-error') ||
                       input.closest('.form-group')?.querySelector('.form-error');

      if (errorMsg) {
        input.classList.add('error');
        if (errorEl) {
          errorEl.textContent = errorMsg;
          errorEl.style.display = 'block';
        }
        return false;
      } else {
        input.classList.remove('error');
        if (errorEl) {
          errorEl.style.display = 'none';
        }
        return true;
      }
    },

    showLoadingOverlay(message) {
      const overlay = document.querySelector('.loading-overlay');
      if (overlay) {
        const msgEl = overlay.querySelector('p');
        if (msgEl) msgEl.textContent = message;
        overlay.classList.add('active');
      }
    }
  };

  // =============================================
  //  SCROLL REVEAL ANIMATIONS
  // =============================================
  const ScrollReveal = {
    init() {
      const elements = document.querySelectorAll('.animate-on-scroll');
      if (!elements.length) return;

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
      });

      elements.forEach(el => observer.observe(el));
    }
  };

  // =============================================
  //  BUTTON RIPPLE EFFECT
  // =============================================
  const RippleEffect = {
    init() {
      document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
          const ripple = document.createElement('span');
          ripple.classList.add('ripple-effect');
          const rect = this.getBoundingClientRect();
          ripple.style.left = (e.clientX - rect.left - 15) + 'px';
          ripple.style.top = (e.clientY - rect.top - 15) + 'px';
          this.appendChild(ripple);
          setTimeout(() => ripple.remove(), 600);
        });
      });
    }
  };

  // =============================================
  //  TOAST NOTIFICATIONS
  // =============================================
  const Toast = {
    show(message, type = 'success', duration = 3000) {
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      toast.textContent = message;
      document.body.appendChild(toast);

      setTimeout(() => {
        toast.classList.add('hiding');
        setTimeout(() => toast.remove(), 300);
      }, duration);
    }
  };

  // Expose Toast globally
  window.Toast = Toast;

  // =============================================
  //  AUTO-DETECT & SHOW DJANGO MESSAGES
  // =============================================
  const DjangoMessages = {
    init() {
      const msgEl = document.querySelector('[data-django-message]');
      if (msgEl) {
        const message = msgEl.dataset.djangoMessage;
        const type = msgEl.dataset.djangoMessageType || 'success';
        if (message && message !== 'None') {
          Toast.show(message, type);
        }
      }
    }
  };

  // =============================================
  //  INITIALIZE ON DOM READY
  // =============================================
  document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    NavbarManager.init();
    FormValidator.init();
    ScrollReveal.init();
    RippleEffect.init();
    DjangoMessages.init();
  });

})();
