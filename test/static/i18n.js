// i18n - Internationalization
let translations = {};
let currentLanguage = 'en';

// Load translations
async function loadTranslations() {
    try {
        const response = await fetch('/static/translations.json');
        translations = await response.json();
        
        // Load saved language preference or default to English
        const savedLang = localStorage.getItem('language') || 'en';
        setLanguage(savedLang);
    } catch (error) {
        console.error('Failed to load translations:', error);
    }
}

// Set language
function setLanguage(lang) {
    if (!translations[lang]) {
        console.warn(`Language ${lang} not found, falling back to English`);
        lang = 'en';
    }
    
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    
    // Update language selector
    const selector = document.getElementById('languageSelect');
    if (selector) {
        selector.value = lang;
    }
    
    // Update all text elements with data-i18n attribute
    updatePageText();
}

// Update all translatable text on page
function updatePageText() {
    const elements = document.querySelectorAll('[data-i18n]');
    elements.forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = getTranslation(key);
        
        // Update text content or placeholder based on element type
        if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
            element.placeholder = translation;
        } else if (element.tagName === 'BUTTON' || element.tagName === 'A') {
            element.textContent = translation;
        } else {
            element.textContent = translation;
        }
    });
}

// Get translation for a key
function getTranslation(key) {
    if (!translations[currentLanguage]) {
        return key;
    }
    return translations[currentLanguage][key] || translations['en'][key] || key;
}

// Get current language
function getCurrentLanguage() {
    return currentLanguage;
}

// Language selector event listener
document.addEventListener('DOMContentLoaded', () => {
    loadTranslations();
    
    const selector = document.getElementById('languageSelect');
    if (selector) {
        selector.addEventListener('change', (e) => {
            setLanguage(e.target.value);
        });
    }
});
