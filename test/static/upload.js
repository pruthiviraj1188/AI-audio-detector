// Upload Mode Logic (Simplified - No Live Mode)

document.addEventListener('DOMContentLoaded', () => {
    initializeUploadMode();
});

function initializeUploadMode() {
    // File input listener
    document.getElementById('fileInput').addEventListener('change', handleFileSelect);

    // Upload form listener
    document.getElementById('uploadForm').addEventListener('submit', handleUploadSubmit);
}

// File selection handler
function handleFileSelect(e) {
    const label = document.getElementById('selectedFile');
    if (e.target.files[0]) {
        label.textContent = '✓ ' + e.target.files[0].name;
        label.style.display = 'block';
    }
}

// Upload form submission
async function handleUploadSubmit(e) {
    e.preventDefault();
    const btn = document.getElementById('analyzeBtn');
    const result = document.getElementById('result');

    btn.disabled = true;
    btn.textContent = getTranslation('analyzing');
    result.style.display = 'none';
    result.className = 'result';

    try {
        const formData = new FormData(e.target);
        const res = await fetch('/predict', { 
            method: 'POST', 
            body: formData 
        });
        const data = await res.json();

        if (data.error) {
            result.textContent = '⚠️ ' + data.error;
            result.classList.add('error');
        } else if (data.prediction === 'Human Audio') {
            displayHumanResult(data, result);
        } else {
            displayAIResult(data, result);
        }
        result.style.display = 'block';
    } catch (error) {
        result.textContent = '⚠️ Could not connect to server.';
        result.classList.add('error');
        result.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = getTranslation('analyze_button');
    }
}

function displayHumanResult(data, resultElement) {
    const pct = parseFloat(data.confidence);
    const humanPct = (100 - pct).toFixed(1);
    const t = data.technical;
    resultElement.classList.add('human');

    let reason = '';
    if (humanPct >= 90) reason = 'Very natural speech patterns, consistent rhythm, and organic vocal variations detected.';
    else if (humanPct >= 75) reason = 'Strong human vocal characteristics found — natural breathing, tone variation, and irregular cadence.';
    else if (humanPct >= 60) reason = 'Mostly human traits detected, though some segments showed minor synthetic patterns.';
    else reason = 'Leans human, but confidence is moderate. Audio may be short or have background noise.';

    resultElement.innerHTML = `
        <div class="result-label">✅ ${getTranslation('human_recorded')}</div>
        <div class="progress-wrap"><div class="progress-bar" style="width:0%" id="pbar"></div></div>
        <div class="progress-label"><span>${getTranslation('result_confidence')}</span><span>${humanPct}%</span></div>
        <div class="result-reason">${reason}</div>
        <div class="tech-grid">
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_mfcc">${getTranslation('technical_mfcc')}</div>
                <div class="tech-value">${t.mfcc_variance}</div>
                <div class="tech-desc">High variance = natural vocal irregularity</div>
            </div>
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_spectral">${getTranslation('technical_spectral')}</div>
                <div class="tech-value">${t.spectral_centroid} Hz</div>
                <div class="tech-desc">Brightness of the audio signal</div>
            </div>
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_harmonic">${getTranslation('technical_harmonic')}</div>
                <div class="tech-value">${t.harmonic_ratio}</div>
                <div class="tech-desc">Tonal vs percussive energy balance</div>
            </div>
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_chunks">${getTranslation('technical_chunks')}</div>
                <div class="tech-value">${t.chunks_analyzed} segments</div>
                <div class="tech-desc">2-second audio chunks processed</div>
            </div>
        </div>
    `;
    setTimeout(() => document.getElementById('pbar').style.width = humanPct + '%', 50);
}

function displayAIResult(data, resultElement) {
    const pct = parseFloat(data.confidence);
    const t = data.technical;
    resultElement.classList.add('ai');

    let reason = '';
    if (pct >= 90) reason = 'Highly uniform pitch, robotic cadence, and absence of natural breathing patterns detected.';
    else if (pct >= 75) reason = 'Strong AI markers found — unnatural smoothness, consistent tone, and lack of vocal imperfections.';
    else if (pct >= 60) reason = 'Several AI-like patterns detected, such as flat intonation and overly clean audio quality.';
    else reason = 'Leans AI-generated, but confidence is moderate. Consider uploading a longer audio clip.';

    resultElement.innerHTML = `
        <div class="result-label">🤖 ${getTranslation('ai_generated')}</div>
        <div class="progress-wrap"><div class="progress-bar" style="width:0%" id="pbar"></div></div>
        <div class="progress-label"><span>${getTranslation('result_confidence')}</span><span>${pct.toFixed(1)}%</span></div>
        <div class="result-reason">${reason}</div>
        <div class="tech-grid">
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_mfcc">${getTranslation('technical_mfcc')}</div>
                <div class="tech-value">${t.mfcc_variance}</div>
                <div class="tech-desc">Low variance = synthetic uniformity</div>
            </div>
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_spectral">${getTranslation('technical_spectral')}</div>
                <div class="tech-value">${t.spectral_centroid} Hz</div>
                <div class="tech-desc">Brightness of the audio signal</div>
            </div>
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_harmonic">${getTranslation('technical_harmonic')}</div>
                <div class="tech-value">${t.harmonic_ratio}</div>
                <div class="tech-desc">Tonal vs percussive energy balance</div>
            </div>
            <div class="tech-item">
                <div class="tech-label" data-i18n="technical_chunks">${getTranslation('technical_chunks')}</div>
                <div class="tech-value">${t.chunks_analyzed} segments</div>
                <div class="tech-desc">2-second audio chunks processed</div>
            </div>
        </div>
    `;
    setTimeout(() => document.getElementById('pbar').style.width = pct + '%', 50);
}
