// ===========================
// CONFIGURATION & CONSTANTS
// ===========================

const API_BASE = 'http://localhost:8000';
let IMGENIE_CONFIG = null;

// ===========================
// STATE MANAGEMENT
// ===========================

const appState = {
    currentTask: 'text-to-image',
    selectedModel: '',
    modelLoaded: false,
    isGenerating: false,
    settings: {},
    gpuMemoryUsed: 0,
    maxGpuMemory: 8,
    availableModels: [],
    availableResolutions: []
};

// ===========================
// INITIALIZATION
// ===========================

document.addEventListener('DOMContentLoaded', async () => {
    initializeTheme();
    await loadAppConfig();
    attachEventListeners();
    await populateModels();
    updateTaskInputs(); // Initialize task-specific UI
    updateUIState();
});

// ===========================
// CONFIGURATION MANAGEMENT
// ===========================

async function loadAppConfig() {
    try {
        const response = await fetch('/api/app-config');
        if (response.ok) {
            IMGENIE_CONFIG = await response.json();
            console.log('✓ IMGENIE config loaded:', IMGENIE_CONFIG);
        } else {
            console.warn('⚠️ Could not load config from server');
        }
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

async function populateModels() {
    try {
        const response = await fetch(`/api/models?task=${appState.currentTask}`);
        if (response.ok) {
            appState.availableModels = await response.json();
            updateModelDropdown();
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function updateModelDropdown() {
    const select = document.getElementById('modelSelect');
    const placeholder = select.querySelector('option:first-child');
    
    // Remove existing options except placeholder
    while (select.options.length > 1) {
        select.remove(1);
    }
    
    // Add models from config
    appState.availableModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name || model.id;
        select.appendChild(option);
    });
}

async function updateResolutionsForModel(modelId) {
    try {
        const configTask = appState.currentTask === 'text-to-image' ? 'txt2img' : 'img2txt';
        const response = await fetch(`/api/models/${modelId}/resolutions?task=${appState.currentTask}`);
        if (response.ok) {
            const data = await response.json();
            appState.availableResolutions = data.resolutions || [];
            updateResolutionSelect();
        }
    } catch (error) {
        console.error('Error loading resolutions:', error);
        appState.availableResolutions = ['512x512', '768x768', '1024x1024'];
        updateResolutionSelect();
    }
}

function updateResolutionSelect() {
    const resSelect = document.getElementById('resolutionSelect');
    resSelect.innerHTML = '';
    
    appState.availableResolutions.forEach(res => {
        const option = document.createElement('option');
        option.value = res;
        option.textContent = res;
        resSelect.appendChild(option);
    });
    
    if (resSelect.options.length > 0) {
        resSelect.selectedIndex = 0;
    }
}

// ===========================
// THEME MANAGEMENT
// ===========================

function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('.theme-icon');
    icon.textContent = theme === 'dark' ? '☀️' : '🌙';
}

document.getElementById('themeToggle').addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    showToast(`Switched to ${newTheme} mode`, 'info');
});

// ===========================
// EVENT LISTENERS
// ===========================

function attachEventListeners() {
    // Task selection
    document.querySelectorAll('input[name="task"]').forEach(radio => {
        radio.addEventListener('change', handleTaskChange);
    });

    // Model selection
    document.getElementById('modelSelect').addEventListener('change', handleModelSelect);

    // Model control
    document.getElementById('loadModelBtn').addEventListener('click', handleLoadModel);
    document.getElementById('unloadModelBtn').addEventListener('click', handleUnloadModel);

    // Parameters
    document.getElementById('stepsSlider').addEventListener('input', (e) => {
        document.getElementById('stepsValue').textContent = e.target.value;
    });

    document.getElementById('guidanceSlider').addEventListener('input', (e) => {
        document.getElementById('guidanceValue').textContent = e.target.value;
    });

    document.getElementById('randomSeedBtn').addEventListener('click', () => {
        document.getElementById('seedInput').value = Math.floor(Math.random() * 1000000);
        showToast('Seed randomized', 'info');
    });

    // Inputs
    document.getElementById('promptInput').addEventListener('input', (e) => {
        document.getElementById('promptCount').textContent = e.target.value.length;
    });

    // File uploads
    setupFileUpload('uploadArea', 'referenceImage', 'imagePreview');
    setupFileUpload('uploadAreaDesc', 'imageForDesc', 'imagePreviewDesc');

    // Generation
    document.getElementById('generateBtn').addEventListener('click', handleGenerate);
    document.getElementById('cancelBtn').addEventListener('click', handleCancel);

    // Results actions
    document.getElementById('saveImageBtn')?.addEventListener('click', handleSaveImage);
    document.getElementById('regenerateBtn')?.addEventListener('click', () => {
        document.getElementById('generateBtn').click();
    });
    document.getElementById('copyMetadataBtn')?.addEventListener('click', handleCopyMetadata);

    // Settings
    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('closeSettingsBtn').addEventListener('click', closeSettings);
    document.getElementById('cancelSettingsBtn').addEventListener('click', closeSettings);
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
    document.getElementById('resetSettingsBtn').addEventListener('click', resetSettings);

    // Modal close on background click
    document.getElementById('settingsModal').addEventListener('click', (e) => {
        if (e.target.id === 'settingsModal') closeSettings();
    });
}

// ===========================
// TASK MANAGEMENT
// ===========================

function handleTaskChange(e) {
    appState.currentTask = e.target.value;
    updateTaskInputs();
    updateGenerationUI(); // Update button text immediately
    populateModels(); // Reload models for new task
    updateUIState();
}

function updateTaskInputs() {
    document.querySelectorAll('.task-inputs').forEach(section => {
        section.classList.remove('active');
    });

    const activeSection = document.getElementById(`${appState.currentTask}-inputs`);
    if (activeSection) {
        activeSection.classList.add('active');
    }
    
    // Show/hide txt2img-only parameters
    const txt2imgParams = document.querySelectorAll('.txt2img-only');
    if (appState.currentTask === 'text-to-image') {
        txt2imgParams.forEach(param => {
            param.style.display = 'flex';
        });
    } else {
        txt2imgParams.forEach(param => {
            param.style.display = 'none';
        });
    }
}

// ===========================
// MODEL MANAGEMENT
// ===========================

function handleModelSelect(e) {
    appState.selectedModel = e.target.value;
    
    // Load resolutions for selected model
    if (appState.selectedModel) {
        updateResolutionsForModel(appState.selectedModel);
    }
    
    // Show model details
    if (appState.selectedModel) {
        const model = appState.availableModels.find(m => m.id === appState.selectedModel);
        if (model) {
            const details = model.description || `Model: ${model.name}`;
            document.getElementById('modelDetailsPlaceholder').textContent = details;
        }
    } else {
        document.getElementById('modelDetailsPlaceholder').textContent = 'Select a model to view details';
    }
    
    updateUIState();
}

async function handleLoadModel() {
    if (!appState.selectedModel) {
        showToast('Please select a model first', 'warning');
        return;
    }

    document.getElementById('loadModelBtn').disabled = true;
    document.getElementById('modelStatus').textContent = 'Loading...';
    document.getElementById('modelStatus').className = 'status-value loading';

    try {
        // Simulate API call
        await simulateModelLoad();
        
        appState.modelLoaded = true;
        document.getElementById('modelStatus').textContent = 'Loaded ✓';
        document.getElementById('modelStatus').className = 'status-value loaded';
        
        // Simulate memory usage
        updateMemoryUsage(4.2, 8);
        
        showToast(`Model "${appState.selectedModel}" loaded successfully`, 'success');
        updateUIState();
    } catch (error) {
        document.getElementById('modelStatus').textContent = 'Error';
        document.getElementById('modelStatus').className = 'status-value unloaded';
        showToast('Failed to load model', 'error');
        updateUIState();
    }
}

async function handleUnloadModel() {
    document.getElementById('unloadModelBtn').disabled = true;
    document.getElementById('modelStatus').textContent = 'Unloading...';

    try {
        await simulateModelUnload();
        
        appState.modelLoaded = false;
        document.getElementById('modelStatus').textContent = 'Not Loaded';
        document.getElementById('modelStatus').className = 'status-value unloaded';
        updateMemoryUsage(0, 8);
        
        showToast('Model unloaded successfully', 'success');
        updateUIState();
    } catch (error) {
        showToast('Failed to unload model', 'error');
        updateUIState();
    }
}

function updateMemoryUsage(used, max) {
    const percent = Math.round((used / max) * 100);
    document.getElementById('memoryPercent').textContent = percent + '%';
    document.getElementById('memoryFill').style.width = percent + '%';
    document.getElementById('memoryText').textContent = `${used.toFixed(2)} GB / ${max} GB`;
}

// ===========================
// FILE UPLOAD
// ===========================

function setupFileUpload(uploadAreaId, fileInputId, previewId) {
    const uploadArea = document.getElementById(uploadAreaId);
    const fileInput = document.getElementById(fileInputId);
    const preview = document.getElementById(previewId);

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent-primary)';
        uploadArea.style.background = 'var(--bg-tertiary)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(fileInput, preview);
        }
    });

    fileInput.addEventListener('change', () => handleFileSelect(fileInput, preview));
}

function handleFileSelect(fileInput, preview) {
    const file = fileInput.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        fileInput.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        showToast('Image loaded successfully', 'success');
    };
    reader.readAsDataURL(file);
}

// ===========================
// IMAGE GENERATION
// ===========================

async function handleGenerate() {
    if (!appState.modelLoaded) {
        showToast('Please load a model first', 'warning');
        return;
    }

    if (appState.currentTask === 'text-to-image') {
        const prompt = document.getElementById('promptInput').value.trim();
        if (!prompt) {
            showToast('Please enter a prompt', 'warning');
            return;
        }
    } else if (appState.currentTask === 'image-to-text') {
        const imageFile = document.getElementById('imageForDesc').files[0];
        if (!imageFile) {
            showToast('Please upload an image to describe', 'warning');
            return;
        }
    }

    appState.isGenerating = true;
    updateGenerationUI();

    try {
        if (appState.currentTask === 'text-to-image') {
            // Text-to-Image generation
            const generationData = {
                task: appState.currentTask,
                model: appState.selectedModel,
                steps: document.getElementById('stepsSlider').value,
                guidance_scale: document.getElementById('guidanceSlider').value,
                resolution: document.getElementById('resolutionSelect').value,
                seed: document.getElementById('seedInput').value || null,
                prompt: document.getElementById('promptInput').value,
                timestamp: new Date().toISOString()
            };

            // Simulate generation process
            await simulateGeneration(generationData);

            // Call backend
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(generationData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Generation failed');
            }

            const result = await response.json();
            
            // Mock generated image
            const mockImageUrl = generateMockImage(
                document.getElementById('resolutionSelect').value,
                document.getElementById('promptInput').value
            );

            displayResults(mockImageUrl, generationData);
            showToast('Image generated successfully!', 'success');
        } else if (appState.currentTask === 'image-to-text') {
            // Image-to-Text generation
            const imageFile = document.getElementById('imageForDesc').files[0];
            const formData = new FormData();
            formData.append('task', appState.currentTask);
            formData.append('model', appState.selectedModel);
            formData.append('image', imageFile);

            // Simulate generation process
            await simulateGeneration({ task: appState.currentTask, model: appState.selectedModel });

            // Call backend
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Description generation failed');
            }

            const result = await response.json();
            const description = result.description || 'No description generated';
            
            displayDescription(description, {
                task: appState.currentTask,
                model: appState.selectedModel,
                timestamp: new Date().toISOString()
            });
            showToast('Image described successfully!', 'success');
        }
    } catch (error) {
        console.error('Generation error:', error);
        showToast('Operation failed: ' + error.message, 'error');
    } finally {
        appState.isGenerating = false;
        updateGenerationUI();
    }
}

function handleCancel() {
    appState.isGenerating = false;
    updateGenerationUI();
    showToast('Generation cancelled', 'info');
}

function updateGenerationUI() {
    const generateBtn = document.getElementById('generateBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const progressContainer = document.getElementById('generationProgress');

    // Update button text based on task
    const isTextToImage = appState.currentTask === 'text-to-image';
    const taskName = isTextToImage ? 'Generate Image' : 'Describe Image';
    const taskEmoji = isTextToImage ? '✨' : '📝';
    
    generateBtn.innerHTML = taskEmoji + ' ' + taskName;

    if (appState.isGenerating) {
        generateBtn.style.display = 'none';
        cancelBtn.style.display = 'block';
        cancelBtn.innerHTML = '⏹ Cancel ' + taskName;
        progressContainer.style.display = 'flex';
    } else {
        generateBtn.style.display = 'block';
        cancelBtn.style.display = 'none';
        progressContainer.style.display = 'none';
    }
}

function displayResults(imageUrl, metadata) {
    const resultsSection = document.getElementById('resultsSection');
    const imageElement = document.getElementById('generatedImage');
    
    // Ensure image is visible
    imageElement.style.display = 'block';
    
    // Hide description if it exists
    const descContainer = document.getElementById('descriptionContainer');
    if (descContainer) {
        descContainer.style.display = 'none';
    }
    
    imageElement.src = imageUrl;
    
    document.getElementById('metaModel').textContent = metadata.model;
    document.getElementById('metaTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('metaResolution').textContent = `${metadata.resolution}`;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayDescription(description, metadata) {
    const resultsSection = document.getElementById('resultsSection');
    const imageElement = document.getElementById('generatedImage');
    
    // Hide image and show description instead
    imageElement.style.display = 'none';
    
    // Create or update description container
    let descContainer = document.getElementById('descriptionContainer');
    if (!descContainer) {
        descContainer = document.createElement('div');
        descContainer.id = 'descriptionContainer';
        descContainer.className = 'description-container';
        imageElement.parentNode.insertBefore(descContainer, imageElement);
    }
    
    descContainer.innerHTML = `
        <div class="description-text">
            <p>${description}</p>
        </div>
    `;
    descContainer.style.display = 'flex';
    
    document.getElementById('metaModel').textContent = metadata.model;
    document.getElementById('metaTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('metaResolution').textContent = 'Image Analysis';
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function handleSaveImage() {
    const image = document.getElementById('generatedImage');
    const link = document.createElement('a');
    link.href = image.src;
    link.download = `imgenie-${Date.now()}.png`;
    link.click();
    showToast('Image saved to downloads', 'success');
}

function handleCopyMetadata() {
    const metadata = {
        model: document.getElementById('metaModel').textContent,
        time: document.getElementById('metaTime').textContent,
        resolution: document.getElementById('metaResolution').textContent,
        steps: document.getElementById('stepsValue').textContent,
        guidance: document.getElementById('guidanceValue').textContent,
        seed: document.getElementById('seedInput').value || 'random'
    };

    const text = Object.entries(metadata)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n');

    navigator.clipboard.writeText(text).then(() => {
        showToast('Metadata copied to clipboard', 'success');
    });
}

// ===========================
// SETTINGS MANAGEMENT
// ===========================

function openSettings() {
    document.getElementById('settingsModal').classList.add('active');
    loadSettingsUI();
}

function closeSettings() {
    document.getElementById('settingsModal').classList.remove('active');
}

function loadSettingsUI() {
    // Load from config that was fetched earlier
    document.getElementById('apiEndpoint').value = 'http://localhost:8000';
    document.getElementById('cacheDir').value = '/home/models';
    document.getElementById('defaultRes').value = '512';
    document.getElementById('autoLoadModel').checked = false;
    document.getElementById('maxGpuMemory').value = 8;
    
    updateCachedModelsList();
}

function saveSettings() {
    // Settings would be synced to server config
    closeSettings();
    showToast('Settings will be applied on server restart', 'info');
}

function resetSettings() {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
        showToast('Settings would be reset on server restart', 'info');
        closeSettings();
    }
}

function updateCachedModelsList() {
    const list = document.getElementById('cachedModelsList');
    // This would come from the backend
    const models = [];
    
    if (models.length === 0) {
        list.innerHTML = '<p class="info-text">No cached models</p>';
        return;
    }
    
    list.innerHTML = models.map(model => `
        <div class="model-item">
            <span>${model}</span>
            <button class="model-item-delete" onclick="deleteModel('${model}')">Delete</button>
        </div>
    `).join('');
}

function deleteModel(model) {
    showToast(`Deleted ${model}`, 'success');
    updateCachedModelsList();
}

// ===========================
// UI STATE MANAGEMENT
// ===========================

function updateUIState() {
    const loadBtn = document.getElementById('loadModelBtn');
    const unloadBtn = document.getElementById('unloadModelBtn');
    const generateBtn = document.getElementById('generateBtn');

    if (appState.modelLoaded) {
        loadBtn.disabled = true;
        unloadBtn.disabled = false;
        generateBtn.disabled = false;
    } else {
        loadBtn.disabled = !appState.selectedModel;
        unloadBtn.disabled = true;
        generateBtn.disabled = true;
    }
}

// ===========================
// TOAST NOTIFICATIONS
// ===========================

function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    }[type] || 'ℹ';

    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ===========================
// SIMULATION FUNCTIONS
// ===========================

function simulateModelLoad() {
    return new Promise((resolve) => {
        setTimeout(resolve, 1500);
    });
}

function simulateModelUnload() {
    return new Promise((resolve) => {
        setTimeout(resolve, 800);
    });
}

async function simulateGeneration(data) {
    const startTime = Date.now();
    const steps = parseInt(data.steps);
    const totalDuration = steps * 200; // 200ms per step
    
    return new Promise((resolve) => {
        const progressFill = document.getElementById('generationProgressFill');
        const progressText = document.getElementById('progressText');
        const timeRemaining = document.getElementById('timeRemaining');
        
        const interval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / totalDuration, 0.95);
            
            progressFill.style.width = (progress * 100) + '%';
            progressText.textContent = `Processing... Step ${Math.floor(progress * steps)} of ${steps}`;
            
            const remaining = Math.max(0, totalDuration - elapsed);
            timeRemaining.textContent = `Time remaining: ${Math.round(remaining / 1000)}s`;
            
            if (elapsed >= totalDuration) {
                clearInterval(interval);
                progressFill.style.width = '100%';
                progressText.textContent = 'Processing... Complete!';
                timeRemaining.textContent = 'Time remaining: 0s';
                setTimeout(resolve, 500);
            }
        }, 100);
    });
}

function generateMockImage(resolution, prompt) {
    // Create a canvas with a gradient and text
    const canvas = document.createElement('canvas');
    const size = parseInt(resolution);
    canvas.width = size;
    canvas.height = size;
    
    const ctx = canvas.getContext('2d');
    
    // Create gradient background
    const gradient = ctx.createLinearGradient(0, 0, size, size);
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(0.5, '#16213e');
    gradient.addColorStop(1, '#0f3460');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    
    // Add some visual elements
    ctx.fillStyle = 'rgba(0, 212, 255, 0.2)';
    for (let i = 0; i < 5; i++) {
        ctx.beginPath();
        ctx.arc(
            Math.random() * size,
            Math.random() * size,
            Math.random() * 100 + 20,
            0,
            Math.PI * 2
        );
        ctx.fill();
    }
    
    // Add text
    ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
    ctx.font = `bold ${Math.floor(size / 12)}px "M Plus 2", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🖼️ IMGENIE', size / 2, size / 3);
    
    if (prompt) {
        ctx.font = `${Math.floor(size / 20)}px "M Plus 2", sans-serif`;
        ctx.fillStyle = 'rgba(200, 200, 200, 0.4)';
        const words = prompt.split(' ').slice(0, 3).join(' ');
        ctx.fillText(words, size / 2, size / 1.5);
    }
    
    return canvas.toDataURL('image/png');
}

// ===========================
// UTILITY FUNCTIONS
// ===========================

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// Export for use in console
window.appState = appState;
window.showToast = showToast;
