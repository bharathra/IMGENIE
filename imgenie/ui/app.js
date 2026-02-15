// ===========================
// CONFIGURATION & CONSTANTS
// ===========================

// API endpoint is relative to where the page is served from
const API_BASE = '/api';
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

    // Check initial model status
    await checkModelStatus();

    updateTaskInputs(); // Initialize task-specific UI
    updateUIState();
});

// ===========================
// CONFIGURATION MANAGEMENT
// ===========================

async function loadAppConfig() {
    try {
        const response = await fetch(`${API_BASE}/app-config`);
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
        const response = await fetch(`${API_BASE}/models?task=${appState.currentTask}`);
        if (response.ok) {
            appState.availableModels = await response.json();
            updateModelDropdown();
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showToast('Error loading models list', 'error');
    }
}

function updateModelDropdown() {
    const select = document.getElementById('modelSelect');

    // Save current selection if possible
    const currentSelection = select.value;

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

    // Restore selection if it exists in new list
    if (currentSelection && appState.availableModels.some(m => m.id === currentSelection)) {
        select.value = currentSelection;
    }
}

async function updateResolutionsForModel(modelId) {
    try {
        const response = await fetch(`${API_BASE}/models/${modelId}/resolutions?task=${appState.currentTask}`);
        if (response.ok) {
            const data = await response.json();
            appState.availableResolutions = data.resolutions || [];
            updateResolutionSelect();
        }
    } catch (error) {
        console.error('Error loading resolutions:', error);
        // Fallback
        appState.availableResolutions = ['720x720', '1024x1024'];
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

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/model/status`);
        if (response.ok) {
            const status = await response.json();

            // Update state based on current task
            const taskKey = appState.currentTask === 'text-to-image' ? 't2i' : 'i2t';
            const modelKey = appState.currentTask === 'text-to-image' ? 't2i_model' : 'i2t_model';

            appState.modelLoaded = status[taskKey];
            const loadedModelId = status[modelKey];

            if (appState.modelLoaded && loadedModelId) {
                appState.selectedModel = loadedModelId;
                const select = document.getElementById('modelSelect');
                if (select) select.value = loadedModelId;

                document.getElementById('modelStatus').textContent = 'Loaded ✓';
                document.getElementById('modelStatus').className = 'status-value loaded';
            } else {
                document.getElementById('modelStatus').textContent = 'Not Loaded';
                document.getElementById('modelStatus').className = 'status-value unloaded';
            }
            updateUIState();
        }
    } catch (e) {
        console.error("Failed to check status", e);
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
    // document.getElementById('cancelBtn').addEventListener('click', handleCancel); // Cancel not impl in backend yet

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

    // Modal close on background click
    document.getElementById('settingsModal').addEventListener('click', (e) => {
        if (e.target.id === 'settingsModal') closeSettings();
    });
}

// ===========================
// TASK MANAGEMENT
// ===========================

async function handleTaskChange(e) {
    appState.currentTask = e.target.value;
    updateTaskInputs();
    updateGenerationUI(); // Update button text immediately
    await populateModels(); // Reload models for new task
    await checkModelStatus(); // Check if model for new task is loaded
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

    const loadBtn = document.getElementById('loadModelBtn');
    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading...';

    document.getElementById('modelStatus').textContent = 'Loading...';
    document.getElementById('modelStatus').className = 'status-value loading';

    try {
        const response = await fetch(`${API_BASE}/model/load`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: appState.selectedModel,
                task: appState.currentTask
            })
        });

        const result = await response.json();

        if (result.success) {
            appState.modelLoaded = true;
            document.getElementById('modelStatus').textContent = 'Loaded ✓';
            document.getElementById('modelStatus').className = 'status-value loaded';
            showToast(`Model "${appState.selectedModel}" loaded successfully`, 'success');
        } else {
            throw new Error(result.error || 'Unknown error');
        }
    } catch (error) {
        document.getElementById('modelStatus').textContent = 'Error';
        document.getElementById('modelStatus').className = 'status-value unloaded';
        showToast(`Failed to load model: ${error.message}`, 'error');
    } finally {
        loadBtn.textContent = '📥 Load Model';
        updateUIState();
    }
}

async function handleUnloadModel() {
    const unloadBtn = document.getElementById('unloadModelBtn');
    unloadBtn.disabled = true;
    unloadBtn.textContent = 'Unloading...';

    try {
        const response = await fetch(`${API_BASE}/model/unload`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                task: appState.currentTask
            })
        });

        const result = await response.json();

        if (result.success) {
            appState.modelLoaded = false;
            document.getElementById('modelStatus').textContent = 'Not Loaded';
            document.getElementById('modelStatus').className = 'status-value unloaded';
            showToast('Model unloaded successfully', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showToast(`Failed to unload model: ${error.message}`, 'error');
    } finally {
        unloadBtn.textContent = '📤 Unload Model';
        updateUIState();
    }
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

    // Show progress bar (indeterminate)
    const progressFill = document.getElementById('generationProgressFill');
    progressFill.style.width = '100%';
    progressFill.classList.add('indeterminate');
    document.getElementById('progressText').textContent = 'Processing... check server logs for details';
    document.getElementById('timeRemaining').textContent = '';

    try {
        let response;

        if (appState.currentTask === 'text-to-image') {
            const payload = {
                task: appState.currentTask,
                model: appState.selectedModel,
                steps: document.getElementById('stepsSlider').value,
                guidance_scale: document.getElementById('guidanceSlider').value,
                resolution: document.getElementById('resolutionSelect').value,
                seed: document.getElementById('seedInput').value || -1,
                prompt: document.getElementById('promptInput').value
            };

            response = await fetch(`${API_BASE}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

        } else {
            const imageFile = document.getElementById('imageForDesc').files[0];
            const formData = new FormData();
            formData.append('task', appState.currentTask);
            formData.append('model', appState.selectedModel);
            formData.append('image', imageFile);

            response = await fetch(`${API_BASE}/generate`, {
                method: 'POST',
                body: formData
            });
        }

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || error.message || 'Generation failed');
        }

        const result = await response.json();

        if (appState.currentTask === 'text-to-image') {
            displayResults(result.image, result.params || {});
            showToast('Image generated successfully!', 'success');
        } else {
            displayDescription(result.description, {
                model: appState.selectedModel
            });
            showToast('Image described successfully!', 'success');
        }

    } catch (error) {
        console.error('Generation error:', error);
        showToast('Operation failed: ' + error.message, 'error');
    } finally {
        appState.isGenerating = false;
        progressFill.classList.remove('indeterminate');
        updateGenerationUI();
    }
}


function updateGenerationUI() {
    const generateBtn = document.getElementById('generateBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const progressContainer = document.getElementById('generationProgress');

    // Update button text based on task
    const isTextToImage = appState.currentTask === 'text-to-image';
    const taskName = isTextToImage ? 'Generate Image' : 'Describe Image';
    const taskEmoji = isTextToImage ? '✨' : '📝';

    if (appState.isGenerating) {
        generateBtn.style.display = 'none';
        cancelBtn.style.display = 'none'; // Backend doesn't support cancel yet
        progressContainer.style.display = 'flex';
    } else {
        generateBtn.innerHTML = taskEmoji + ' ' + taskName;
        generateBtn.style.display = 'block';
        cancelBtn.style.display = 'none';
        progressContainer.style.display = 'none';
    }
}

function displayResults(imageUrl, params) {
    const resultsSection = document.getElementById('resultsSection');
    const imageElement = document.getElementById('generatedImage');
    const descContainer = document.getElementById('descriptionContainer');

    if (descContainer) descContainer.style.display = 'none';
    imageElement.style.display = 'block';

    imageElement.src = imageUrl;

    document.getElementById('metaModel').textContent = appState.selectedModel || 'Unknown';
    document.getElementById('metaTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('metaResolution').textContent = params.resolution || 'Unknown';

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayDescription(description, metadata) {
    const resultsSection = document.getElementById('resultsSection');
    const imageElement = document.getElementById('generatedImage');

    imageElement.style.display = 'none';

    let descContainer = document.getElementById('descriptionContainer');
    if (!descContainer) {
        descContainer = document.createElement('div');
        descContainer.id = 'descriptionContainer';
        descContainer.className = 'description-container';
        imageElement.parentNode.insertBefore(descContainer, imageElement);
    }

    descContainer.innerHTML = `
        <div class="description-text">
            <h3>Image Description</h3>
            <p>${description}</p>
        </div>
    `;
    descContainer.style.display = 'flex';

    document.getElementById('metaModel').textContent = metadata.model;
    document.getElementById('metaTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('metaResolution').textContent = 'N/A';

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function handleSaveImage() {
    const image = document.getElementById('generatedImage');
    const link = document.createElement('a');
    link.href = image.src;
    link.download = `imgenie-${Date.now()}.png`;
    link.click();
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
}

function closeSettings() {
    document.getElementById('settingsModal').classList.remove('active');
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
        loadBtn.style.display = 'none';

        unloadBtn.disabled = false;
        unloadBtn.style.display = 'inline-block';

        generateBtn.disabled = false;
    } else {
        loadBtn.disabled = !appState.selectedModel;
        loadBtn.style.display = 'inline-block';

        unloadBtn.disabled = true;
        unloadBtn.style.display = 'none';

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
