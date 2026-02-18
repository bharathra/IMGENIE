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
    availableResolutions: [],
    savedModelId: ''
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

    // Start polling for status/memory updates
    setInterval(checkModelStatus, 5000);

    // Initial fetch of LoRAs
    fetchLoRAs();
    // Initial fetch of prompt templates
    fetchPromptTemplates();
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
    let currentSelection = select.value;

    // Use saved model preference if available and nothing currently selected
    if (!currentSelection && appState.savedModelId) {
        currentSelection = appState.savedModelId;
    }

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
        appState.selectedModel = currentSelection;

        // Trigger select event to update details etc
        handleModelSelect({ target: select });
    }
}

async function updateResolutionsForModel(modelId) {
    try {
        const response = await fetch(`${API_BASE}/models/${modelId}/resolutions?task=${appState.currentTask}`);
        if (response.ok) {
            const data = await response.json();
            appState.availableResolutions = data.resolutions || [];
            updateResolutionSelect();

            // Restore saved resolution if available
            const saved = localStorage.getItem('imgenie_ui_config');
            if (saved) {
                const config = JSON.parse(saved);
                if (config.resolution && appState.availableResolutions.includes(config.resolution)) {
                    document.getElementById('resolutionSelect').value = config.resolution;
                }
            }
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
                // Only update if we aren't generating to avoid UI jumps
                if (!appState.isGenerating && appState.selectedModel !== loadedModelId) {
                    appState.selectedModel = loadedModelId;
                    const select = document.getElementById('modelSelect');
                    if (select) select.value = loadedModelId;
                }

                document.getElementById('modelStatus').textContent = 'Loaded ✓';
                document.getElementById('modelStatus').className = 'status-value loaded';
            } else {
                document.getElementById('modelStatus').textContent = 'Not Loaded';
                document.getElementById('modelStatus').className = 'status-value unloaded';
            }

            // Update memory usage
            if (status.gpu_memory && status.gpu_memory.max > 0) {
                updateMemoryUsage(status.gpu_memory.reserved, status.gpu_memory.max);
            }

            updateUIState();
        }
    } catch (e) {
        console.error("Failed to check status", e);
    }
}

function updateMemoryUsage(used, max) {
    const percent = Math.round((used / max) * 100);
    document.getElementById('memoryPercent').textContent = percent + '%';
    document.getElementById('memoryFill').style.width = percent + '%';
    document.getElementById('memoryText').textContent = `${used.toFixed(2)} GB / ${max.toFixed(2)} GB`;
}

// ===========================
// LOCAL STORAGE
// ===========================

function saveConfigToLocalStorage() {
    const config = {
        prompt: document.getElementById('promptInput').value,
        steps: document.getElementById('stepsSlider').value,
        guidance: document.getElementById('guidanceSlider').value,
        strength: document.getElementById('strengthSlider') ? document.getElementById('strengthSlider').value : 0.8,
        resolution: document.getElementById('resolutionSelect').value,
        seed: document.getElementById('seedInput').value,
        model: appState.selectedModel,
        loras: {
            char1: document.getElementById('char1Select') ? document.getElementById('char1Select').value : '',
            char1w: document.getElementById('char1Weight') ? document.getElementById('char1Weight').value : 0.66,
            char2: document.getElementById('char2Select') ? document.getElementById('char2Select').value : '',
            char2w: document.getElementById('char2Weight') ? document.getElementById('char2Weight').value : 0.66,
            concept: document.getElementById('conceptSelect') ? document.getElementById('conceptSelect').value : '',
            conceptw: document.getElementById('conceptWeight') ? document.getElementById('conceptWeight').value : 0.33
        }
    };
    localStorage.setItem('imgenie_ui_config', JSON.stringify(config));
}

function loadConfigFromLocalStorage() {
    try {
        const saved = localStorage.getItem('imgenie_ui_config');
        if (saved) {
            const config = JSON.parse(saved);
            if (config.prompt) document.getElementById('promptInput').value = config.prompt;

            if (config.steps) {
                const stepSlider = document.getElementById('stepsSlider');
                if (stepSlider) {
                    stepSlider.value = config.steps;
                    document.getElementById('stepsValue').textContent = config.steps;
                }
            }

            if (config.guidance) {
                const guidanceSlider = document.getElementById('guidanceSlider');
                if (guidanceSlider) {
                    guidanceSlider.value = config.guidance;
                    document.getElementById('guidanceValue').textContent = config.guidance;
                }
            }

            // Strength slider might be added via HTML replacement, check existence
            if (config.strength) {
                const strengthSlider = document.getElementById('strengthSlider');
                if (strengthSlider) {
                    strengthSlider.value = config.strength;
                    document.getElementById('strengthValue').textContent = config.strength;
                }
            }

            // Resolution and Model handled in their respective update/populate functions
            if (config.model) appState.savedModelId = config.model;

            if (config.seed) document.getElementById('seedInput').value = config.seed;

            // Restore LoRAs
            if (config.loras) {
                appState.savedLoRas = config.loras;
            }

            // Update char count
            document.getElementById('promptCount').textContent = document.getElementById('promptInput').value.length;
        }
    } catch (e) {
        console.error("Error loading saved config", e);
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

    // Add listener for strength slider if it exists (might be added dynamically or via HTML update)
    // We delegate or check periodically, but since we modify HTML, checking on init is usually fine if element exists
    const strengthSlider = document.getElementById('strengthSlider');
    if (strengthSlider) {
        strengthSlider.addEventListener('input', (e) => {
            document.getElementById('strengthValue').textContent = e.target.value;
        });
    }

    document.getElementById('randomSeedBtn').addEventListener('click', () => {
        document.getElementById('seedInput').value = Math.floor(Math.random() * 1000000);
        // showToast('Seed randomized', 'info');
        saveConfigToLocalStorage();
    });

    // Inputs
    document.getElementById('promptInput').addEventListener('input', (e) => {
        document.getElementById('promptCount').textContent = e.target.value.length;
        saveConfigToLocalStorage();
    });

    // Save config on change for other inputs
    ['stepsSlider', 'guidanceSlider', 'strengthSlider', 'resolutionSelect', 'seedInput'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', saveConfigToLocalStorage);
    });

    // Save config on LoRA changes
    ['char1Select', 'char1Weight', 'char2Select', 'char2Weight', 'conceptSelect', 'conceptWeight'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', saveConfigToLocalStorage);
    });

    // Prompt template selection
    const promptTemplateSelect = document.getElementById('promptTemplateSelect');
    if (promptTemplateSelect) {
        promptTemplateSelect.addEventListener('change', (e) => {
            const selectedPrompt = e.target.value;
            if (selectedPrompt) {
                const promptInput = document.getElementById('promptInput');
                promptInput.value = selectedPrompt; // Clears and sets new value
                // Trigger input event to update char count and save
                promptInput.dispatchEvent(new Event('input'));

                // Reset select so user can select the same one again if needed
                e.target.value = "";
            }
        });
    }

    // File uploads
    setupFileUpload('uploadArea', 'referenceImage', 'imagePreview', 'clearRefImageBtn');
    setupFileUpload('uploadAreaDesc', 'imageForDesc', 'imagePreviewDesc');

    // Generation
    document.getElementById('generateBtn').addEventListener('click', handleGenerate);

    // Results actions
    document.getElementById('saveImageBtn')?.addEventListener('click', handleSaveImage);
    document.getElementById('deleteImageBtn')?.addEventListener('click', handleDeleteImage);


    // Settings (Removed)
    // document.getElementById('settingsBtn').addEventListener('click', openSettings);
    // document.getElementById('closeSettingsBtn').addEventListener('click', closeSettings);
    // document.getElementById('cancelSettingsBtn').addEventListener('click', closeSettings);


    // Initial Load of config
    loadConfigFromLocalStorage();

    // Setup Image Interactions (Zoom/Pan)
    // Setup Image Interactions (Zoom/Pan)
    setupImageInteraction();

    // Setup Collapsible Sections
    setupCollapsibles();
}

function setupCollapsibles() {
    const cards = document.querySelectorAll('.card');

    // Initially expand first card, collapse others (except generation)
    cards.forEach((card, index) => {
        const header = card.querySelector('.section-header');
        if (!header) return;

        // Special handling for generation card - always visible
        if (card.classList.contains('generation')) {
            card.classList.add('active');
            return; // Skip adding click listener
        }

        // Initially expand first card, collapse others
        if (index === 0) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }

        header.addEventListener('click', (e) => {
            if (e.target.closest('button')) return; // Ignore button clicks

            const isActive = card.classList.contains('active');

            // Determine scope for collapsing siblings
            const parentPanel = card.closest('.controls-panel, .viewport-panel');
            const scope = parentPanel ? parentPanel : document;

            // Close all except generation in the same scope
            scope.querySelectorAll('.card').forEach(c => {
                if (!c.classList.contains('generation') && c !== card) {
                    c.classList.remove('active');
                }
            });

            if (!isActive) {
                card.classList.add('active');
            }
        });
    });
}

function expandSection(sectionClass) {
    const section = document.querySelector(`.card.${sectionClass}`);
    if (section) {
        // Determine scope
        const parentPanel = section.closest('.controls-panel, .viewport-panel');
        const scope = parentPanel ? parentPanel : document;

        // Collapse all first in same scope, except generation
        scope.querySelectorAll('.card').forEach(c => {
            if (!c.classList.contains('generation') && c !== section) {
                c.classList.remove('active');
            }
        });

        section.classList.add('active');

        // Scroll only if on mobile or if not in split view context
        if (window.innerWidth < 1024) {
            section.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
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
    saveConfigToLocalStorage();

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
        // Auto-expand Control section
        expandSection('model-control');
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
            // showToast(`Model "${appState.selectedModel}" loaded successfully`, 'success');

            checkModelStatus(); // Update memory usage immediately
            fetchLoRAs(); // Update LoRAs for the new model
            fetchPromptTemplates(); // Update prompt templates for the new model

            // Auto-advance to Parameters
            expandSection('parameters');
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
            updateMemoryUsage(0, 8); // Reset memory display
            // showToast('Model unloaded successfully', 'success');
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

async function handleDeleteImage() {
    if (!appState.lastImageId) {
        showToast('No image to delete', 'warning');
        return;
    }

    // if (!confirm('Are you sure you want to delete this image? This cannot be undone.')) {
    //     return;
    // }

    const deleteBtn = document.getElementById('deleteImageBtn');
    let originalText = '🗑️ Delete';
    if (deleteBtn) {
        originalText = deleteBtn.innerHTML;
        deleteBtn.innerHTML = 'Deleting...';
        deleteBtn.disabled = true;
    }

    try {
        const response = await fetch(`${API_BASE}/delete_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: appState.lastImageId
            })
        });

        const result = await response.json();

        if (result.success) {
            // showToast('Image deleted successfully', 'success');
            // Clear viewport
            // Clear viewport
            const imgEl = document.getElementById('generatedImage');
            imgEl.removeAttribute('src');
            imgEl.style.display = 'none'; // Explicitly hide

            // In landscape, section is forced visible, so hiding display:none is moot there
            // but harmless for portrait.
            document.getElementById('resultsSection').style.display = 'none';

            appState.lastImageId = null;
        } else {
            throw new Error(result.error || 'Failed to delete image');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast(`Delete failed: ${error.message}`, 'error');
    } finally {
        if (deleteBtn) {
            deleteBtn.innerHTML = originalText;
            deleteBtn.disabled = false;
        }
    }
}

function setupFileUpload(uploadAreaId, fileInputId, previewId, clearBtnId = null) {
    const uploadArea = document.getElementById(uploadAreaId);
    const fileInput = document.getElementById(fileInputId);
    const preview = document.getElementById(previewId);

    if (clearBtnId) {
        const clearBtn = document.getElementById(clearBtnId);
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.value = '';
            preview.innerHTML = '';
            clearBtn.style.display = 'none';
        });
    }

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
            const success = handleFileSelect(fileInput, preview);
            if (clearBtnId) {
                document.getElementById(clearBtnId).style.display = success ? 'block' : 'none';
            }
        }
    });

    fileInput.addEventListener('change', () => {
        const success = handleFileSelect(fileInput, preview);
        if (clearBtnId) {
            document.getElementById(clearBtnId).style.display = success ? 'block' : 'none';
        }
    });
}

function handleFileSelect(fileInput, preview) {
    const file = fileInput.files[0];
    if (!file) {
        // Input cleared or empty, clear preview too
        preview.innerHTML = '';
        return false;
    }

    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        fileInput.value = '';
        preview.innerHTML = '';
        return false;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        // showToast('Image loaded successfully', 'success');
    };
    reader.readAsDataURL(file);
    return true;
}

// ===========================
// IMAGE GENERATION
// ===========================

async function pollGenerationProgress() {
    if (!appState.isGenerating) return;

    try {
        const response = await fetch(`${API_BASE}/progress`);
        if (response.ok) {
            const data = await response.json();

            const progressFill = document.getElementById('generationProgressFill');
            const progressText = document.getElementById('progressText');

            if (data.status === 'generating') {
                progressFill.classList.remove('indeterminate');
                progressFill.style.width = `${data.progress}%`;
                progressText.textContent = data.message || `Processing... ${data.progress}%`;
            } else if (data.status === 'starting') {
                progressFill.style.width = '100%';
                progressFill.classList.add('indeterminate');
                progressText.textContent = 'Starting generation...';
            }

            if (appState.isGenerating && data.status !== 'completed' && data.status !== 'failed') {
                setTimeout(pollGenerationProgress, 500); // Poll every 500ms
            }
        }
    } catch (e) {
        console.error("Error polling progress", e);
    }
}

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

    // Auto-expand Generation panel
    expandSection('generation');

    // Start polling progress
    pollGenerationProgress();

    // Show progress bar
    const progressFill = document.getElementById('generationProgressFill');
    progressFill.style.width = '100%';
    progressFill.classList.add('indeterminate');
    document.getElementById('progressText').textContent = 'Initializing...';
    document.getElementById('timeRemaining').textContent = '';

    try {
        let response;

        if (appState.currentTask === 'text-to-image') {
            const refImageFile = document.getElementById('referenceImage').files[0];

            if (refImageFile) {
                // Use FormData for Img2Img
                const formData = new FormData();
                formData.append('task', appState.currentTask);
                formData.append('model', appState.selectedModel);
                formData.append('steps', document.getElementById('stepsSlider').value);
                formData.append('guidance_scale', document.getElementById('guidanceSlider').value);
                formData.append('strength', document.getElementById('strengthSlider') ? document.getElementById('strengthSlider').value : 0.8);
                formData.append('resolution', document.getElementById('resolutionSelect').value);
                formData.append('seed', document.getElementById('seedInput').value || -1);
                formData.append('prompt', document.getElementById('promptInput').value);
                formData.append('image', refImageFile);

                // Add LoRAs to FormData
                const loras = [];
                const addLoRa = (selectId, weightId, type) => {
                    const el = document.getElementById(selectId);
                    const wEl = document.getElementById(weightId);
                    if (el && el.value) {
                        loras.push({
                            name: el.value,
                            weight: wEl ? wEl.value : 1.0,
                            type: type
                        });
                    }
                };

                addLoRa('char1Select', 'char1Weight', 'character');
                addLoRa('char2Select', 'char2Weight', 'character');
                addLoRa('conceptSelect', 'conceptWeight', 'concept');

                formData.append('loras', JSON.stringify(loras));

                response = await fetch(`${API_BASE}/generate`, {
                    method: 'POST',
                    body: formData
                });
            } else {
                // Use JSON for Txt2Img
                const payload = {
                    task: appState.currentTask,
                    model: appState.selectedModel,
                    steps: document.getElementById('stepsSlider').value,
                    guidance_scale: document.getElementById('guidanceSlider').value,
                    strength: document.getElementById('strengthSlider') ? document.getElementById('strengthSlider').value : 0.8,
                    resolution: document.getElementById('resolutionSelect').value,
                    seed: document.getElementById('seedInput').value || -1,
                    prompt: document.getElementById('promptInput').value,
                    loras: []
                };

                // Add LoRAs
                const addLoRa = (selectId, weightId, type) => {
                    const el = document.getElementById(selectId);
                    const wEl = document.getElementById(weightId);
                    if (el && el.value) {
                        payload.loras.push({
                            name: el.value,
                            weight: wEl ? wEl.value : 1.0,
                            type: type
                        });
                    }
                };

                addLoRa('char1Select', 'char1Weight', 'character');
                addLoRa('char2Select', 'char2Weight', 'character');
                addLoRa('conceptSelect', 'conceptWeight', 'concept');

                response = await fetch(`${API_BASE}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }

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
            displayResults(result.image, result.params || {}, result.image_id);
            // showToast('Image generated successfully!', 'success');
        } else {
            displayDescription(result.description, {
                model: appState.selectedModel
            });
            // showToast('Image described successfully!', 'success');
        }

    } catch (error) {
        console.error('Generation error:', error);
        showToast('Operation failed: ' + error.message, 'error');
    } finally {
        appState.isGenerating = false;
        progressFill.classList.remove('indeterminate');
        updateGenerationUI();
        saveConfigToLocalStorage();
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

function displayResults(imageUrl, params, imageId) {
    const resultsSection = document.getElementById('resultsSection');
    const imageElement = document.getElementById('generatedImage');
    const descContainer = document.getElementById('descriptionContainer');

    if (descContainer) descContainer.style.display = 'none';
    imageElement.style.display = 'block';

    imageElement.src = imageUrl;
    appState.lastImageId = imageId;
    appState.lastImageParams = params;

    // document.getElementById('metaModel').textContent = appState.selectedModel || 'Unknown';
    // document.getElementById('metaTime').textContent = new Date().toLocaleTimeString();
    // document.getElementById('metaResolution').textContent = params.resolution || 'Unknown';

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    // Auto-expand Results panel
    expandSection('results');
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

    // document.getElementById('metaModel').textContent = metadata.model;
    // document.getElementById('metaTime').textContent = new Date().toLocaleTimeString();
    // document.getElementById('metaResolution').textContent = 'N/A';

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

async function handleSaveImage() {
    if (!appState.lastImageId) {
        showToast('No image to save', 'warning');
        return;
    }

    const saveBtn = document.getElementById('saveImageBtn');
    const originalText = saveBtn.textContent;
    saveBtn.textContent = 'Saving...';
    saveBtn.disabled = true;

    try {
        const payload = {
            image_id: appState.lastImageId,
            metadata: {
                model_id: appState.selectedModel,
                params: appState.lastImageParams || {},
                task: appState.currentTask
            }
        };

        const response = await fetch(`${API_BASE}/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.success) {
            // showToast(`Image saved to server output folder`, 'success');
        } else {
            showToast(`Save failed: ${result.error}`, 'error');
        }
    } catch (e) {
        console.error("Save error:", e);
        showToast('Save failed', 'error');
    } finally {
        saveBtn.textContent = originalText;
        saveBtn.disabled = false;
    }
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

// ===========================
// IMAGE INTERACTION
// ===========================

function setupImageInteraction() {
    const viewport = document.getElementById('imageViewport');
    const image = document.getElementById('generatedImage');

    if (!viewport || !image) return;

    let scale = 1;
    let pointX = 0;
    let pointY = 0;
    let startX = 0;
    let startY = 0;
    let isPanning = false;

    viewport.style.cursor = 'grab';

    function updateTransform() {
        image.style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
    }

    // Zoom
    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();

        const rect = viewport.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Calculate offset of image relative to viewport due to flex centering
        // We use the image's bounding rect relative to viewport when scale=1, translate=0
        // But since we are transforming it, image.getBoundingClientRect() changes.
        // However, image.offsetLeft/Top are relative to the offsetParent (viewport).
        // They are reliable if the element is not transformed? No, transforms affect layout? 
        // No, transforms do not affect flow layout.

        const offX = image.offsetLeft;
        const offY = image.offsetTop;

        // Mouse relative to the *initial layout position* of the image
        const relMouseX = mouseX - offX;
        const relMouseY = mouseY - offY;

        // Calculate point in "image coordinates" (0 to width/height)
        const xs = (relMouseX - pointX) / scale;
        const ys = (relMouseY - pointY) / scale;

        const delta = -Math.sign(e.deltaY);
        const step = 0.1;

        if (delta > 0) {
            scale *= (1 + step);
        } else {
            scale /= (1 + step);
        }

        scale = Math.min(Math.max(0.1, scale), 10);

        // Update pointX/Y to keep xs/ys under mouse
        pointX = relMouseX - xs * scale;
        pointY = relMouseY - ys * scale;

        updateTransform();
    });

    // Pan
    viewport.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // Left click
            e.preventDefault();
            startX = e.clientX - pointX;
            startY = e.clientY - pointY;
            isPanning = true;
            viewport.style.cursor = 'grabbing';
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        e.preventDefault();
        pointX = e.clientX - startX;
        pointY = e.clientY - startY;
        updateTransform();
    });

    window.addEventListener('mouseup', () => {
        if (isPanning) {
            isPanning = false;
            viewport.style.cursor = 'grab';
        }
    });

    // Reset
    viewport.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        scale = 1;
        pointX = 0;
        pointY = 0;
        updateTransform();
    });
}


// ===========================
// LORA MANAGEMENT
// ===========================

async function fetchLoRAs() {
    try {
        const response = await fetch(`${API_BASE}/loras`);
        if (response.ok) {
            const data = await response.json();
            updateLoRaDropdowns(data.characters || [], data.concepts || []);
        }
    } catch (e) {
        console.error("Error fetching LoRAs:", e);
    }
}

function updateLoRaDropdowns(characters, concepts) {
    const updateSelect = (id, items) => {
        const select = document.getElementById(id);
        if (!select) return;

        const current = select.value;
        select.innerHTML = '<option value="">None</option>';

        items.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item;
            opt.textContent = item;
            select.appendChild(opt);
        });

        if (current && items.includes(current)) {
            select.value = current;
        }
    };

    updateSelect('char1Select', characters);
    updateSelect('char2Select', characters);
    updateSelect('conceptSelect', concepts);

    // Restore saved LoRAs if available
    if (appState.savedLoRas) {
        const l = appState.savedLoRas;
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el && val) el.value = val;
        };

        setVal('char1Select', l.char1);
        setVal('char1Weight', l.char1w);
        setVal('char2Select', l.char2);
        setVal('char2Weight', l.char2w);
        setVal('conceptSelect', l.concept);
        setVal('conceptWeight', l.conceptw);

        // Clear after restoring so we don't overwrite if user changes subsequently
        appState.savedLoRas = null;
    }
}


// ===========================
// PROMPT TEMPLATES
// ===========================

async function fetchPromptTemplates() {
    try {
        const response = await fetch(`${API_BASE}/prompts`);
        if (response.ok) {
            const data = await response.json();
            updatePromptTemplateDropdown(data);
        }
    } catch (e) {
        console.error("Error fetching prompt templates:", e);
    }
}

function updatePromptTemplateDropdown(prompts) {
    const select = document.getElementById('promptTemplateSelect');
    if (!select) return;

    select.innerHTML = '<option value="">-- Select a prompt template --</option>';

    // prompts is expected to be a dict: { "keyword": "prompt text", ... }
    if (prompts && typeof prompts === 'object') {
        Object.entries(prompts).forEach(([keyword, text]) => {
            const opt = document.createElement('option');
            opt.value = text;
            opt.textContent = keyword; // specific format ? keyword: prompt ... user said "keyword: prompt" format in yaml
            select.appendChild(opt);
        });

        // If there are prompts, show the group, otherwise maybe hide it?
        // document.getElementById('promptTemplateGroup').style.display = Object.keys(prompts).length > 0 ? 'block' : 'none';
    }
}
