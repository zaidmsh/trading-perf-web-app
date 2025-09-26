// Trading Performance Analyzer - Frontend JavaScript

// Utility functions
function showAlert(message, type, containerId = 'alertArea') {
    const alertContainer = document.getElementById(containerId);
    if (!alertContainer) return;

    alertContainer.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="bi bi-${getAlertIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;

    // Auto-dismiss success alerts after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            const alert = alertContainer.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
}

function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateCSVFile(file) {
    const validTypes = ['text/csv', 'application/vnd.ms-excel'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!file) {
        return { valid: false, message: 'Please select a file.' };
    }

    if (!file.name.toLowerCase().endsWith('.csv') && !validTypes.includes(file.type)) {
        return { valid: false, message: 'Please select a CSV file.' };
    }

    if (file.size > maxSize) {
        return { valid: false, message: `File size (${formatFileSize(file.size)}) exceeds the 10MB limit.` };
    }

    return { valid: true };
}

// File upload handling
class FileUploader {
    constructor() {
        this.uploadForm = document.getElementById('uploadForm');
        this.fileInput = document.getElementById('csvFile');
        this.dropArea = document.getElementById('dropArea');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.uploadBtn = document.getElementById('uploadBtn');

        this.init();
    }

    init() {
        if (!this.uploadForm) return; // Not on upload page

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Form submission
        this.uploadForm.addEventListener('submit', (e) => this.handleSubmit(e));

        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.dropArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.dropArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.dropArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.dropArea.addEventListener('click', () => this.fileInput.click());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropArea.classList.add('border-primary');
        this.dropArea.style.backgroundColor = '#e3f2fd';
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.dropArea.classList.remove('border-primary');
        this.dropArea.style.backgroundColor = '#f8f9fa';
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropArea.classList.remove('border-primary');
        this.dropArea.style.backgroundColor = '#f8f9fa';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.fileInput.files = files;
            this.handleFileSelect({ target: { files: files } });
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        const validation = validateCSVFile(file);

        if (!validation.valid) {
            showAlert(validation.message, 'warning');
            this.fileInput.value = '';
            this.resetDropArea();
            return;
        }

        this.updateDropArea(file);
    }

    updateDropArea(file) {
        const fileSize = formatFileSize(file.size);
        this.dropArea.innerHTML = `
            <div class="d-flex align-items-center justify-content-center">
                <i class="bi bi-file-earmark-check text-success display-4 me-3"></i>
                <div>
                    <p class="mb-1 fw-bold">${file.name}</p>
                    <p class="mb-0 text-muted small">${fileSize}</p>
                    <p class="mb-0 text-muted small">Click to select a different file</p>
                </div>
            </div>
        `;
    }

    resetDropArea() {
        this.dropArea.innerHTML = `
            <i class="bi bi-cloud-upload display-4 mb-3"></i>
            <p class="mb-2">Drag and drop your CSV file here</p>
            <p class="small">or click browse to select a file</p>
        `;
    }

    async handleSubmit(e) {
        e.preventDefault();

        const file = this.fileInput.files[0];
        const validation = validateCSVFile(file);

        if (!validation.valid) {
            showAlert(validation.message, 'warning');
            return;
        }

        this.showProgress();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                showAlert(result.message, 'success');
                this.progressText.textContent = 'Redirecting to results...';

                setTimeout(() => {
                    window.location.href = `/results/${result.session_id}`;
                }, 1500);
            } else {
                showAlert(result.detail || 'Upload failed. Please try again.', 'danger');
                this.hideProgress();
            }
        } catch (error) {
            console.error('Upload error:', error);
            showAlert('Network error. Please check your connection and try again.', 'danger');
            this.hideProgress();
        }
    }

    showProgress() {
        this.progressContainer.classList.remove('d-none');
        this.uploadBtn.disabled = true;
        this.uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Processing...';
        this.progressBar.style.width = '100%';
        this.progressText.textContent = 'Processing your trades...';
    }

    hideProgress() {
        this.progressContainer.classList.add('d-none');
        this.uploadBtn.disabled = false;
        this.uploadBtn.innerHTML = '<i class="bi bi-upload me-2"></i>Analyze Performance';
        this.progressBar.style.width = '0%';
    }
}

// Chart management for results page
class ChartManager {
    constructor(performanceData) {
        this.performanceData = performanceData;
        this.charts = {};
    }

    initializeCharts() {
        if (!this.performanceData) return;

        this.createPerformanceChart();
        this.createDistributionChart();
        this.createBattingAverageChart();
    }

    createPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx || !this.performanceData.monthly || this.performanceData.monthly.length === 0) return;

        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.performanceData.monthly.map(d => d.Date),
                datasets: [
                    {
                        label: 'Net Return (%)',
                        data: this.performanceData.monthly.map(d => d.Net),
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Batting Average (%)',
                        data: this.performanceData.monthly.map(d => d['Win %']),
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Monthly Performance Trend',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    createDistributionChart() {
        const ctx = document.getElementById('winLossChart');
        if (!ctx || !this.performanceData.since_inception) return;

        const data = this.performanceData.since_inception;
        this.charts.distribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Wins', 'Losses', 'Break-even'],
                datasets: [{
                    data: [
                        data.Wins || 0,
                        data.Losses || 0,
                        data['Break-even'] || 0
                    ],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Trade Distribution',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createBattingAverageChart() {
        const ctx = document.getElementById('battingAverageChart');
        if (!ctx || !this.performanceData.monthly || this.performanceData.monthly.length === 0) return;

        this.charts.battingAverage = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.performanceData.monthly.map(d => d.Date),
                datasets: [{
                    label: 'Batting Average (%)',
                    data: this.performanceData.monthly.map(d => d['Win %']),
                    backgroundColor: this.performanceData.monthly.map(d =>
                        d['Win %'] >= 50 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'
                    ),
                    borderColor: this.performanceData.monthly.map(d =>
                        d['Win %'] >= 50 ? 'rgb(40, 167, 69)' : 'rgb(220, 53, 69)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Monthly Batting Average',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    destroyCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};
    }
}

// Export functionality
function exportData(format, period) {
    if (!window.sessionId) {
        showAlert('Session expired. Please upload your file again.', 'warning');
        return;
    }

    const exportBtn = event.target;
    const originalText = exportBtn.innerHTML;

    exportBtn.disabled = true;
    exportBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Exporting...';

    // Create download link
    const url = `/export/${window.sessionId}?format=${format}&period=${period}`;

    // Use a temporary link to trigger download
    const link = document.createElement('a');
    link.href = url;
    link.download = '';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Reset button after a short delay
    setTimeout(() => {
        exportBtn.disabled = false;
        exportBtn.innerHTML = originalText;
    }, 2000);
}

// Table sorting functionality
function sortTable(table, column, type = 'string') {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    const sortedRows = rows.sort((a, b) => {
        const aVal = a.children[column].textContent.trim();
        const bVal = b.children[column].textContent.trim();

        if (type === 'number') {
            const aNum = parseFloat(aVal.replace(/[^-?\d.]/g, ''));
            const bNum = parseFloat(bVal.replace(/[^-?\d.]/g, ''));
            return aNum - bNum;
        }

        return aVal.localeCompare(bVal);
    });

    // Clear tbody and append sorted rows
    tbody.innerHTML = '';
    sortedRows.forEach(row => tbody.appendChild(row));
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize file uploader
    const uploader = new FileUploader();

    // Initialize charts on results page
    if (window.performanceData) {
        const chartManager = new ChartManager(window.performanceData);
        chartManager.initializeCharts();

        // Store chart manager globally for potential cleanup
        window.chartManager = chartManager;
    }

    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-hide alerts after 10 seconds
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 10000);
});