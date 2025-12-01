// DOM 요소들
const themeToggle = document.getElementById('themeToggle');
const body = document.body;
const fileUploadArea = document.getElementById('fileUploadArea');
const fileInput = document.getElementById('fileInput');
const analyzeButton = document.getElementById('analyzeButton');
const uploadContainer = document.querySelector('.upload-container');

// 메인 좌우 섹션 공용 애니메이션 함수
function animateMainSections() {
    const leftSection = document.querySelector('.left-section');
    const rightSection = document.querySelector('.right-section');

    if (leftSection) {
        leftSection.style.opacity = '0';
        leftSection.style.transform = 'translateX(-50px)';
        leftSection.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
        
        setTimeout(() => {
            leftSection.style.opacity = '1';
            leftSection.style.transform = 'translateX(0)';
        }, 300);
    }
    
    if (rightSection) {
        rightSection.style.opacity = '0';
        rightSection.style.transform = 'translateX(50px)';
        rightSection.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
        
        setTimeout(() => {
            rightSection.style.opacity = '1';
            rightSection.style.transform = 'translateX(0)';
        }, 500);
    }
}

// 테마 관리 클래스
class ThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }

    applyTheme(theme) {
        body.setAttribute('data-theme', theme);
        this.currentTheme = theme;
        localStorage.setItem('theme', theme);
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme(newTheme);

        if (themeToggle) {
            themeToggle.style.transform = 'scale(0.95)';
            setTimeout(() => {
                themeToggle.style.transform = 'scale(1)';
            }, 150);
        }
    }
}

// 파일 업로드 관리 클래스
class FileUploadManager {
    constructor() {
        this.selectedFile = null;
        this.isResultShown = false;
        this.init();
    }

    init() {
        if (!fileUploadArea || !fileInput || !analyzeButton) return;

        fileUploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        fileUploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        fileUploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileUploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        analyzeButton.addEventListener('click', this.handleAnalyze.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        fileUploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        if (!this.isValidFile(file)) {
            this.showError('지원하지 않는 파일 형식입니다. PNG 또는 JPG 파일을 선택해주세요.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            this.showError('파일 크기가 너무 큽니다. 10MB 이하의 파일을 선택해주세요.');
            return;
        }

        this.selectedFile = file;

        this.displayFileInfo(file);

        analyzeButton.disabled = false;
        analyzeButton.textContent = 'Analyze';
    }

    isValidFile(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        return validTypes.includes(file.type);
    }

    displayFileInfo(file) {
        const uploadText = fileUploadArea.querySelector('.upload-text');
        
        uploadText.textContent = `선택된 파일: ${file.name}`;
        uploadText.style.color = 'var(--accent-color)';
        uploadText.style.fontWeight = '600';
    }

    showError(message) {
        if (!fileUploadArea) return;

        const uploadText = fileUploadArea.querySelector('.upload-text');
        
        uploadText.textContent = message;
        uploadText.style.color = '#e74c3c';
        uploadText.style.fontWeight = '600';
    }

    // 실제 백엔드와 통신하는 함수
    handleAnalyze() {
        if (this.isResultShown) {
            this.resetView();
            return;
        }

        const file = this.selectedFile || (fileInput && fileInput.files[0]);
        if (!file) {
            this.showError('먼저 이미지 파일을 선택해주세요.');
            return;
        }

        analyzeButton.classList.add('loading');
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';

        const formData = new FormData();
        formData.append("file", file);

        fetch("http://127.0.0.1:5050/api/analyze", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("서버 오류");
            }
            return response.json();
        })
        .then(result => {
            this.showAnalysisResult(result);
        })
        .catch(err => {
            console.error(err);
            this.showError("서버 통신 중 오류가 발생했습니다!");
            analyzeButton.classList.remove('loading');
            analyzeButton.disabled = false;
            analyzeButton.textContent = "Analyze";
        });
    }

    //백엔드 결과 적용
    showAnalysisResult(result) {
        analyzeButton.classList.remove('loading');
        analyzeButton.disabled = false;

        const file = this.selectedFile;
        if (!file) {
            this.showError('이미지 정보를 불러오지 못했습니다.');
            return;            
        }

        const leftContent = document.querySelector('.left-section .content');
        const previewWrapper = document.getElementById('imagePreviewWrapper');
        const previewImage = document.getElementById('previewImage');

        if (leftContent && previewWrapper && previewImage) {
            leftContent.style.display = 'none';
            previewWrapper.style.display = 'flex';

            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        const resultBox = document.getElementById('resultBox');
        const percentEl = document.getElementById('resultPercent');
        const badgeEl = document.getElementById('resultBadge');
        const messageEl = document.getElementById('resultMessage');
        const barFill = document.getElementById('resultProgressFill');

        if (fileUploadArea && resultBox) {
            fileUploadArea.style.display = 'none';
            resultBox.style.display = 'block';
        }

        const confidence = result.confidence || 0;

        percentEl.textContent = `${Math.round(confidence * 100)}%`;
        barFill.style.width = `${confidence * 100}%`;

        if (confidence >= 0.7) {
            badgeEl.textContent = 'AI 사용됨';
            messageEl.textContent = '이 이미지는 AI가 사용되었을 가능성이 매우 높아요.';
        } else if (confidence >= 0.4) {
            badgeEl.textContent = '애매함';
            messageEl.textContent = 'AI가 일부 사용되었을 가능성이 있어요.';
        } else {
            badgeEl.textContent = 'AI 사용 적음';
            messageEl.textContent = '이 이미지는 실제 사진일 가능성이 더 높아요.';
        }

        analyzeButton.textContent = 'Home';
        this.isResultShown = true;

        if (uploadContainer) {
            uploadContainer.classList.add('result-mode');
        }

        animateMainSections();
    }

    resetView() {
        this.isResultShown = false;

        const leftContent = document.querySelector('.left-section .content');
        const previewWrapper = document.getElementById('imagePreviewWrapper');
        const previewImage = document.getElementById('previewImage');

        if (leftContent && previewWrapper && previewImage) {
            leftContent.style.display = 'block';
            previewWrapper.style.display = 'none';
            previewImage.src = '';
        }

        const resultBox = document.getElementById('resultBox');
        if (fileUploadArea && resultBox) {
            fileUploadArea.style.display = 'block';
            resultBox.style.display = 'none';
        }

        const uploadText = fileUploadArea.querySelector('.upload-text');
        if (uploadText) {
            uploadText.textContent = 'Drag & drop your image here, or click to select a file.';
            uploadText.style.color = 'var(--text-color)';
            uploadText.style.fontWeight = '500';
        }

        this.selectedFile = null;
        if (fileInput) {
            fileInput.value = '';
        }

        analyzeButton.textContent = 'Analyze';
        analyzeButton.disabled = false;

        if (uploadContainer) {
            uploadContainer.classList.remove('result-mode');
        }

        animateMainSections();
    }
}

// 유틸리티 함수들
class Utils {
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    static throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// 반응형 헤더 관리
class ResponsiveManager {
    constructor() {
        this.init();
    }

    init() {
        window.addEventListener('resize', Utils.debounce(() => {
            this.handleResize();
        }, 250));

        this.handleResize();
    }
}

// 애니메이션 관리
class AnimationManager {
    constructor() {
        this.init();
    }

    init() {
        this.animateOnLoad();
        this.initScrollAnimations();
    }

    animateOnLoad() {
        const header = document.querySelector('.header');
        if (header) {
            header.style.transform = 'translateY(-100%)';
            header.style.transition = 'transform 0.6s ease-out';
            
            setTimeout(() => {
                header.style.transform = 'translateY(0)';
            }, 100);
        }

        animateMainSections();
    }

    initScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        const elementsToAnimate = document.querySelectorAll('.upload-container, .ascii-art');
        elementsToAnimate.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
            observer.observe(el);
        });
    }
}

// 앱 초기화
class App {
    constructor() {
        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.start();
            });
        } else {
            this.start();
        }
    }

    start() {
        this.themeManager = new ThemeManager();
        this.fileUploadManager = new FileUploadManager();
        this.responsiveManager = new ResponsiveManager();
        this.animationManager = new AnimationManager();

        window.addEventListener('error', (e) => {
            console.error('애플리케이션 오류:', e.error);
        });

        console.log('AI Image Detector 앱이 성공적으로 초기화되었습니다.');
    }
}

// 앱 시작
new App();

function initAboutPage() {
    const illustration = document.querySelector('.about-illustration');
    const cards = document.querySelectorAll('.about-card');

    if (!illustration) return; // about 페이지가 아니면 종료

    // Illustration 먼저 등장
    setTimeout(() => {
        illustration.classList.add('show');
    }, 300);

    // 카드 하나씩 등장
    cards.forEach((card, i) => {
        setTimeout(() => {
            card.classList.add('show');
        }, 400 + i * 200);
    });
}

const originalStartApp = App.prototype.start;
App.prototype.start = function() {
    originalStartApp.call(this);
    initAboutPage();
}