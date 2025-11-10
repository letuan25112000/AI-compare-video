document.addEventListener('DOMContentLoaded', function() {
    // --- Get all DOM elements ---
    const uploadForm = document.getElementById('uploadForm');
    const image1Input = document.getElementById('image1');
    const image2Input = document.getElementById('image2');
    const uploadArea1 = document.getElementById('uploadArea1');
    const uploadArea2 = document.getElementById('uploadArea2');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    const compareBtn = document.getElementById('compareBtn');
    const resultSection = document.getElementById('resultSection');
    const resultContent = document.getElementById('resultContent');
    const timingInfo = document.getElementById('timingInfo');
    const loading = document.getElementById('loading');
    const loadingStats = document.getElementById('loadingStats');

    // Used to measure total request duration
    let requestStartTime;

    // --- Upload area 1 events ---
    uploadArea1.addEventListener('click', () => image1Input.click());
    uploadArea1.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea1.style.borderColor = '#4f46e5';
        uploadArea1.style.background = '#f0f4ff';
    });
    uploadArea1.addEventListener('dragleave', () => {
        uploadArea1.style.borderColor = '#cbd5e1';
        uploadArea1.style.background = '#f8fafc';
    });
    uploadArea1.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0], image1Input, preview1, uploadArea1);
        }
    });

    // --- Upload area 2 events ---
    uploadArea2.addEventListener('click', () => image2Input.click());
    uploadArea2.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea2.style.borderColor = '#4f46e5';
        uploadArea2.style.background = '#f0f4ff';
    });
    uploadArea2.addEventListener('dragleave', () => {
        uploadArea2.style.borderColor = '#cbd5e1';
        uploadArea2.style.background = '#f8fafc';
    });
    uploadArea2.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0], image2Input, preview2, uploadArea2);
        }
    });

    // --- Handle normal file selection ---
    image1Input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0], image1Input, preview1, uploadArea1);
        }
    });

    image2Input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0], image2Input, preview2, uploadArea2);
        }
    });

    // --- Handle image upload and preview ---
    function handleImageUpload(file, input, preview, uploadArea) {
        if (!file.type.match('image.*')) {
            alert('画像ファイルを選択してください！'); // Please select an image file
            return;
        }

        // Update file input manually (for drag & drop)
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;

        // Display image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="プレビュー">`;
            uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // --- Update loading animation and progress ---
    function updateLoadingStats(elapsedTime) {
        const seconds = Math.floor(elapsedTime / 1000);
        const milliseconds = elapsedTime % 1000;
        loadingStats.innerHTML = `
            <div>処理中: ${seconds}.${milliseconds.toString().padStart(3, '0')}秒</div>
            <div class="stage-indicators">
                <div class="stage ${elapsedTime < 3000 ? 'active' : ''}">
                    <div class="stage-dot"></div>
                    <div class="stage-label">画像圧縮</div>
                </div>
                <div class="stage ${elapsedTime >= 3000 ? 'active' : ''}">
                    <div class="stage-dot"></div>
                    <div class="stage-label">AI解析</div>
                </div>
            </div>
        `;
    }

    // --- Handle form submission ---
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Validate image selection
        if (!image1Input.files[0] || !image2Input.files[0]) {
            alert('2枚の画像を選択してください！'); // Please select both images
            return;
        }

        // Start timer
        requestStartTime = Date.now();
        
        // Show loading and hide result
        loading.style.display = 'block';
        resultSection.style.display = 'none';
        compareBtn.disabled = true;

        // Update elapsed time every 100ms
        const loadingInterval = setInterval(() => {
            const elapsedTime = Date.now() - requestStartTime;
            updateLoadingStats(elapsedTime);
        }, 100);

        try {
            const formData = new FormData(uploadForm);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            const totalTime = Date.now() - requestStartTime;

            if (response.ok) {
                // Display success result
                resultContent.textContent = data.response || data.error;
                resultContent.className = 'result-content success';
                
                // Show timing info
                if (data.timing) {
                    timingInfo.innerHTML = `
                        <div class="timing-item">
                            <div class="timing-value">${data.timing.compress_time}s</div>
                            <div class="timing-label">画像圧縮</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-value">${data.timing.api_time}s</div>
                            <div class="timing-label">AI処理</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-value">${data.timing.total_time}s</div>
                            <div class="timing-label">合計時間</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-value">${(totalTime / 1000).toFixed(2)}s</div>
                            <div class="timing-label">実測時間</div>
                        </div>
                    `;
                }
            } else {
                // Display error message
                resultContent.textContent = data.error || 'エラーが発生しました。'; // Error occurred
                resultContent.className = 'result-content error';
                timingInfo.innerHTML = `
                    <div class="timing-item">
                        <div class="timing-value">${(totalTime / 1000).toFixed(2)}s</div>
                        <div class="timing-label">処理時間</div>
                    </div>
                `;
            }
            
            resultSection.style.display = 'block';
        } catch (error) {
            const totalTime = Date.now() - requestStartTime;
            resultContent.textContent = '接続エラー: ' + error.message; // Connection error
            resultContent.className = 'result-content error';
            timingInfo.innerHTML = `
                <div class="timing-item">
                    <div class="timing-value">${(totalTime / 1000).toFixed(2)}s</div>
                    <div class="timing-label">処理時間</div>
                </div>
            `;
            resultSection.style.display = 'block';
        } finally {
            // Always reset loading state
            clearInterval(loadingInterval);
            loading.style.display = 'none';
            compareBtn.disabled = false;
            loadingStats.innerHTML = '';
        }
    });

    // --- Allow reset by double-clicking preview ---
    preview1.addEventListener('dblclick', () => {
        preview1.innerHTML = '';
        uploadArea1.style.display = 'flex';
        image1Input.value = '';
    });

    preview2.addEventListener('dblclick', () => {
        preview2.innerHTML = '';
        uploadArea2.style.display = 'flex';
        image2Input.value = '';
    });
});
