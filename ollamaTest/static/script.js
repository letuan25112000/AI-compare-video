document.addEventListener('DOMContentLoaded', function() {
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

    // Biến để theo dõi thời gian
    let requestStartTime;

    // Xử lý upload area 1
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

    // Xử lý upload area 2
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

    // Xử lý chọn file thông thường
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

    // Hàm xử lý upload ảnh
    function handleImageUpload(file, input, preview, uploadArea) {
        if (!file.type.match('image.*')) {
            alert('Vui lòng chọn file ảnh!');
            return;
        }

        // Cập nhật input file
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;

        // Hiển thị preview
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Hàm cập nhật thông tin loading
    function updateLoadingStats(elapsedTime) {
        const seconds = Math.floor(elapsedTime / 1000);
        const milliseconds = elapsedTime % 1000;
        loadingStats.innerHTML = `
            <div>Đang xử lý: ${seconds}.${milliseconds.toString().padStart(3, '0')}s</div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <div class="stage-indicators">
                <div class="stage ${elapsedTime < 3000 ? 'active' : ''}">
                    <div class="stage-dot"></div>
                    <div class="stage-label">Nén ảnh</div>
                </div>
                <div class="stage ${elapsedTime >= 3000 ? 'active' : ''}">
                    <div class="stage-dot"></div>
                    <div class="stage-label">AI phân tích</div>
                </div>
            </div>
        `;
    }

    // Xử lý submit form
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Kiểm tra đã chọn ảnh chưa
        if (!image1Input.files[0] || !image2Input.files[0]) {
            alert('Vui lòng chọn cả hai ảnh!');
            return;
        }

        // Bắt đầu đếm thời gian
        requestStartTime = Date.now();
        
        // Hiển thị loading
        loading.style.display = 'block';
        resultSection.style.display = 'none';
        compareBtn.disabled = true;

        // Cập nhật thời gian liên tục
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

            // Tính tổng thời gian
            const totalTime = Date.now() - requestStartTime;

            if (response.ok) {
                // Hiển thị kết quả và thời gian
                resultContent.textContent = data.response || data.error;
                resultContent.className = 'result-content success';
                
                // Hiển thị thông tin thời gian
                if (data.timing) {
                    timingInfo.innerHTML = `
                        <div class="timing-item">
                            <div class="timing-value">${data.timing.compress_time}s</div>
                            <div class="timing-label">Nén ảnh</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-value">${data.timing.api_time}s</div>
                            <div class="timing-label">AI xử lý</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-value">${data.timing.total_time}s</div>
                            <div class="timing-label">Tổng thời gian</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-value">${(totalTime / 1000).toFixed(2)}s</div>
                            <div class="timing-label">Thời gian thực</div>
                        </div>
                    `;
                }
            } else {
                resultContent.textContent = data.error || 'Có lỗi xảy ra';
                resultContent.className = 'result-content error';
                timingInfo.innerHTML = `
                    <div class="timing-item">
                        <div class="timing-value">${(totalTime / 1000).toFixed(2)}s</div>
                        <div class="timing-label">Thời gian xử lý</div>
                    </div>
                `;
            }
            
            resultSection.style.display = 'block';
        } catch (error) {
            const totalTime = Date.now() - requestStartTime;
            resultContent.textContent = 'Lỗi kết nối: ' + error.message;
            resultContent.className = 'result-content error';
            timingInfo.innerHTML = `
                <div class="timing-item">
                    <div class="timing-value">${(totalTime / 1000).toFixed(2)}s</div>
                    <div class="timing-label">Thời gian xử lý</div>
                </div>
            `;
            resultSection.style.display = 'block';
        } finally {
            clearInterval(loadingInterval);
            loading.style.display = 'none';
            compareBtn.disabled = false;
            loadingStats.innerHTML = '';
        }
    });

    // Cho phép reset bằng cách click đúp vào preview
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