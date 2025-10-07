// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function () {
    console.log("✅ Script initialized");

    // ========= UPLOAD LOGIC =========
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const detectBtn = document.getElementById("detectBtn");
    const resultDiv = document.getElementById("result");

    let selectedImage = null;

    // Preview uploaded image
    fileInput?.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = () => {
                preview.src = reader.result;
                preview.classList.remove("d-none");
                detectBtn.classList.remove("d-none");
                selectedImage = file;
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle detect button click (upload mode)
    detectBtn?.addEventListener("click", async () => {
        if (!selectedImage) {
            resultDiv.innerHTML = "⚠️ Please select an image first.";
            resultDiv.className = "alert alert-warning";
            return;
        }

        resultDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="spinner-border text-success me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span>Detecting waste objects...</span>
            </div>
        `;
        resultDiv.className = "alert alert-warning";

        const formData = new FormData();
        formData.append("file", selectedImage);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                const html = await response.text();
                document.open();
                document.write(html);
                document.close();
            } else {
                const errorText = await response.text();
                throw new Error(`Server error: ${errorText}`);
            }
        } catch (error) {
            resultDiv.innerHTML = `⚠️ Error: ${error.message}`;
            resultDiv.className = "alert alert-danger";
        }
    });

    // ========= LIVE CAMERA LOGIC =========
    const openLiveBtn = document.getElementById("openLiveBtn");
    const liveSection = document.getElementById("liveSection");
    const closeLiveBtn = document.getElementById("closeLiveBtn");
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const liveStatus = document.getElementById("liveStatus");

    let stream = null;
    let isDetecting = false;
    let canvas = document.createElement("canvas");
    let ctx = canvas.getContext("2d");

    // Color map for waste categories
    const colorMap = {
        "Dry Waste": "#00FF00",
        "Wet Waste": "#FFFF00",
        "Hazardous Waste": "#FF0000",
        "Electronic Waste": "#FF00FF",
        "Construction Waste": "#00FFFF",
        "Biomedical Waste": "#800080"
    };

    // ✅ OPEN LIVE CAMERA SECTION + AUTO START CAMERA
    openLiveBtn?.addEventListener("click", async () => {
        console.log("📹 Opening live camera...");

        liveSection.classList.remove("d-none");
        window.scrollTo({ top: liveSection.offsetTop - 50, behavior: 'smooth' });

        try {
            liveStatus.innerText = "⏳ Requesting camera access...";
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });

            video.srcObject = stream;
            video.play();

            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                console.log(`📹 Camera ready: ${video.videoWidth} x ${video.videoHeight}`);
                liveStatus.innerText = "✅ Tap on any object to detect it!";
            };

        } catch (err) {
            if (err.name === 'NotAllowedError') {
                liveStatus.innerText = "❌ Please allow camera access.";
            } else {
                liveStatus.innerText = `❌ Camera error: ${err.message}`;
            }
            console.error("Camera Error:", err);
        }
    });

    // ✅ CLOSE LIVE CAMERA SECTION
    closeLiveBtn?.addEventListener("click", () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        liveSection.classList.add("d-none");
        overlay.innerHTML = '';
        liveStatus.innerText = "";
    });

    // ✅ HANDLE VIDEO CLICK — TRIGGER DETECTION
    video?.addEventListener('click', async function(e) {
        if (!stream || isDetecting) return;

        isDetecting = true;
        liveStatus.innerText = "🔍 Analyzing...";

        const rect = video.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const result = await response.json();
                if (result.success) {
                    processClickDetection(result.results, clickX, clickY, rect.width, rect.height);
                }

            } catch (err) {
                liveStatus.innerText = "⚠️ Detection failed.";
                console.error("Detection Error:", err);
            } finally {
                isDetecting = false;
            }
        }, 'image/jpeg', 0.7);
    });

    // ✅ Handle click detection (proper scaling)
    function processClickDetection(detections, clickX, clickY, displayWidth, displayHeight) {
        overlay.innerHTML = '';

        const naturalWidth = video.videoWidth;
        const naturalHeight = video.videoHeight;
        const scaleX = naturalWidth / displayWidth;
        const scaleY = naturalHeight / displayHeight;
        const naturalClickX = clickX * scaleX;
        const naturalClickY = clickY * scaleY;

        for (let det of detections) {
            const [x1, y1, x2, y2] = det.bbox;

            if (naturalClickX >= x1 && naturalClickX <= x2 &&
                naturalClickY >= y1 && naturalClickY <= y2) {

                const cxDisplay = ((x1 + x2) / 2) / scaleX;
                const cyDisplay = ((y1 + y2) / 2) / scaleY;

                drawDotAndLabel(cxDisplay, cyDisplay, det.category, det.object);
                liveStatus.innerText = `✅ Detected: ${det.object} (${det.category})`;

                // ✅ NEW: Display disposal info (with proper <br> formatting)
                if (det.disposalInfoHTML) {
                    showDisposalInfo(det.disposalInfoHTML);
                }
                return;
            }
        }

        liveStatus.innerText = "🔍 No object detected at this location.";
    }

    // ✅ Draw dot and label at DISPLAY coordinates
    function drawDotAndLabel(x, y, category, objectName = "Unknown Object") {
        overlay.innerHTML = '';

        const dot = document.createElement('div');
        dot.className = 'dot';
        dot.style.left = `${x}px`;
        dot.style.top = `${y}px`;
        dot.style.backgroundColor = colorMap[category] || '#FFFFFF';
        overlay.appendChild(dot);

        const label = document.createElement('div');
        label.className = 'label';
        label.style.left = `${x}px`;
        label.style.top = `${y}px`;
        label.innerText = `${objectName} (${category})`;
        overlay.appendChild(label);
    }

    // ✅ Render disposal info in clean HTML format (supports <br>)
    function showDisposalInfo(htmlContent) {
        const disposalDiv = document.getElementById("disposalInfo");
        if (disposalDiv) {
            disposalDiv.innerHTML = htmlContent; // no escaping, supports <br>
            disposalDiv.classList.remove("d-none");
        }
    }

    // ✅ CLEANUP
    window.addEventListener('beforeunload', () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
    });

    console.log("✅ Ready for action!");
});
