const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const detectBtn = document.getElementById("detectBtn");
const resultDiv = document.getElementById("result");
let selectedImage = null;

// Preview uploaded image
fileInput.addEventListener("change", (e) => {
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

// Detect button click
detectBtn.addEventListener("click", async () => {
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
