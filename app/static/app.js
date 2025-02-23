function showLoading(target) {
    document.getElementById(target).innerHTML = `<p style="color: blue;">Đang kiểm tra... ⏳</p>`;
}

function checkURL() {
    const url = document.getElementById("urlInput").value;
    if (!url) {
        alert("Please enter a URL!");
        return;
    }

    showLoading("urlResult");

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        document.getElementById("urlResult").innerHTML = `
            <p><strong>URL:</strong> ${data.url}</p>
            <p><strong>Random Forest Confidence:</strong> ${data.rf_confidence}</p>
            <p><strong>SVM Confidence:</strong> ${data.svm_confidence}</p>
            <p><strong>Ensemble Confidence:</strong> ${data.ensemble_confidence}</p>
            <p><strong>Final Decision:</strong> <span style="color: ${data.result === 'Phishing' ? 'red' : 'green'}">${data.result}</span></p>
        `;
    })
    .catch(error => console.error("Error:", error));
}

function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    showLoading("imageResult");

    const formData = new FormData();
    formData.append("file", file);

    // Hiển thị ảnh ngay khi chọn
    const reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById("preview").innerHTML = `
            <p><strong>Uploaded Image:</strong></p>
            <img src="${e.target.result}" alt="Uploaded Image" style="max-width: 300px; border: 2px solid #ddd; padding: 5px; border-radius: 8px;">
        `;
    };
    reader.readAsDataURL(file);

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        document.getElementById("imageResult").innerHTML = `
            <p><strong>Random Forest Confidence:</strong> ${data.rf_confidence}</p>
            <p><strong>CNN Confidence:</strong> ${data.cnn_confidence}</p>
            <p><strong>Ensemble Confidence:</strong> ${data.ensemble_confidence}</p>
            <p><strong>Final Decision:</strong> <span style="color: ${data.result === 'Phishing' ? 'red' : 'green'}">${data.result}</span></p>
        `;
    })
    .catch(error => console.error("Error:", error));
}
