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

        // Nếu phát hiện QR Code
        if (data.qr_results) {
            let qrContent = `<p><strong>QR Codes Detected:</strong></p>`;
            data.qr_results.forEach((qr, index) => {
                qrContent += `
                    <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 8px;">
                        <p><strong>QR #${index + 1}:</strong> ${qr.qr_url}</p>
                        <p><strong>RF Confidence:</strong> ${qr.rf_confidence}</p>
                        <p><strong>SVM Confidence:</strong> ${qr.svm_confidence}</p>
                        <p><strong>Ensemble Confidence:</strong> ${qr.ensemble_confidence}</p>
                        <p><strong>Final Decision:</strong> <span style="color: ${qr.result === 'Phishing' ? 'red' : 'green'}">${qr.result}</span></p>
                    </div>
                `;
            });
            document.getElementById("imageResult").innerHTML = qrContent;
        } else {
            // Nếu không có QR, xử lý ảnh phishing
            document.getElementById("imageResult").innerHTML = `
                <p><strong>Random Forest Confidence:</strong> ${data.rf_confidence}</p>
                <p><strong>CNN Confidence:</strong> ${data.cnn_confidence}</p>
                <p><strong>Ensemble Confidence:</strong> ${data.ensemble_confidence}</p>
                <p><strong>Final Decision:</strong> <span style="color: ${data.result === 'Phishing' ? 'red' : 'green'}">${data.result}</span></p>
            `;
        }
    })
    .catch(error => console.error("Error:", error));
}
