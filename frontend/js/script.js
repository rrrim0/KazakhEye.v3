document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('.file-input');
    const uploadLabel  = document.querySelector('.upload-label');
    const uploadBtn = document.querySelector('.upload-btn');
    const loader = document.querySelector('.loader');
    const resultSection = document.querySelector('.result-section');
    const imagePreview = document.querySelector('.uploaded-image');
    const plateNumberField = document.getElementById('plate-number');
    const regionField = document.getElementById('region');
    const imageIcon = document.querySelector('.image-icon')

    uploadBtn.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];
        uploadLabel.textContent = file.name
        if (file) {
            loader.style.display = 'block';
            resultSection.style.visibility = 'hidden';

            const formData = new FormData();
            formData.append('photo', file);

            fetch('http://127.0.0.1:49501/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loader.style.display = 'none';
                imageIcon.style.display = 'none';

                const reader = new FileReader();
                reader.onload = function (e) {
                    imagePreview.src = e.target.result;
                    resultSection.style.visibility = 'visible';
                };
                reader.readAsDataURL(file);

                plateNumberField.textContent = data.plate || 'N/A';
                regionField.textContent = data.region || 'N/A';
            })
            .catch(error => {
                console.error('Error:', error);
                loader.style.display = 'none';
                alert('An error occurred while processing the image.');
            });
        }
    });
});
