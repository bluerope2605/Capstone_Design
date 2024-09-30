document.addEventListener('DOMContentLoaded', (event) => {
    const webcam = document.getElementById('webcam');
    const capturedImage = document.getElementById('capturedImage');

    // 웹캠 스트림 설정
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            webcam.srcObject = stream;
        })
        .catch(error => {
            console.error('Error accessing webcam: ', error);
        });

    window.capture = () => {
        const canvas = document.createElement('canvas');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg');
        capturedImage.src = imageData;
        capturedImage.style.display = 'block';
    };

    window.upload = () => {
        const imageData = capturedImage.src;
        fetch('/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ webcamImage: imageData })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            if (data.processed_image_url) {
                window.open(data.processed_image_url, '_blank');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    };
});
