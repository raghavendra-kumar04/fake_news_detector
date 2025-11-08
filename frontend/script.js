document.addEventListener('DOMContentLoaded', () => {
    const detectButton = document.getElementById('detectButton');
    const inputText = document.getElementById('inputText');
    const isTitleCheckbox = document.getElementById('isTitle');
    const resultDiv = document.getElementById('result');
    const spinner = document.getElementById('spinner');
    const historyContent = document.getElementById('historyContent');
    const historyToggle = document.getElementById('historyToggle');
    const themeToggle = document.getElementById('themeToggle');

    // Load dark mode preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
    }

    // Toggle dark mode
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    });

    // Toggle history collapse
    historyToggle.addEventListener('click', () => {
        const history = document.getElementById('history');
        history.classList.toggle('collapsed');
    });

    // Load history from localStorage
    const loadHistory = () => {
        const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
        historyContent.innerHTML = '';
        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.textContent = `${item.text}: ${item.result} (Confidence: ${item.confidence}, ${item.type})`;
            historyContent.appendChild(div);
        });
    };

    // Save prediction to history
    const saveToHistory = (text, result, confidence, type) => {
        const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
        history.unshift({ text: text.slice(0, 50) + '...', result, confidence, type });
        if (history.length > 10) history.pop();
        localStorage.setItem('predictionHistory', JSON.stringify(history));
        loadHistory();
    };

    loadHistory();

    detectButton.addEventListener('click', async () => {
        const text = inputText.value.trim();
        const isTitle = isTitleCheckbox.checked;
        console.log('Sending request:', { text, is_title: isTitle });  // Debug log
        if (text.length < 10 && isTitle) {
            resultDiv.textContent = 'Please enter at least 10 characters for a title.';
            resultDiv.classList.add('error');
            resultDiv.classList.remove('real', 'fake', 'show');
            setTimeout(() => resultDiv.classList.add('show'), 10);
            return;
        } else if (text.length < 50 && !isTitle) {
            resultDiv.textContent = 'Please enter at least 50 characters for an article.';
            resultDiv.classList.add('error');
            resultDiv.classList.remove('real', 'fake', 'show');
            setTimeout(() => resultDiv.classList.add('show'), 10);
            return;
        }

        resultDiv.textContent = '';
        resultDiv.classList.remove('error', 'real', 'fake', 'show');
        spinner.style.display = 'block';
        detectButton.disabled = true;

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, is_title: isTitle })
            });

            spinner.style.display = 'none';
            detectButton.disabled = false;

            if (!response.ok) {
                const errorText = await response.text();
                console.log('Error response:', errorText);  // Debug log
                throw new Error(`HTTP error! Status: ${response.status}, Message: ${errorText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);  // Debug log

            if (data.error) {
                resultDiv.textContent = data.error;
                resultDiv.classList.add('error', 'show');
            } else {
                resultDiv.textContent = `Result: ${data.result} (Confidence: ${data.confidence})`;
                resultDiv.classList.add(data.result.toLowerCase(), 'show');
                saveToHistory(text, data.result, data.confidence, isTitle ? 'Title' : 'Article');
            }
        } catch (error) {
            spinner.style.display = 'none';
            detectButton.disabled = false;
            resultDiv.textContent = `Error: ${error.message}`;
            resultDiv.classList.add('error', 'show');
            console.error('Fetch error:', error);  // Debug log
        }
    });

    // Keyboard navigation
    inputText.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            detectButton.click();
        }
    });
});