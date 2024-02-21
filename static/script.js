document.addEventListener('DOMContentLoaded', function() {
    fetchModels();
});

function fetchModels() {
    fetch('/models')
        .then(response => response.json())
        .then(models => {
            const modelSelect = document.getElementById('modelSelect');
            models.forEach(model => {
                let option = new Option(model, model);
                modelSelect.add(option);
            });
        });
}

function fetchFiles() {
    const model = document.getElementById('modelSelect').value;
    fetch(`/models/${model}/files`)
        .then(response => response.json())
        .then(files => {
            const fileSelect = document.getElementById('fileSelect');
            fileSelect.innerHTML = '<option>Select a file</option>'; // Clear previous options
            files.forEach(file => {
                let option = new Option(file, file);
                fileSelect.add(option);
            });
        });
}

function fetchFileContent() {
    const model = document.getElementById('modelSelect').value;
    const filename = document.getElementById('fileSelect').value;
    fetch(`/models/${model}/files/${filename}`)
        .then(response => response.json())
        .then(jsonData => {
            const contentDisplay = document.getElementById('contentDisplay');
            contentDisplay.innerHTML = ''; // Clear previous content
            
            // Create a table and append it to contentDisplay
            const table = document.createElement('table');
            table.setAttribute('id', 'contentTable');
            contentDisplay.appendChild(table);
            
            // Assuming jsonData is an array of objects
            if (jsonData.length > 0) {
                // Create table headers based on keys of the first object
                const thead = document.createElement('thead');
                table.appendChild(thead);
                const headerRow = document.createElement('tr');
                thead.appendChild(headerRow);
                Object.keys(jsonData[0]).forEach(key => {
                    const th = document.createElement('th');
                    th.textContent = key;
                    headerRow.appendChild(th);
                });
                
                // Populate table rows
                const tbody = document.createElement('tbody');
                table.appendChild(tbody);
                jsonData.forEach(item => {
                    const row = document.createElement('tr');
                    tbody.appendChild(row);
                    Object.values(item).forEach(val => {
                        const td = document.createElement('td');
                        td.textContent = val;
                        row.appendChild(td);
                    });
                });
            }
        });
}

