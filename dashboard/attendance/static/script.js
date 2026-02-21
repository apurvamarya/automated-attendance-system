let attendanceData = [];

fetch("/attendance")
    .then(response => response.json())
    .then(data => {
        attendanceData = data;
        renderTable(data);
        renderSummaryTable(data);
    });

function renderTable(data) {
    const tableBody = document.querySelector("#attendanceTable tbody");
    tableBody.innerHTML = "";

    data.forEach(record => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${record.Name}</td>
            <td>${record.Date}</td>
            <td>${record.Time}</td>
            <td>${record.Status}</td>
        `;
        tableBody.appendChild(row);
    });
}

function renderSummaryTable(data) {
    const summary = {};

    data.forEach(r => {
        summary[r.Name] = (summary[r.Name] || 0) + 1;
    });

    const tableBody = document.querySelector("#summaryTable tbody");
    tableBody.innerHTML = "";

    for (let name in summary) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${name}</td>
            <td>${summary[name]}</td>
        `;
        tableBody.appendChild(row);
    }
}


document.getElementById("nameFilter").addEventListener("input", filterData);
document.getElementById("dateFilter").addEventListener("change", filterData);

function filterData() {
    const nameValue = document.getElementById("nameFilter").value.toLowerCase();
    const dateValue = document.getElementById("dateFilter").value;

    const filtered = attendanceData.filter(record => {
        const nameMatch = record.Name.toLowerCase().includes(nameValue);
        const dateMatch = dateValue === "" || record.Date === dateValue;
        return nameMatch && dateMatch;
    });

    renderTable(filtered);
    renderSummaryTable(filtered);
}
