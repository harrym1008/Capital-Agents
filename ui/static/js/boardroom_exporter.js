/**
 * boardroom_exporter.js
 * Standalone Boardroom HTML Report Exporter for CapitalAgents.
 * Captures the current DOM state of a completed boardroom simulation,
 * sanitises live dependencies, inlines stylesheets & assets, sets tool/thinking
 * blocks to collapsed by default, and downloads a 100% self-contained HTML file.
 */

const BoardroomExporter = (function () {
    let lastSimTotalTime = "";

    function getCleanInitialCapital() {
        const val = document.getElementById("initialCapitalInput")?.value;
        const num = parseFloat(String(val).replace(/[^0-9.]/g, ""));
        return isNaN(num) || num <= 0 ? 100000 : num;
    }

    function formatOrdinalDate(dateInput) {
        if (!dateInput) {
            dateInput = new Date();
        }
        let year, monthIdx, day;
        if (typeof dateInput === "string" && dateInput.match(/^\d{4}-\d{2}-\d{2}/)) {
            const parts = dateInput.split("T")[0].split("-");
            year = parseInt(parts[0], 10);
            monthIdx = parseInt(parts[1], 10) - 1;
            day = parseInt(parts[2], 10);
        } else {
            const d = (dateInput instanceof Date) ? dateInput : new Date(dateInput);
            const validDate = isNaN(d.getTime()) ? new Date() : d;
            year = validDate.getFullYear();
            monthIdx = validDate.getMonth();
            day = validDate.getDate();
        }

        const months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ];
        const monthName = months[monthIdx] || "";

        let suffix = "th";
        const lastTwo = day % 100;
        if (lastTwo < 11 || lastTwo > 13) {
            const lastDigit = day % 10;
            if (lastDigit === 1) suffix = "st";
            else if (lastDigit === 2) suffix = "nd";
            else if (lastDigit === 3) suffix = "rd";
        }
        return `${day}${suffix} ${monthName} ${year}`;
    }

    function formatOrdinalDateTime(dateObj) {
        const d = dateObj || new Date();
        const datePart = formatOrdinalDate(d);
        const hours = String(d.getHours()).padStart(2, "0");
        const minutes = String(d.getMinutes()).padStart(2, "0");
        return `${datePart}, ${hours}:${minutes}`;
    }

    function isElementVisible(el) {
        if (!el) return false;
        if (el.style && el.style.display === "none") return false;
        if (el.hasAttribute("hidden")) return false;
        const computed = window.getComputedStyle(el);
        return computed.display !== "none" && computed.visibility !== "hidden";
    }

    function extractReportMetadata() {
        // Determine target ticker or scenario title
        const tickerInput = document.getElementById("tickerInput");
        let target = "";
        let isSingleEquity = false;

        if (tickerInput && isElementVisible(tickerInput)) {
            target = tickerInput.value.trim().toUpperCase() || "Unknown Ticker";
            isSingleEquity = true;
        } else if (document.getElementById("initialCapitalInput")) {
            target = "Portfolio Creation";
        } else {
            target = "Boardroom Evaluation";
        }

        const title = `Boardroom Report - ${target}`;
        const todayStr = new Date().toISOString().split("T")[0];
        const filename = `Boardroom_Report_${target.replace(/[^a-zA-Z0-9_-]/g, "_")}_${todayStr}.html`;

        // Extract visible form parameters
        const params = [];

        // 1. Date Produced with time
        params.push({ key: "Date Produced", value: formatOrdinalDateTime(new Date()) });

        // 2. Sim Date or Date of Analysis
        const simCheckbox = document.getElementById("simulatedDateCheckbox");
        const simDateInput = document.getElementById("simulatedDateInput");
        if (simCheckbox && simCheckbox.checked && simDateInput && simDateInput.value) {
            params.push({ key: "Sim Date", value: formatOrdinalDate(simDateInput.value) });
        } else {
            params.push({ key: "Date of Analysis", value: formatOrdinalDate(new Date()) });
        }

        // 3. Extract other visible form controls
        const configGroups = document.querySelectorAll(".config-bar .config-group");
        configGroups.forEach(group => {
            if (!isElementVisible(group)) return;

            // Skip groups that contain only action buttons (like sector modal button)
            if (group.querySelector("button") && !group.querySelector("input, select")) {
                return;
            }

            // Skip simulated date checkbox group (already extracted above)
            if (group.querySelector("#simulatedDateCheckbox") || group.querySelector("#simulatedDateInput")) {
                return;
            }

            // Skip allocation bias slider container group
            if (group.querySelector("#allocationBiasSlider")) {
                return;
            }

            const allocBiasCheck = group.querySelector("#allocationBiasCheckbox");
            if (allocBiasCheck) {
                if (allocBiasCheck.checked) {
                    const slider = document.getElementById("allocationBiasSlider");
                    const val = slider ? parseInt(slider.value, 10) : 1;
                    const biasLabels = {
                        1: "Maximum Growth Bias",
                        2: "Moderate Growth Bias",
                        3: "Minor Growth Bias",
                        4: "Minor Defensive Bias",
                        5: "Moderate Defensive Bias",
                        6: "Maximum Defensive Bias"
                    };
                    const biasLabel = biasLabels[val] || "Maximum Growth Bias";
                    params.push({ key: "Allocation Bias", value: biasLabel });
                }
                return;
            }

            const labelEl = group.querySelector("label");
            const labelText = labelEl ? labelEl.innerText.replace(/:$/, "").trim() : "";

            const selectEl = group.querySelector("select");
            const textOrNumInput = group.querySelector("input[type='text'], input[type='number']");

            if (selectEl) {
                const optText = selectEl.options[selectEl.selectedIndex]?.text || selectEl.value;
                params.push({ key: labelText || "Option", value: optText });
            } else if (textOrNumInput) {
                let val = textOrNumInput.value.trim();
                if (!val && textOrNumInput.placeholder) {
                    val = textOrNumInput.placeholder;
                }
                if (textOrNumInput.id === "initialCapitalInput") {
                    const cap = getCleanInitialCapital();
                    val = `$${cap.toLocaleString("en-US")}`;
                } else if (textOrNumInput.id === "maxSectorAllocationInput" || textOrNumInput.id === "maxStockAllocationInput") {
                    if (val && !val.endsWith("%")) val = `${val}%`;
                }
                if (val) {
                    params.push({ key: labelText || "Param", value: val });
                }
            }
        });

        return { target, title, filename, params, isSingleEquity };
    }

    async function collectAllStylesheets() {
        let fullCss = "";

        // Collect all external stylesheet links and inline style tags
        const styleNodes = Array.from(document.querySelectorAll('link[rel="stylesheet"], style'));

        for (const node of styleNodes) {
            if (node.tagName.toLowerCase() === "style") {
                fullCss += `\n/* Inline Style */\n${node.innerHTML}\n`;
            } else if (node.tagName.toLowerCase() === "link" && node.href) {
                try {
                    const res = await fetch(node.href);
                    if (res.ok) {
                        const cssText = await res.text();
                        fullCss += `\n/* From ${node.href} */\n${cssText}\n`;
                    }
                } catch (e) {
                    console.warn("Could not inline stylesheet:", node.href, e);
                }
            }
        }

        // Add report specific classes & overrides
        fullCss += `
            /* Standalone Report Styling Overrides */
            body {
                height: 100vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                background-color: #f8fafc;
                margin: 0;
            }
            .report-header {
                background-color: #0f172a;
                color: #f8fafc;
                padding: 14px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 20px;
                border-bottom: 1px solid #1e293b;
                box-sizing: border-box;
            }
            .report-header-brand {
                padding-right: 20px;
            }
            .report-header-center {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                flex: 1;
            }
            .report-header-title {
                font-size: 18px;
                font-weight: 700;
                color: #ffffff;
                margin: 0 0 4px 0;
            }
            .report-params-list {
                display: flex;
                gap: 6px 18px;
                align-items: center;
                justify-content: center;
                font-size: 13px;
                color: #cbd5e1;
                flex-wrap: wrap;
            }
            .report-param-item strong {
                color: #94a3b8;
                font-weight: 600;
                margin-right: 4px;
            }
            .report-header-right {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                font-size: 12px;
                color: #cbd5e1;
                text-align: right;
                white-space: nowrap;
                gap: 3px;
            }
            .report-meta-label {
                color: #94a3b8;
                font-weight: 600;
                margin-right: 4px;
            }
            .report-meta-value {
                color: #ffffff;
                font-weight: 600;
            }
            .stage-item.viewing {
                color: #2563eb;
                font-weight: 700;
            }
            .stage-item.viewing::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background-color: #2563eb;
            }
            .collapsible-header {
                cursor: pointer;
                user-select: none;
            }
        `;

        return fullCss;
    }

    function freezeCanvases(targetClone, originalRoot) {
        const origCanvases = Array.from(originalRoot.querySelectorAll("canvas"));
        const targetCanvases = Array.from(targetClone.querySelectorAll("canvas"));

        origCanvases.forEach((origCanvas, idx) => {
            try {
                let targetCanvas = null;
                if (origCanvas.id) {
                    targetCanvas = targetClone.querySelector(`#${origCanvas.id}`);
                }
                if (!targetCanvas && targetCanvases[idx]) {
                    targetCanvas = targetCanvases[idx];
                }

                if (!targetCanvas) return;

                // Check visibility of original canvas
                const isExplicitlyHidden = origCanvas.style.display === "none" || 
                                           origCanvas.hasAttribute("hidden") || 
                                           (origCanvas.getAttribute("style") && origCanvas.getAttribute("style").includes("display: none")) ||
                                           targetCanvas.style.display === "none" ||
                                           (targetCanvas.getAttribute("style") && targetCanvas.getAttribute("style").includes("display: none"));

                if (isExplicitlyHidden) {
                    targetCanvas.remove();
                    return;
                }

                const dataUrl = origCanvas.toDataURL("image/png");
                if (!dataUrl || dataUrl === "data:," || dataUrl.length < 50) {
                    targetCanvas.remove();
                    return;
                }

                const img = document.createElement("img");
                img.src = dataUrl;
                img.alt = origCanvas.id || "Chart Snapshot";
                if (origCanvas.id) img.id = origCanvas.id + "_img";

                // Ensure snapshot preserves natural responsive dimensions without overflowing container
                img.style.cssText = "max-width: 100%; width: 100%; height: 100%; max-height: 100%; display: block; margin: 0 auto; object-fit: contain;";

                if (targetCanvas.parentNode) {
                    targetCanvas.parentNode.replaceChild(img, targetCanvas);
                }
            } catch (err) {
                console.warn("Could not freeze canvas:", origCanvas, err);
            }
        });
    }

    function buildOfflineScript(sources) {
        const serializedSources = JSON.stringify(sources || []);
        return `
        <script>
            var capturedSources = ${serializedSources};

            function escapeHtml(str) {
                if (str === null || str === undefined) return "";
                return String(str)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            function showStageWorkspace(stageNum) {
                document.querySelectorAll('.stage-item').forEach(function(item) {
                    item.classList.remove('viewing');
                });
                var tab = document.getElementById('stageStep-' + stageNum);
                if (tab) tab.classList.add('viewing');

                document.querySelectorAll('.stage-workspace').forEach(function(ws) {
                    ws.style.display = 'none';
                });
                var targetWs = document.getElementById('stageWorkspace-' + stageNum);
                if (targetWs) {
                    targetWs.style.display = 'flex';
                }
            }

            function toggleSidebar() {
                var sidebar = document.getElementById('summarySidebar');
                if (sidebar) sidebar.classList.toggle('collapsed');
            }

            function switchSidebarTab(tabName) {
                document.querySelectorAll('.sidebar-tab-item').forEach(function(tab) {
                    tab.classList.remove('viewing');
                });
                document.querySelectorAll('.sidebar-pane').forEach(function(pane) {
                    pane.style.display = 'none';
                });
                var selectedTab = document.getElementById('sidebarTab-' + tabName);
                var selectedPane = document.getElementById('sidebarPane-' + tabName);
                if (selectedTab) selectedTab.classList.add('viewing');
                if (selectedPane) selectedPane.style.display = 'flex';
            }

            function openSidebar(tabName) {
                var sidebar = document.getElementById('summarySidebar');
                if (sidebar && sidebar.classList.contains('collapsed')) {
                    sidebar.classList.remove('collapsed');
                }
                if (tabName) {
                    switchSidebarTab(tabName);
                }
            }

            function openSourceModal(citationNumber) {
                var source = capturedSources.find(function(s) { return s.citationNumber === citationNumber; });
                if (!source) return;

                var modalOverlay = document.getElementById('sourceModalOverlay');
                if (!modalOverlay) {
                    modalOverlay = document.createElement('div');
                    modalOverlay.id = 'sourceModalOverlay';
                    modalOverlay.className = 'source-modal-overlay';
                    modalOverlay.onclick = function(e) {
                        if (e.target === modalOverlay) closeSourceModal();
                    };
                    document.body.appendChild(modalOverlay);
                }

                var formattedArgs = JSON.stringify(source.args || {}, null, 2).replace(/\\\\r\\\\n/g, '\\n').replace(/\\\\n/g, '\\n');
                var formattedResult = typeof source.result === 'string'
                    ? source.result
                    : JSON.stringify(source.result || {}, null, 2);
                formattedResult = formattedResult.replace(/\\\\r\\\\n/g, '\\n').replace(/\\\\n/g, '\\n');

                modalOverlay.innerHTML =
                    '<div class="source-modal-window" onclick="event.stopPropagation()">' +
                        '<div class="source-modal-header">' +
                            '<div class="source-modal-title">' +
                                '<span>[' + source.citationNumber + ']</span>' +
                                '<span>' + escapeHtml(source.toolName || 'Tool Call Result') + '</span>' +
                            '</div>' +
                            '<button type="button" class="modal-close-btn" onclick="BoardroomCore.closeSourceModal()" title="Close">&times;</button>' +
                        '</div>' +
                        '<div class="source-modal-body">' +
                            '<div>' +
                                '<div class="source-modal-section-title">Arguments:</div>' +
                                '<pre class="source-modal-args-pre">' + escapeHtml(formattedArgs) + '</pre>' +
                            '</div>' +
                            '<div class="source-modal-result-container">' +
                                '<div class="source-modal-section-title">Tool Output Result:</div>' +
                                '<pre class="source-modal-pre source-modal-result-pre">' + escapeHtml(formattedResult) + '</pre>' +
                            '</div>' +
                        '</div>' +
                        '<div class="source-modal-footer">' +
                            '<button class="run-btn" onclick="BoardroomCore.closeSourceModal()">Close</button>' +
                        '</div>' +
                    '</div>';
                modalOverlay.style.display = 'flex';
            }

            function closeSourceModal() {
                var modalOverlay = document.getElementById('sourceModalOverlay');
                if (modalOverlay) {
                    modalOverlay.style.display = 'none';
                }
            }

            function showSourceCitation(citationNumber, subIndex) {
                openSidebar('sources');
                setTimeout(function() {
                    document.querySelectorAll('.sources-table tr.source-highlight').forEach(function(r) {
                        r.classList.remove('source-highlight');
                    });

                    var row = document.getElementById('source-row-' + citationNumber);
                    if (row) {
                        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        row.classList.remove('source-highlight');
                        void row.offsetWidth;
                        row.classList.add('source-highlight');
                        row.addEventListener('animationend', function() {
                            row.classList.remove('source-highlight');
                        }, { once: true });
                    }
                }, 50);
            }

            window.BoardroomCore = {
                openSourceModal: openSourceModal,
                closeSourceModal: closeSourceModal,
                showSourceCitation: showSourceCitation,
                showStageWorkspace: showStageWorkspace,
                toggleSidebar: toggleSidebar,
                switchSidebarTab: switchSidebarTab,
                openSidebar: openSidebar,
                getSources: function() { return capturedSources; }
            };
            window.showStageWorkspace = showStageWorkspace;
            window.toggleSidebar = toggleSidebar;
            window.switchSidebarTab = switchSidebarTab;
            window.openSourceModal = openSourceModal;
            window.closeSourceModal = closeSourceModal;
            window.showSourceCitation = showSourceCitation;
            window.openSidebar = openSidebar;

            document.addEventListener('DOMContentLoaded', function() {
                document.body.addEventListener('click', function(e) {
                    var header = e.target.closest('.collapsible-header');
                    if (header) {
                        var block = header.closest('.collapsible-block');
                        if (block) {
                            block.classList.toggle('collapsed');
                        }
                    }
                });
            });
        <\/script>
        `;
    }

    async function generateSelfContainedHtml() {
        const meta = extractReportMetadata();
        const inlinedCss = await collectAllStylesheets();

        const sources = (window.BoardroomCore && window.BoardroomCore.getSources)
            ? window.BoardroomCore.getSources()
            : [];

        // Retrieve server details and simulation runtime
        let serverName = "Unknown";
        let modelName = "";
        let timeTaken = lastSimTotalTime;

        if (!timeTaken) {
            const finalTimeVal = document.getElementById("finalTimeReportValue");
            if (finalTimeVal && finalTimeVal.innerText) {
                timeTaken = finalTimeVal.innerText.replace(/^Time Taken:\s*/i, "").trim();
            }
        }
        if (!timeTaken) {
            const timerEl = document.getElementById("simulationTimer");
            if (timerEl && timerEl.innerText && timerEl.innerText !== "0 mins 0.00 secs") {
                timeTaken = timerEl.innerText.trim();
            }
        }
        if (!timeTaken) {
            timeTaken = "Completed";
        }

        if (window.latestServerStatus) {
            const sData = window.latestServerStatus;
            const p = (sData.provider || "").toLowerCase();
            if (p === "openrouter") serverName = "OpenRouter";
            else if (p === "llamacpp") serverName = "Llama.cpp";
            else if (p === "openaicompatible") serverName = "OpenAI Compatible";
            else if (sData.provider) serverName = sData.provider;

            if (sData.modelName && String(sData.modelName).trim() !== "") {
                modelName = String(sData.modelName).trim();
            }
        }

        if (serverName === "Unknown") {
            const sLabel = document.getElementById("serverStatusLabel");
            if (sLabel && sLabel.innerText) {
                serverName = sLabel.innerText.replace(/:$/, "").trim();
            }
        }

        let modelHtml = "";
        if (modelName) {
            modelHtml = `<div><span class="report-meta-label">Model:</span><span class="report-meta-value" title="${modelName}">${modelName}</span></div>`;
        }

        // 1. Build Clean Top Header
        let paramsHtml = meta.params.map(p => `
            <span class="report-param-item">
                <strong>${p.key}:</strong> ${p.value}
            </span>
        `).join("");

        const headerHtml = `
        <header class="report-header">
            <div class="report-header-brand">
                <h1>CapitalAgents</h1>
            </div>

            <div class="report-header-center">
                <h2 class="report-header-title">${meta.title}</h2>
                <div class="report-params-list">
                    ${paramsHtml}
                </div>
            </div>

            <div class="report-header-right">
                <div><span class="report-meta-label">Server:</span><span class="report-meta-value">${serverName}</span></div>
                ${modelHtml}
                <div><span class="report-meta-label">Time:</span><span class="report-meta-value">${timeTaken}</span></div>
            </div>
        </header>
        `;

        // 2. Clone & Prepare Stages Bar
        const origStagesBar = document.getElementById("stagesBar");
        let stagesBarHtml = "";
        if (origStagesBar) {
            const stagesClone = origStagesBar.cloneNode(true);

            // Remove Q&A stage tab if present
            const qaTab = stagesClone.querySelector("#stageStep-qa");
            if (qaTab) qaTab.remove();

            // Clear timer & download button, keep only Sidebar Toggle
            const stagesActions = stagesClone.querySelector("#stagesActions");
            if (stagesActions) {
                stagesActions.innerHTML = `
                    <button id="sidebarToggleBtn" class="grey-btn sidebar-toggle-btn" onclick="toggleSidebar()">Sidebar Toggle</button>
                `;
            }

            // Ensure stage tab onclick calls showStageWorkspace(num)
            const stageTabs = stagesClone.querySelectorAll(".stage-item");
            stageTabs.forEach((tab, index) => {
                tab.classList.remove("active");
                tab.classList.remove("disabled");
                tab.classList.add("completed");
                if (index === 0) {
                    tab.classList.add("viewing");
                } else {
                    tab.classList.remove("viewing");
                }

                const stageNumMatch = tab.id.match(/stageStep-(\d+)/);
                if (stageNumMatch) {
                    const sNum = stageNumMatch[1];
                    tab.setAttribute("onclick", `showStageWorkspace(${sNum})`);
                }
            });

            stagesBarHtml = stagesClone.outerHTML;
        }

        // 3. Clone & Prepare Workspaces Layout
        const origLayout = document.querySelector(".main-layout");
        let mainLayoutHtml = "";
        if (origLayout) {
            const layoutClone = origLayout.cloneNode(true);

            // Remove Q&A workspace if present
            const qaWorkspace = layoutClone.querySelector("#stageWorkspace-qa");
            if (qaWorkspace) qaWorkspace.remove();

            // Remove any action buttons inside panes (like Q&A delete turn buttons, interactive toggle buttons)
            layoutClone.querySelectorAll(".qa-delete-btn, .qa-user-pane, .modal-close-btn, #portfolioPieToggleContainer, #portfolioBacktestToggleContainer, .pie-toggle-btn").forEach(btn => btn.remove());

            // Freeze all canvases in the clone
            freezeCanvases(layoutClone, origLayout);

            // Collapse all thinking & tool call blocks by default
            layoutClone.querySelectorAll(".collapsible-block").forEach(block => {
                block.classList.add("collapsed");
            });

            // Ensure first stage workspace is visible and other stage workspaces are hidden
            const workspaces = layoutClone.querySelectorAll(".stage-workspace");
            workspaces.forEach((ws, idx) => {
                if (idx === 0) {
                    ws.style.display = "flex";
                } else {
                    ws.style.display = "none";
                }
            });

            // Ensure summary sidebar is present and tabs have clean onclick handlers
            const sumTab = layoutClone.querySelector("#sidebarTab-summaries");
            const srcTab = layoutClone.querySelector("#sidebarTab-sources");
            if (sumTab) sumTab.setAttribute("onclick", "switchSidebarTab('summaries')");
            if (srcTab) srcTab.setAttribute("onclick", "switchSidebarTab('sources')");

            mainLayoutHtml = layoutClone.outerHTML;
        }

        // 4. Assemble Complete Standalone HTML Document
        const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${meta.title} - CapitalAgents</title>
    <style>
        ${inlinedCss}
    </style>
</head>
<body>
    ${headerHtml}
    ${stagesBarHtml}
    ${mainLayoutHtml}
    ${buildOfflineScript(sources)}
</body>
</html>`;

        return { fullHtml, filename: meta.filename };
    }

    async function downloadHtmlReport() {
        try {
            const btn = document.getElementById("downloadReportBtn");
            if (btn) {
                btn.disabled = true;
                btn.innerText = "Exporting...";
            }

            const { fullHtml, filename } = await generateSelfContainedHtml();

            const blob = new Blob([fullHtml], { type: "text/html;charset=utf-8" });
            const url = URL.createObjectURL(blob);

            const downloadAnchor = document.createElement("a");
            downloadAnchor.href = url;
            downloadAnchor.download = filename;
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();

            setTimeout(() => {
                document.body.removeChild(downloadAnchor);
                URL.revokeObjectURL(url);
                if (btn) {
                    btn.disabled = false;
                    btn.innerText = "Download Report";
                }
            }, 300);
        } catch (e) {
            console.error("Error exporting self-contained HTML report:", e);
            alert("Failed to export HTML report: " + e.message);
            const btn = document.getElementById("downloadReportBtn");
            if (btn) {
                btn.disabled = false;
                btn.innerText = "Download Report";
            }
        }
    }

    function handleSimStart() {
        const downloadBtn = document.getElementById("downloadReportBtn");
        const timerEl = document.getElementById("simulationTimer");
        if (downloadBtn) {
            downloadBtn.style.display = "none";
        }
        if (timerEl) {
            timerEl.style.display = "inline-flex";
            timerEl.innerText = "0 mins 0.00 secs";
        }
    }

    function handleSimComplete(payload) {
        if (payload && payload.totalTime) {
            lastSimTotalTime = payload.totalTime;
        }

        const timerEl = document.getElementById("simulationTimer");
        const downloadBtn = document.getElementById("downloadReportBtn");

        if (timerEl) {
            timerEl.style.display = "none";
        }

        if (downloadBtn) {
            downloadBtn.style.display = "inline-flex";
            downloadBtn.innerText = "Download Report";
        }
    }

    return {
        handleSimStart,
        handleSimComplete,
        downloadHtmlReport,
        generateSelfContainedHtml
    };
})();

window.BoardroomExporter = BoardroomExporter;
