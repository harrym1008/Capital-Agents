/**
 * boardroom_core.js
 * Universal Boardroom Client Engine for CapitalAgents.
 * Manages WebSocket streaming events, agent feeds, collapsible tool call blocks,
 * markdown parsing, stage navigation, execution timers, and generation settings.
 */

const BoardroomCore = (function () {
    let serverIsLoaded = false;
    let activeAgentPanes = {}; // Map of "stageNum_agentRole" -> { pane, feed }
    let activeBlocks = {};     // Map of "stageNum_agentRole" -> { type, element, block, rawText }
    let streamingArgs = {};    // Map of "stageNum_toolIndex" -> raw string
    let currentStageNumber = 0;
    let highestStageNumber = 0;
    let timerInterval = null;
    let currentServerProvider = "";
    let defaultStagesConfig = {};
    let getPayloadFn = null;
    let customStageLayoutHandler = null;
    const customEventListeners = {};

    const roleConfigs = {
        "Macro Analyst": { class: "cyan", initials: "MA", name: "Macro Analyst" },
        "Bullish Value Analyst": { class: "green", initials: "BL", name: "Bullish Analyst" },
        "Bearish Risk Analyst": { class: "red", initials: "BR", name: "Bearish Analyst" },
        "Aggressive Risk Analyst": { class: "yellow", initials: "AR", name: "Aggressive Risk" },
        "Conservative Risk Analyst": { class: "blue", initials: "CR", name: "Conservative Risk" },
        "Impartial Portfolio Manager": { class: "magenta", initials: "PM", name: "Portfolio Manager" },
        "One-Shot Analyst": { class: "cyan", initials: "OS", name: "One-Shot Analyst" },
        "Boardroom Spokesperson": { class: "cyan", initials: "SP", name: "Boardroom Spokesperson" }
    };

    function registerRole(roleName, config) {
        roleConfigs[roleName] = config;
    }

    function formatOrdinalDate(dateInput) {
        if (!dateInput) {
            dateInput = new Date().toISOString().split('T')[0];
        }
        let dateStr = String(dateInput).split('T')[0];
        const parts = dateStr.split('-');
        if (parts.length < 3) return dateStr;

        const year = parseInt(parts[0], 10);
        const monthIdx = parseInt(parts[1], 10) - 1;
        const day = parseInt(parts[2], 10);

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

    function extractTableColumns(rowStr) {
        if (!rowStr) return [];
        let content = rowStr.trim();
        if (content.startsWith('|')) content = content.slice(1);
        if (content.endsWith('|')) content = content.slice(0, -1);
        return content.split('|').map(c => c.trim());
    }

    function isTableSeparatorRow(rowStr) {
        if (!rowStr) return false;
        let content = rowStr.trim();
        if (content.startsWith('|')) content = content.slice(1);
        if (content.endsWith('|')) content = content.slice(0, -1);
        return content.length > 0 && /^[\s\-:|]+$/.test(content) && content.includes('-');
    }

    function replaceCitationTags(text) {
        if (!text) return "";

        function makeCitationButton(citeNumStr, subIndexStr) {
            const citeNum = parseInt(String(citeNumStr).trim(), 10);
            if (isNaN(citeNum)) return "";

            if (subIndexStr !== undefined && subIndexStr !== null && String(subIndexStr).trim() !== "") {
                const subIndex = parseInt(String(subIndexStr).trim(), 10);
                if (!isNaN(subIndex)) {
                    return `<button type="button" class="citation-ref-btn" onclick="BoardroomCore.showSourceCitation(${citeNum}, ${subIndex})" title="View Source Citation [${citeNum}, ${subIndex}]"><sup>[${citeNum}, ${subIndex}]</sup></button>`;
                }
            }
            return `<button type="button" class="citation-ref-btn" onclick="BoardroomCore.showSourceCitation(${citeNum})" title="View Source Citation [${citeNum}]"><sup>[${citeNum}]</sup></button>`;
        }

        // 1. Specific schema: <newsCitation>X:Y</newsCitation> or <newsCitation>X, Y</newsCitation>
        text = text.replace(/<newsCitation(?:\s+id=["']?([0-9,\s:]+)["']?)?>([\s\S]*?)<\/newsCitation>/gi, (match, idAttr, innerContent) => {
            const content = (innerContent && innerContent.trim()) ? innerContent.trim() : (idAttr || "");
            const pairMatch = content.match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = content.match(/\d+/g);
            if (numbers && numbers.length >= 2) {
                return makeCitationButton(numbers[0], numbers[1]);
            } else if (numbers && numbers.length === 1) {
                return makeCitationButton(numbers[0]);
            }
            return "";
        });
        text = text.replace(/<newsCitation\s+id=["']?([0-9,\s:]+)["']?\s*\/>/gi, (match, idStr) => {
            const pairMatch = idStr.trim().match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = idStr.match(/\d+/g);
            if (numbers && numbers.length >= 2) {
                return makeCitationButton(numbers[0], numbers[1]);
            } else if (numbers && numbers.length === 1) {
                return makeCitationButton(numbers[0]);
            }
            return "";
        });

        // 2. Specific schema: <toolCitation>X</toolCitation>
        text = text.replace(/<toolCitation(?:\s+id=["']?(\d+)["']?)?>([\s\S]*?)<\/toolCitation>/gi, (match, idAttr, innerContent) => {
            const content = (innerContent && innerContent.trim()) ? innerContent.trim() : (idAttr || "");
            const numbers = content.match(/\d+/g);
            if (!numbers || numbers.length === 0) return "";
            return numbers.map(n => makeCitationButton(n)).join("");
        });
        text = text.replace(/<toolCitation\s+id=["']?(\d+)["']?\s*\/>/gi, (match, id) => {
            return makeCitationButton(id);
        });

        // 3. Fallback/legacy: <citation>X</citation> or <citation>X, Y</citation> or <citation>X:Y</citation>
        text = text.replace(/<citation(?:\s+id=["']?(\d+)["']?)?>([\s\S]*?)<\/citation>/gi, (match, idAttr, innerContent) => {
            const content = (innerContent && innerContent.trim()) ? innerContent : (idAttr || "");
            const pairMatch = content.match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = content.match(/\d+/g);
            if (!numbers || numbers.length === 0) return "";
            return numbers.map(n => makeCitationButton(n)).join("");
        });

        // 4. Self-closing: <citation id="X" /> or <citation id="X, Y" />
        text = text.replace(/<citation\s+id=["']?([0-9,\s:]+)["']?\s*\/>/gi, (match, idStr) => {
            const pairMatch = idStr.trim().match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = idStr.match(/\d+/g);
            if (!numbers || numbers.length === 0) return "";
            return numbers.map(n => makeCitationButton(n)).join("");
        });

        // 5. Short variant: <cite>X, Y</cite> or <cite>X</cite>
        text = text.replace(/<cite(?:\s+id=["']?(\d+)["']?)?>([\s\S]*?)<\/cite>/gi, (match, idAttr, innerContent) => {
            const content = (innerContent && innerContent.trim()) ? innerContent : (idAttr || "");
            const pairMatch = content.match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = content.match(/\d+/g);
            if (!numbers || numbers.length === 0) return "";
            return numbers.map(n => makeCitationButton(n)).join("");
        });
        text = text.replace(/<cite\s+id=["']?([0-9,\s:]+)["']?\s*\/>/gi, (match, idStr) => {
            const pairMatch = idStr.trim().match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = idStr.match(/\d+/g);
            if (!numbers || numbers.length === 0) return "";
            return numbers.map(n => makeCitationButton(n)).join("");
        });

        // 6. Bracket variant: [citation: X, Y] or [cite: X, Y] or [citation: X]
        text = text.replace(/\[(?:citation|cite):\s*([0-9,\s:]+)\]/gi, (match, numbersStr) => {
            const pairMatch = numbersStr.trim().match(/^(\d+)\s*[,:]\s*(\d+)$/);
            if (pairMatch) {
                return makeCitationButton(pairMatch[1], pairMatch[2]);
            }
            const numbers = numbersStr.match(/\d+/g);
            if (!numbers || numbers.length === 0) return "";
            return numbers.map(n => makeCitationButton(n)).join("");
        });

        // 7. Streaming in-progress tags at end of string:
        text = text.replace(/<newsCitation(?:\s+id=["']?([0-9,\s:]+)["']?)?>\s*(\d+)\s*[,:]\s*(\d+)\s*$/i, (match, idAttr, num1, num2) => {
            return makeCitationButton(num1, num2);
        });
        text = text.replace(/<newsCitation(?:\s+id=["']?([0-9,\s:]+)["']?)?>\s*(\d+)\s*$/i, (match, idAttr, num) => {
            return makeCitationButton(num);
        });
        text = text.replace(/<toolCitation(?:\s+id=["']?(\d+)["']?)?>\s*(\d+)\s*$/i, (match, idAttr, num) => {
            const targetNum = num || idAttr;
            if (!targetNum) return "";
            return makeCitationButton(targetNum);
        });
        text = text.replace(/<(?:citation|cite)(?:\s+id=["']?(\d+)["']?)?>\s*(\d+)\s*[,:]\s*(\d+)\s*$/i, (match, idAttr, num1, num2) => {
            return makeCitationButton(num1, num2);
        });
        text = text.replace(/<(?:citation|cite)(?:\s+id=["']?(\d+)["']?)?>\s*(\d+)\s*$/i, (match, idAttr, num) => {
            const targetNum = num || idAttr;
            if (!targetNum) return "";
            return makeCitationButton(targetNum);
        });

        return text;
    }

    function parseMarkdown(text) {
        if (!text) return "";

        // 1. Parse tables
        const lines = text.split('\n');
        let inTable = false;
        let tableRows = [];
        let resultLines = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('|')) {
                inTable = true;
                tableRows.push(line);
            } else {
                if (inTable) {
                    resultLines.push(renderHtmlTable(tableRows));
                    tableRows = [];
                    inTable = false;
                }
                resultLines.push(line);
            }
        }
        if (inTable) {
            resultLines.push(renderHtmlTable(tableRows));
        }

        let formattedText = resultLines.join('\n');

        // 2. Parse bold: **text** or __text__
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/__(.*?)__/g, '<strong>$1</strong>');

        // 3. Parse italic: *text* or _text_
        formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
        formattedText = formattedText.replace(/_(.*?)_/g, '<em>$1</em>');

        // 4. Parse hashtag headings: # (H2) to ###### (H6)
        formattedText = formattedText.replace(/^######\s+(.*?)$/gm, '<h6 style="font-size: 15px; font-weight: bold;">$1</h6>');
        formattedText = formattedText.replace(/^#####\s+(.*?)$/gm, '<h6 style="font-size: 16px; font-weight: bold;">$1</h6>');
        formattedText = formattedText.replace(/^####\s+(.*?)$/gm, '<h5 style="font-size: 17px; font-weight: bold;">$1</h5>');
        formattedText = formattedText.replace(/^###\s+(.*?)$/gm, '<h4 style="font-size: 19px; font-weight: bold;">$1</h4>');
        formattedText = formattedText.replace(/^##\s+(.*?)$/gm, '<h3 style="font-size: 21px; font-weight: bold;">$1</h3>');
        formattedText = formattedText.replace(/^#\s+(.*?)$/gm, '<h2 style="font-size: 23px; font-weight: bold; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px;">$1</h2>');

        // 5. Parse horizontal rules: ---
        formattedText = formattedText.replace(/^---+\s*$/gm, '<hr style="border: none; border-top: 1.5px solid #cbd5e1; margin: 10px 0 3px 0;">');

        // 6. Parse backslash dollar signs: \$ -> $
        formattedText = formattedText.replace(/\\\$/g, '$');

        // 7. Parse source citations: <citation>X</citation> -> blue underlined superscript button [X]
        formattedText = replaceCitationTags(formattedText);

        return formattedText;
    }

    function renderHtmlTable(rows) {
        if (!rows || rows.length === 0) return "";

        let html = '<table style="border-collapse: collapse; width: 100%; margin: 10px 0; border: 2px solid #cbd5e1;">';

        // Header row
        const firstRowCols = extractTableColumns(rows[0]);
        if (firstRowCols.length === 0) return "";

        const numCols = firstRowCols.length;

        html += '<thead><tr style="background-color: #f1f5f9; border-bottom: 2px solid #cbd5e1; font-weight: bold;">';
        firstRowCols.forEach(col => {
            html += `<th style="padding: 8px; border: 1px solid #cbd5e1; text-align: left; font-weight: bold; font-size: 13px;">${col}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Data rows (skip the separator row if contains ---)
        const startIdx = (rows.length > 1 && isTableSeparatorRow(rows[1])) ? 2 : 1;

        for (let i = startIdx; i < rows.length; i++) {
            if (isTableSeparatorRow(rows[i])) continue;
            const cols = extractTableColumns(rows[i]);
            html += '<tr style="border-bottom: 1px solid #cbd5e1;">';
            const totalCols = Math.max(numCols, cols.length);
            for (let c = 0; c < totalCols; c++) {
                const colVal = c < cols.length ? cols[c] : "";
                html += `<td style="padding: 8px; border: 1px solid #cbd5e1; font-size: 13px;">${colVal}</td>`;
            }
            html += '</tr>';
        }

        html += '</tbody></table>';
        return html;
    }

    function updateGenerationSettingsUI(provider) {
        currentServerProvider = (provider || "").toLowerCase();
        const thinkingBudgetGroup = document.getElementById("thinkingBudgetGroup");
        if (thinkingBudgetGroup) {
            thinkingBudgetGroup.style.display = "flex";
        }
        const warningEl = document.getElementById("thinkingBudgetWarning");
        if (warningEl) {
            if (currentServerProvider === "openrouter" || currentServerProvider.includes("openai")) {
                warningEl.style.display = "block";
            } else {
                warningEl.style.display = "none";
            }
        }
    }

    function applyServerStatus(data) {
        if (!data) return;
        const isRunning = !!(data.running || data.llamacppRunning || data.openrouterRunning || data.openaiCompatibleRunning);
        const provider = data.provider || (data.openrouterRunning ? "openrouter" : (data.openaiCompatibleRunning ? "openaicompatible" : (data.llamacppRunning ? "llamacpp" : "")));
        currentServerProvider = (provider || "").toLowerCase();

        if (typeof updateServerStatus === "function") {
            updateServerStatus(isRunning, provider);
        } else if (typeof window.updateServerStatus === "function") {
            window.updateServerStatus(isRunning, provider);
        } else {
            setServerConnectedState(isRunning);
        }

        updateGenerationSettingsUI(currentServerProvider);
        if (data.costData) {
            if (typeof updateCostUI === "function") {
                updateCostUI(data.costData);
            } else if (typeof window.updateCostUI === "function") {
                window.updateCostUI(data.costData);
            }
        }
    }

    function checkServerStatus() {
        if (window.latestServerStatus) {
            applyServerStatus(window.latestServerStatus);
        } else if (typeof window.sendWsMessage === 'function') {
            window.sendWsMessage({ action: "get_server_status" });
        }
    }

    let simulationIsRunning = false;

    function setServerConnectedState(isConnected) {
        serverIsLoaded = isConnected;
        window.serverIsLoaded = isConnected;
        const runBtn = document.getElementById("runBtn");
        if (runBtn) {
            if (isConnected) {
                runBtn.disabled = simulationIsRunning;
                if (simulationIsRunning) {
                    runBtn.classList.add("disabled");
                    runBtn.style.display = "none";
                } else {
                    runBtn.classList.remove("disabled");
                    runBtn.style.display = "inline-block";
                }
                runBtn.title = "";
            } else {
                runBtn.disabled = true;
                runBtn.classList.add("disabled");
                runBtn.style.display = simulationIsRunning ? "none" : "inline-block";
                runBtn.title = "Please start and load an LLM model before running.";
            }
        }
    }

    function setControlsRunningState(isRunning) {
        simulationIsRunning = isRunning;
        const tickerInp = document.getElementById("tickerInput");
        if (tickerInp) tickerInp.disabled = isRunning;
        const modeSel = document.getElementById("modeSelect");
        if (modeSel) modeSel.disabled = isRunning;
        const horizonSel = document.getElementById("horizonSelect");
        if (horizonSel) horizonSel.disabled = isRunning;
        const simDateCheck = document.getElementById("simulatedDateCheckbox");
        if (simDateCheck) simDateCheck.disabled = isRunning;
        const simDateInp = document.getElementById("simulatedDateInput");
        if (simDateInp) simDateInp.disabled = isRunning;
        const openSettingsBtn = document.getElementById("openSettingsBtn");
        if (openSettingsBtn) openSettingsBtn.disabled = isRunning;

        // Portfolio Creation specific inputs if present
        const initCapInp = document.getElementById("initialCapitalInput");
        if (initCapInp) initCapInp.disabled = isRunning;
        const secCountInp = document.getElementById("targetSectorCountInput");
        if (secCountInp) secCountInp.disabled = isRunning;
        const secAllocInp = document.getElementById("maxSectorAllocationInput");
        if (secAllocInp) secAllocInp.disabled = isRunning;
        const stockCountInp = document.getElementById("targetStockCountInput");
        if (stockCountInp) stockCountInp.disabled = isRunning;
        const stockAllocInp = document.getElementById("maxStockAllocationInput");
        if (stockAllocInp) stockAllocInp.disabled = isRunning;
        const openSectorModalBtn = document.getElementById("openSectorModalBtn");
        if (openSectorModalBtn) openSectorModalBtn.disabled = isRunning;

        const runBtn = document.getElementById("runBtn");
        if (runBtn) {
            runBtn.disabled = isRunning || !serverIsLoaded;
            if (runBtn.disabled) {
                runBtn.classList.add("disabled");
            } else {
                runBtn.classList.remove("disabled");
            }
            runBtn.style.display = isRunning ? "none" : "inline-block";
        }

        const stopBtn = document.getElementById("stopBtn");
        if (stopBtn) {
            stopBtn.style.display = isRunning ? "inline-block" : "none";
            stopBtn.disabled = false;
        }
    }

    function startSimulationTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
        }
        const timerEl = document.getElementById("simulationTimer");
        if (timerEl) {
            timerEl.style.display = "inline-block";
            timerEl.innerText = "0 mins 0.00 secs";
        }
        const startTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsedMs = Date.now() - startTime;
            const elapsedSec = elapsedMs / 1000;
            const mins = Math.floor(elapsedSec / 60);
            const secs = elapsedSec % 60;
            if (timerEl) {
                timerEl.innerText = `${mins} mins ${secs.toFixed(2)} secs`;
            }
        }, 10);
    }

    function stopSimulationTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        const timerEl = document.getElementById("simulationTimer");
        if (timerEl) {
            timerEl.style.display = "none";
        }
    }

    function updateSummariesBtnState() {
        const btn = document.getElementById("sidebarToggleBtn") || document.getElementById("summariesBtn");
        if (!btn) return;
        if (highestStageNumber >= 1 || (capturedSources && capturedSources.length > 0)) {
            btn.disabled = false;
            btn.classList.remove("disabled");
        } else {
            btn.disabled = true;
            btn.classList.add("disabled");
        }
    }

    function initStagesBar(mode, stagesConfig) {
        if (stagesConfig) {
            defaultStagesConfig = stagesConfig;
        }
        const bar = document.getElementById("stagesScrollContainer");
        if (!bar) return;
        bar.innerHTML = "";

        const stages = (defaultStagesConfig && defaultStagesConfig[mode]) ? defaultStagesConfig[mode] : [];
        stages.forEach(stage => {
            const div = document.createElement("div");
            div.className = "stage-item";
            div.id = `stageStep-${stage.num}`;
            div.innerText = `Stage ${stage.num}: ${stage.name}`;

            div.onclick = () => {
                const workspace = document.getElementById(`stageWorkspace-${stage.num}`);
                if (workspace) {
                    showStageWorkspace(stage.num);
                }
            };
            bar.appendChild(div);
        });

        // Trigger custom stages hook if child template registered one
        emit("stagesBarInit", { mode, stages, bar });
    }

    function showStageWorkspace(stageNum) {
        const allWorkspaces = document.querySelectorAll(".stage-workspace");
        allWorkspaces.forEach(ws => ws.style.display = "none");

        const targetWorkspace = document.getElementById(`stageWorkspace-${stageNum}`);
        if (targetWorkspace) {
            targetWorkspace.style.display = "flex";
        }

        const allSteps = document.querySelectorAll(".stage-item");
        allSteps.forEach(step => step.classList.remove("viewing"));

        const targetStep = document.getElementById(`stageStep-${stageNum}`);
        if (targetStep) {
            targetStep.classList.add("viewing");
        }

        emit("stageViewChanged", { stageNum });
    }

    // Sources Manager UI State & Functions
    let capturedSources = [];

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function switchSidebarTab(tabName) {
        const summariesTab = document.getElementById("sidebarTab-summaries");
        const sourcesTab = document.getElementById("sidebarTab-sources");
        const summariesPane = document.getElementById("sidebarPane-summaries");
        const sourcesPane = document.getElementById("sidebarPane-sources");

        if (tabName === "sources") {
            if (summariesTab) summariesTab.classList.remove("viewing");
            if (sourcesTab) sourcesTab.classList.add("viewing");
            if (summariesPane) summariesPane.style.display = "none";
            if (sourcesPane) sourcesPane.style.display = "flex";
            document.querySelectorAll(".sources-table tr.source-highlight").forEach(r => {
                r.classList.remove("source-highlight");
            });
        } else {
            if (sourcesTab) sourcesTab.classList.remove("viewing");
            if (summariesTab) summariesTab.classList.add("viewing");
            if (sourcesPane) sourcesPane.style.display = "none";
            if (summariesPane) summariesPane.style.display = "flex";
        }
    }

    function openSidebar(defaultTab) {
        const sidebar = document.getElementById("summarySidebar");
        if (sidebar && sidebar.classList.contains("collapsed")) {
            sidebar.classList.remove("collapsed");
        }
        if (defaultTab) {
            switchSidebarTab(defaultTab);
        }
    }

    function ensureSourcesWorkspace() {
        // In the sidebar architecture, sourcesTable exists inside #sidebarPane-sources.
        // If needed for dynamic re-render, verify table body is present
        const tbody = document.getElementById("sourcesTableBody");
        if (!tbody) {
            const container = document.getElementById("sidebarSourcesContent");
            if (container) {
                container.innerHTML = `
                    <table class="sources-table" id="sourcesTable">
                        <thead>
                            <tr>
                                <th style="width: 60px; text-align: center;">Citation</th>
                                <th>Tool</th>
                                <th style="width: 80px; text-align: right;"></th>
                            </tr>
                        </thead>
                        <tbody id="sourcesTableBody">
                            <tr>
                                <td colspan="3">
                                    <div class="sources-empty-state">No sources yet</div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                `;
            }
        }
    }

    function resetSourcesUI() {
        capturedSources = [];
        const sourcesTab = document.getElementById("sidebarTab-sources");
        if (sourcesTab) {
            sourcesTab.innerText = "Sources Log";
        }
        const tbody = document.getElementById("sourcesTableBody");
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="3">
                        <div class="sources-empty-state">No sources yet</div>
                    </td>
                </tr>
            `;
        }
        closeSourceModal();
        updateSummariesBtnState();
    }

    function addSourceRecord(source) {
        if (!source || !source.citationNumber) return;
        ensureSourcesWorkspace();

        const existingIdx = capturedSources.findIndex(s => s.citationNumber === source.citationNumber);
        if (existingIdx >= 0) {
            capturedSources[existingIdx] = source;
        } else {
            capturedSources.push(source);
        }

        const sourcesTab = document.getElementById("sidebarTab-sources");
        if (sourcesTab) {
            sourcesTab.innerText = `Sources Log (${capturedSources.length})`;
        }

        renderSourcesTable();
        updateSummariesBtnState();
    }

    function renderSourcesTable() {
        const tbody = document.getElementById("sourcesTableBody");
        if (!tbody) return;

        if (capturedSources.length === 0) {
            resetSourcesUI();
            return;
        }

        tbody.innerHTML = capturedSources.map(s => {
            const hasArgs = s.args && Object.keys(s.args).length > 0;
            const argsStr = hasArgs
                ? Object.entries(s.args).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join(", ")
                : "";

            return `
                <tr id="source-row-${s.citationNumber}" data-citation="${s.citationNumber}">
                    <td style="text-align: center; vertical-align: top;">
                        <span class="source-citation-text">[${s.citationNumber}]</span>
                    </td>
                    <td style="vertical-align: top;">
                        <div class="source-tool-name">${escapeHtml(s.toolName || "")}</div>
                        ${hasArgs ? `<div class="source-args-text" title="${escapeHtml(argsStr)}">${escapeHtml(argsStr)}</div>` : ""}
                    </td>
                    <td style="text-align: right; vertical-align: top;">
                        <button class="source-view-btn" onclick="BoardroomCore.openSourceModal(${s.citationNumber})">View</button>
                    </td>
                </tr>
            `;
        }).join("");
    }

    function openSourceModal(citationNumber) {
        const source = capturedSources.find(s => s.citationNumber === citationNumber);
        if (!source) return;

        let modalOverlay = document.getElementById("sourceModalOverlay");
        if (!modalOverlay) {
            modalOverlay = document.createElement("div");
            modalOverlay.id = "sourceModalOverlay";
            modalOverlay.className = "source-modal-overlay";
            modalOverlay.onclick = (e) => {
                if (e.target === modalOverlay) closeSourceModal();
            };
            document.body.appendChild(modalOverlay);
        }

        const formattedArgs = JSON.stringify(source.args || {}, null, 2).replace(/\\r\\n/g, "\n").replace(/\\n/g, "\n");
        let formattedResult = typeof source.result === "string"
            ? source.result
            : JSON.stringify(source.result || {}, null, 2);
        formattedResult = formattedResult.replace(/\\r\\n/g, "\n").replace(/\\n/g, "\n");

        modalOverlay.innerHTML = `
            <div class="source-modal-window" onclick="event.stopPropagation()">
                <div class="source-modal-header">
                    <div class="source-modal-title">
                        <span>[${source.citationNumber}]</span>
                        <span>${escapeHtml(source.toolName || "Tool Call Result")}</span>
                    </div>
                    <button type="button" class="modal-close-btn" onclick="BoardroomCore.closeSourceModal()" title="Close">&times;</button>
                </div>
                <div class="source-modal-body">
                    <div>
                        <div class="source-modal-section-title">Arguments:</div>
                        <pre class="source-modal-args-pre">${escapeHtml(formattedArgs)}</pre>
                    </div>
                    <div class="source-modal-result-container">
                        <div class="source-modal-section-title">Tool Output Result:</div>
                        <pre class="source-modal-pre source-modal-result-pre">${escapeHtml(formattedResult)}</pre>
                    </div>
                </div>
                <div class="source-modal-footer">
                    <button class="run-btn" onclick="BoardroomCore.closeSourceModal()">Close</button>
                </div>
            </div>
        `;
        modalOverlay.style.display = "flex";
    }

    function closeSourceModal() {
        const modalOverlay = document.getElementById("sourceModalOverlay");
        if (modalOverlay) {
            modalOverlay.style.display = "none";
        }
    }

    function getNewsArticle(source, subIndex) {
        if (!source || !source.result) return null;
        const res = source.result;
        const news = Array.isArray(res.news) ? res.news : (Array.isArray(res) ? res : null);
        if (!news || news.length === 0) return null;

        // 1. Direct match by newsCitationNumber
        let article = news.find(a => a && a.newsCitationNumber === subIndex);
        if (article && article.url) return article;

        // 2. Direct match by index property
        article = news.find(a => a && a.index === subIndex);
        if (article && article.url) return article;

        // 3. If subIndex is 0, definitely the first article
        if (subIndex === 0) {
            return news[0];
        }

        // 4. 1-based indexing resolving to array position news[subIndex - 1]
        if (subIndex > 0 && news[subIndex - 1] && news[subIndex - 1].url) {
            return news[subIndex - 1];
        }

        // 5. 0-based array index news[subIndex]
        if (news[subIndex] && news[subIndex].url) return news[subIndex];

        return null;
    }

    function showSourceCitation(citationNumber, subIndex) {
        openSidebar("sources");
        setTimeout(() => {
            // Remove any existing highlights from all rows
            document.querySelectorAll(".sources-table tr.source-highlight").forEach(r => {
                r.classList.remove("source-highlight");
            });

            const row = document.getElementById(`source-row-${citationNumber}`);
            if (row) {
                row.scrollIntoView({ behavior: "smooth", block: "center" });
                row.classList.remove("source-highlight");
                void row.offsetWidth;
                row.classList.add("source-highlight");

                row.addEventListener("animationend", () => {
                    row.classList.remove("source-highlight");
                }, { once: true });
            }

            // If subIndex is specified, check if it references a news article with a URL
            if (subIndex !== undefined && subIndex !== null) {
                const source = capturedSources.find(s => s.citationNumber === citationNumber);
                const article = getNewsArticle(source, subIndex);
                if (article && article.url) {
                    const headline = article.headline || "News Article";
                    setTimeout(() => {
                        const proceed = window.confirm(`Do you want to open this article's page in a new tab?  It is called: '${headline}'`);
                        if (proceed) {
                            window.open(article.url, "_blank", "noopener,noreferrer");
                        }
                    }, 120);
                }
            }
        }, 50);
    }


    function setupStageLayout(stageNum, stageName, agents) {
        if (stageNum === 0) {
            // Ignore stage 0 layout setup so we stay on the final Decision Upload page!
            return;
        }

        currentStageNumber = stageNum;
        if (stageNum > highestStageNumber) {
            highestStageNumber = stageNum;
            updateSummariesBtnState();
        }

        const items = document.querySelectorAll(".stage-item");
        items.forEach(item => {
            const itemNum = parseInt(item.id.replace("stageStep-", ""), 10);
            item.className = "stage-item";
            if (itemNum === stageNum) {
                item.classList.add("active");
                item.classList.add("viewing");
            } else if (itemNum < stageNum) {
                item.classList.add("completed");
            }
        });

        const panesContainer = document.getElementById("panesContainer");
        if (!panesContainer) return;

        const workspaces = document.querySelectorAll(".stage-workspace");
        workspaces.forEach(w => w.style.display = "none");

        let stageWorkspace = document.getElementById(`stageWorkspace-${stageNum}`);
        if (!stageWorkspace) {
            stageWorkspace = document.createElement("div");
            stageWorkspace.className = "stage-workspace";
            stageWorkspace.id = `stageWorkspace-${stageNum}`;
            panesContainer.appendChild(stageWorkspace);

            let handled = false;
            if (typeof customStageLayoutHandler === "function") {
                handled = customStageLayoutHandler({ stageNum, stageName, agents, workspace: stageWorkspace });
            }
            if (!handled) {
                if (agents && agents.length > 0) {
                    agents.forEach(agent => {
                        createAgentPane(stageNum, agent.role, agent.color, agent.name);
                    });
                }
            }
        }
        stageWorkspace.style.display = "flex";
        emit("stageLayoutReady", { stageNum, stageName, agents, workspace: stageWorkspace });
    }

    function setupAutoScroll(el, threshold = 30) {
        if (!el || el._autoScrollReady) return;
        el._autoScrollReady = true;
        el._userScrolledAway = false;
        el._lastScrollTop = el.scrollTop;

        // Native scroll listener catches scrollbar thumb dragging, touch swipe, keyboard nav, and trackpads
        el.addEventListener('scroll', () => {
            const currentScrollTop = el.scrollTop;
            const scrollDelta = currentScrollTop - (el._lastScrollTop !== undefined ? el._lastScrollTop : currentScrollTop);
            el._lastScrollTop = currentScrollTop;

            const gap = el.scrollHeight - el.clientHeight - currentScrollTop;

            if (scrollDelta < -1) {
                // User intentionally scrolled upward away from bottom: latch detached state immediately
                el._userScrolledAway = true;
            } else if (gap <= threshold && scrollDelta > 1) {
                // User intentionally scrolled downward and reached near bottom: re-anchor
                el._userScrolledAway = false;
            } else if (gap <= 5) {
                // Directly at bottom: re-anchor
                el._userScrolledAway = false;
            }
        }, { passive: true });

        // Immediate intent latch on wheel:
        el.addEventListener('wheel', (e) => {
            if (e.deltaY < 0) {
                // User rolled wheel up: latch detached immediately
                el._userScrolledAway = true;
            } else if (e.deltaY > 0) {
                const gap = el.scrollHeight - el.clientHeight - el.scrollTop;
                if (gap <= threshold) {
                    el._userScrolledAway = false;
                }
            }
        }, { passive: true });

        el.addEventListener('touchmove', () => {
            const gap = el.scrollHeight - el.clientHeight - el.scrollTop;
            if (gap <= threshold) {
                el._userScrolledAway = false;
            }
        }, { passive: true });
    }

    function autoScroll(el) {
        if (!el || el._userScrolledAway) return;
        el.scrollTop = el.scrollHeight;
        el._lastScrollTop = el.scrollTop;
    }

    function scrollFeedToBottom(feed) {
        if (!feed) return;
        feed._userScrolledAway = false;
        feed.scrollTop = feed.scrollHeight;
        feed._lastScrollTop = feed.scrollTop;
    }

    function autoScrollFeedOrWorkspace(stageNum, agentRole) {
        if (stageNum === "qa") {
            const chatLog = document.getElementById("qaChatHistory");
            if (chatLog) autoScroll(chatLog);
        } else {
            const compoundKey = `${stageNum}_${agentRole}`;
            const paneObj = activeAgentPanes[compoundKey];
            if (paneObj && paneObj.feed) {
                autoScroll(paneObj.feed);
            }
        }
    }

    function createAgentPane(stageNum, agentRole, agentColor, displayName) {
        const compoundKey = `${stageNum}_${agentRole}`;
        if (activeAgentPanes[compoundKey]) {
            return activeAgentPanes[compoundKey];
        }

        let workspace;
        if (stageNum === "qa") {
            workspace = document.getElementById("qaChatHistory");
        } else {
            workspace = document.getElementById(`stageWorkspace-${stageNum}`);
        }
        if (!workspace) return null;

        const config = roleConfigs[agentRole] || {
            class: agentColor || "cyan",
            initials: (agentRole || "AG").slice(0, 2).toUpperCase(),
            name: displayName || agentRole
        };
        const colorClass = config.class || agentColor || "cyan";
        const initials = config.initials || (agentRole || "AG").slice(0, 2).toUpperCase();
        const roleName = displayName || config.name || agentRole;

        const pane = document.createElement("div");
        pane.className = (stageNum === "qa") ? "agent-pane qa-agent-pane" : "agent-pane";
        pane.id = `pane-${stageNum}-${agentRole.replace(/\s+/g, '')}-${Date.now()}`;

        const header = document.createElement("div");
        header.className = "agent-header";

        const avatar = document.createElement("div");
        avatar.className = `agent-avatar agent-${colorClass}`;
        avatar.innerText = initials;

        const name = document.createElement("div");
        name.className = "agent-role-name";
        name.innerText = roleName;

        header.appendChild(avatar);
        header.appendChild(name);
        pane.appendChild(header);

        const feed = document.createElement("div");
        feed.className = "agent-feed";
        if (stageNum === "qa") {
            feed.style.overflowY = "visible";
        }
        setupAutoScroll(feed);
        pane.appendChild(feed);

        workspace.appendChild(pane);
        activeAgentPanes[compoundKey] = { pane, feed };

        if (stageNum === "qa") {
            const chatLog = document.getElementById("qaChatHistory");
            if (chatLog) {
                setupAutoScroll(chatLog, 70);
                autoScroll(chatLog);
                requestAnimationFrame(() => autoScroll(chatLog));
            }
        }

        return activeAgentPanes[compoundKey];
    }

    function getOrCreateAgentPane(stageNum, agentRole, agentColor, displayName) {
        const compoundKey = `${stageNum}_${agentRole}`;
        if (!activeAgentPanes[compoundKey]) {
            createAgentPane(stageNum, agentRole, agentColor, displayName || agentRole);
        }
        return activeAgentPanes[compoundKey];
    }

    function appendRateLimitToAgentFeed(stageNum, agentRole, payload) {
        const paneObj = getOrCreateAgentPane(stageNum, agentRole, payload.agentColor);
        if (!paneObj || !paneObj.feed) return;

        const el = document.createElement("div");
        el.className = "rate-limit-feed-item";
        el.innerText = payload.message || `Received 429 "Too Many Requests". Waiting for ${payload.waitTime}s...`;
        paneObj.feed.appendChild(el);
        autoScroll(paneObj.feed);
    }

    function startReasoningBlock(stageNum, agentRole) {
        const paneObj = getOrCreateAgentPane(stageNum, agentRole);
        if (!paneObj || !paneObj.feed) return;
        const compoundKey = `${stageNum}_${agentRole}`;

        // Collapse any tool call blocks in this pane's feed
        const toolBlocks = paneObj.feed.querySelectorAll(`.tool-call-block`);
        toolBlocks.forEach(tb => tb.classList.add("collapsed"));

        const block = document.createElement("div");
        block.className = "collapsible-block";

        const header = document.createElement("div");
        header.className = "collapsible-header";
        header.innerText = "Thinking";
        header.onclick = () => block.classList.toggle("collapsed");

        const content = document.createElement("div");
        content.className = "collapsible-content";
        setupAutoScroll(content, 35);

        block.appendChild(header);
        block.appendChild(content);
        paneObj.feed.appendChild(block);

        autoScrollFeedOrWorkspace(stageNum, agentRole);
        activeBlocks[compoundKey] = {
            type: 'thinking',
            element: content,
            block: block,
            rawText: "",
            isScrolledAway: () => content._userScrolledAway
        };
    }

    function appendReasoningToken(stageNum, agentRole, token) {
        const compoundKey = `${stageNum}_${agentRole}`;
        let active = activeBlocks[compoundKey];
        if (!active || active.type !== 'thinking') {
            startReasoningBlock(stageNum, agentRole);
            active = activeBlocks[compoundKey];
        }
        if (active && active.type === 'thinking') {
            active.rawText = (active.rawText || "") + token;

            const el = active.element;
            const savedTop = el.scrollTop;
            const scrolledAway = el._userScrolledAway;

            el.innerHTML = parseMarkdown(active.rawText);

            if (scrolledAway) {
                el.scrollTop = savedTop;
                el._lastScrollTop = savedTop;
            } else {
                autoScroll(el);
            }

            autoScrollFeedOrWorkspace(stageNum, agentRole);
        }
    }

    function startContentBlock(stageNum, agentRole, phase) {
        const paneObj = getOrCreateAgentPane(stageNum, agentRole);
        if (!paneObj || !paneObj.feed) return;
        const compoundKey = `${stageNum}_${agentRole}`;

        // Collapse any tool call blocks in this pane's feed
        const toolBlocks = paneObj.feed.querySelectorAll(`.tool-call-block`);
        toolBlocks.forEach(tb => tb.classList.add("collapsed"));

        if (phase === "summary") {
            const config = roleConfigs[agentRole] || { class: "cyan", name: agentRole };
            const summaryBox = document.createElement("div");
            summaryBox.className = `summary-box agent-${config.class}`;

            const title = document.createElement("div");
            title.className = "summary-box-title";
            title.innerText = "Summary:";
            summaryBox.appendChild(title);

            const content = document.createElement("div");
            summaryBox.appendChild(content);
            paneObj.feed.appendChild(summaryBox);

            autoScrollFeedOrWorkspace(stageNum, agentRole);
            activeBlocks[compoundKey] = { type: 'summary', element: content, block: summaryBox, rawText: "" };

            createSidebarSummary(stageNum, agentRole, config);
        } else {
            const div = document.createElement("div");
            div.className = "raw-output-area";
            paneObj.feed.appendChild(div);

            autoScrollFeedOrWorkspace(stageNum, agentRole);
            activeBlocks[compoundKey] = { type: 'content', element: div, block: div, rawText: "" };
        }
    }

    function appendContentToken(stageNum, agentRole, token, phase) {
        const compoundKey = `${stageNum}_${agentRole}`;
        let active = activeBlocks[compoundKey];
        if (!active || (active.type !== 'content' && active.type !== 'summary')) {
            startContentBlock(stageNum, agentRole, phase);
            active = activeBlocks[compoundKey];
        }
        if (active && (active.type === 'content' || active.type === 'summary')) {
            active.rawText = (active.rawText || "") + token;
            active.element.innerHTML = parseMarkdown(active.rawText);

            autoScrollFeedOrWorkspace(stageNum, agentRole);

            if (active.type === 'summary') {
                const sidebarSumContent = document.getElementById(`sidebarSumContent-${stageNum}-${agentRole.replace(/\s+/g, '')}`);
                if (sidebarSumContent) {
                    sidebarSumContent.innerHTML = parseMarkdown(active.rawText);
                    const sidebarContent = document.getElementById("sidebarContent");
                    if (sidebarContent) {
                        autoScroll(sidebarContent);
                    }
                }
            }
        }
    }

    function createSidebarSummary(stageNum, agentRole, config) {
        const sidebarContent = document.getElementById("sidebarContent");
        if (!sidebarContent) return;

        const existingBox = document.getElementById(`sidebarSum-${stageNum}-${agentRole.replace(/\s+/g, '')}`);
        if (existingBox) return;

        const box = document.createElement("div");
        box.className = `summary-box agent-${config.class || "cyan"}`;
        box.id = `sidebarSum-${stageNum}-${agentRole.replace(/\s+/g, '')}`;
        box.style.marginTop = "0";

        const header = document.createElement("div");
        header.style.fontWeight = "800";
        header.style.fontSize = "12px";
        header.style.marginBottom = "6px";
        header.style.borderBottom = "1px solid #eaeff5";
        header.style.paddingBottom = "4px";
        header.innerText = `Stage ${stageNum} - ${config.name || agentRole}`;

        const content = document.createElement("div");
        content.id = `sidebarSumContent-${stageNum}-${agentRole.replace(/\s+/g, '')}`;

        box.appendChild(header);
        box.appendChild(content);
        sidebarContent.appendChild(box);

        autoScroll(sidebarContent);

        // Automatically open sidebar when summaries start streaming and switch to summaries tab
        openSidebar("summaries");
    }

    function toggleSidebar() {
        const btn = document.getElementById("sidebarToggleBtn") || document.getElementById("summariesBtn");
        if (btn && btn.disabled) return;
        const sidebar = document.getElementById("summarySidebar");
        if (sidebar) {
            sidebar.classList.toggle("collapsed");
        }
    }

    function tryFormatJson(str) {
        try {
            const trimmed = String(str).trim();
            if ((trimmed.startsWith("{") && trimmed.endsWith("}")) ||
                (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
                const parsed = JSON.parse(trimmed);
                return JSON.stringify(parsed, null, 2);
            }
        } catch (e) {}
        return str;
    }

    function extractPartialCode(rawJson) {
        const match = rawJson.match(/"code"\s*:\s*"((?:[^"\\]|\\.)*)/);
        if (match) {
            return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
        }
        return rawJson;
    }

    function startToolCallStream(stageNum, agentRole, index, toolName) {
        const paneObj = getOrCreateAgentPane(stageNum, agentRole);
        if (!paneObj || !paneObj.feed) return;
        const compoundKey = `${stageNum}_${agentRole}`;
        const streamId = `streamToolCall-${stageNum}-${index}`;

        let toolBlock = document.getElementById(streamId);
        if (!toolBlock) {
            // Collapse any active tool call blocks in this pane's feed
            const toolBlocks = paneObj.feed.querySelectorAll(`.tool-call-block`);
            toolBlocks.forEach(tb => tb.classList.add("collapsed"));

            toolBlock = document.createElement("div");
            toolBlock.className = "collapsible-block tool-call-block";
            toolBlock.id = streamId;

            const header = document.createElement("div");
            header.className = "collapsible-header tool-call-header";
            header.onclick = () => toolBlock.classList.toggle("collapsed");

            const label = document.createElement("span");
            label.className = "tool-call-label";
            label.innerHTML = `Tool: <strong class="tool-call-name">${toolName}</strong>`;
            header.appendChild(label);

            const badge = document.createElement("span");
            badge.className = "tool-status-badge status-running";
            badge.innerText = "generating";
            header.appendChild(badge);

            const argsPreview = document.createElement("span");
            argsPreview.className = "tool-call-args-preview";
            header.appendChild(argsPreview);

            const content = document.createElement("div");
            content.className = "collapsible-content tool-call-content";

            const sectionTitle = document.createElement("div");
            sectionTitle.className = "tool-section-title";
            sectionTitle.style.fontWeight = "bold";
            sectionTitle.style.fontSize = "12px";
            sectionTitle.style.color = "#475569";
            sectionTitle.style.marginBottom = "4px";
            sectionTitle.innerText = toolName === "executePythonCalculation" ? "Python Code:" : "Arguments:";
            content.appendChild(sectionTitle);

            const pre = document.createElement("pre");
            pre.className = "tool-stream-pre";
            pre.style.margin = "0";
            if (toolName === "executePythonCalculation") {
                pre.style.backgroundColor = "#0f172a";
                pre.style.color = "#38bdf8";
                pre.style.padding = "10px";
            } else {
                pre.style.backgroundColor = "#f1f5f9";
                pre.style.color = "#0f172a";
                pre.style.padding = "8px";
            }
            pre.style.borderRadius = "4px";
            pre.style.fontFamily = "monospace";
            pre.style.fontSize = "12px";
            pre.style.overflowX = "auto";
            pre.style.whiteSpace = "pre-wrap";
            content.appendChild(pre);

            toolBlock.appendChild(header);
            toolBlock.appendChild(content);
            paneObj.feed.appendChild(toolBlock);

            autoScrollFeedOrWorkspace(stageNum, agentRole);
            activeBlocks[compoundKey] = { type: 'tool', element: content, block: toolBlock };
        } else {
            const label = toolBlock.querySelector(".tool-call-name");
            if (label) label.innerText = toolName;
        }
    }

    function appendToolCallStreamToken(stageNum, agentRole, index, token) {
        const streamId = `streamToolCall-${stageNum}-${index}`;
        const key = `${stageNum}_${index}`;

        streamingArgs[key] = (streamingArgs[key] || "") + token;
        const raw = streamingArgs[key];

        const toolBlock = document.getElementById(streamId);
        if (toolBlock) {
            const nameEl = toolBlock.querySelector(".tool-call-name");
            const toolName = nameEl ? nameEl.innerText : "";

            const argsPreview = toolBlock.querySelector(".tool-call-args-preview");
            if (argsPreview) {
                if (toolName === "executePythonCalculation") {
                    argsPreview.innerText = "";
                } else {
                    argsPreview.innerText = `Arguments: ${raw.trim()}`;
                }
            }

            const pre = toolBlock.querySelector(".tool-stream-pre");
            if (pre) {
                if (toolName === "executePythonCalculation") {
                    pre.innerText = extractPartialCode(raw);
                } else {
                    pre.innerText = raw.trim();
                }
            }

            autoScrollFeedOrWorkspace(stageNum, agentRole);
        }
    }

    function startToolBlock(stageNum, agentRole, toolName, args, callId, toolIndex) {
        const paneObj = getOrCreateAgentPane(stageNum, agentRole);
        if (!paneObj || !paneObj.feed) return;
        const compoundKey = `${stageNum}_${agentRole}`;
        const streamId = `streamToolCall-${stageNum}-${toolIndex}`;

        let toolBlock = document.getElementById(streamId);
        if (!toolBlock) {
            // Collapse any active tool call blocks
            const toolBlocks = paneObj.feed.querySelectorAll(`.tool-call-block`);
            toolBlocks.forEach(tb => tb.classList.add("collapsed"));

            toolBlock = document.createElement("div");
            toolBlock.className = "collapsible-block tool-call-block";
            paneObj.feed.appendChild(toolBlock);
        }

        // Bind the actual callId to the block ID
        toolBlock.id = `toolCall-${stageNum}-${callId}`;

        // Re-render header and content with final pretty-formatted json/code
        toolBlock.innerHTML = "";

        const header = document.createElement("div");
        header.className = "collapsible-header tool-call-header";
        header.onclick = () => toolBlock.classList.toggle("collapsed");

        const label = document.createElement("span");
        label.className = "tool-call-label";
        label.innerHTML = `Tool: <strong class="tool-call-name">${toolName}</strong>`;
        header.appendChild(label);

        const badge = document.createElement("span");
        badge.className = "tool-status-badge status-running";
        badge.innerText = "running";
        header.appendChild(badge);

        const trimmedArgs = String(args || "").trim();
        const prettyArgs = tryFormatJson(trimmedArgs);

        const argsPreview = document.createElement("span");
        argsPreview.className = "tool-call-args-preview";
        if (toolName === "executePythonCalculation") {
            argsPreview.innerText = "";
        } else {
            argsPreview.innerText = `Arguments: ${trimmedArgs}`;
        }
        header.appendChild(argsPreview);

        const content = document.createElement("div");
        content.className = "collapsible-content tool-call-content";

        if (toolName === "executePythonCalculation") {
            let codeVal = "";
            try {
                const parsed = JSON.parse(args);
                codeVal = String(parsed.code || "").trim();
            } catch(e) {
                codeVal = trimmedArgs;
            }
            const codeDiv = document.createElement("div");
            codeDiv.style.marginBottom = "10px";
            codeDiv.innerHTML = `<div style="font-weight: bold; font-size: 12px; color: #475569; margin-bottom: 4px;">Python Code:</div>
                                 <pre style="margin: 0; background-color: #0f172a; color: #38bdf8; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; overflow-x: auto; white-space: pre-wrap;">${codeVal}</pre>`;
            content.appendChild(codeDiv);
        } else {
            const argsDiv = document.createElement("div");
            argsDiv.style.marginBottom = "10px";
            argsDiv.innerHTML = `<div style="font-weight: bold; font-size: 12px; color: #475569; margin-bottom: 4px;">Arguments:</div>
                                 <pre style="margin: 0; background-color: #f1f5f9; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; color: #0f172a;">${prettyArgs}</pre>`;
            content.appendChild(argsDiv);
        }

        toolBlock.appendChild(header);
        toolBlock.appendChild(content);

        autoScrollFeedOrWorkspace(stageNum, agentRole);
        activeBlocks[compoundKey] = { type: 'tool', element: content, block: toolBlock };
    }

    function updateToolBlockProgress(stageNum, callId, progress) {
        let toolBlock = document.getElementById(`toolCall-${stageNum}-${callId}`);
        if (!toolBlock) {
            toolBlock = document.querySelector(`[id$="-${callId}"]`);
        }
        if (toolBlock) {
            const badge = toolBlock.querySelector(".tool-status-badge");
            if (badge) {
                badge.className = "tool-status-badge status-running";
                badge.innerText = String(progress).toUpperCase();
            }
        }
    }

    function endToolBlock(stageNum, agentRole, toolName, callId, status, result, stdout, variables, toolIndex) {
        let toolBlock = document.getElementById(`toolCall-${stageNum}-${callId}`);
        if (!toolBlock) {
            toolBlock = document.querySelector(`[id$="-${callId}"]`);
        }
        if (toolBlock) {
            const badge = toolBlock.querySelector(".tool-status-badge");
            if (badge) {
                badge.className = `tool-status-badge status-${status}`;
                const displayStatus = status === "success" ? "COMPLETE" : (status === "error" ? "ERROR" : status.toUpperCase());
                badge.innerText = displayStatus;
            }

            const content = toolBlock.querySelector(".collapsible-content");
            if (content) {
                const trimmedResult = String(result || "").trim();
                const prettyResult = tryFormatJson(trimmedResult);

                const resDiv = document.createElement("div");
                resDiv.style.marginTop = "10px";
                resDiv.innerHTML = `<div style="font-weight: bold; font-size: 12px; color: #475569; margin-bottom: 4px;">Result:</div>
                                     <pre style="margin: 0; background-color: #f8fafc; border: 1.5px solid #e2e8f0; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; color: #0f172a;">${prettyResult}</pre>`;
                content.appendChild(resDiv);

                if (stdout && stdout.trim()) {
                    const trimmedStdout = String(stdout).trim();
                    const stdoutDiv = document.createElement("div");
                    stdoutDiv.style.marginTop = "10px";
                    stdoutDiv.innerHTML = `<div style="font-weight: bold; font-size: 12px; color: #475569; margin-bottom: 4px;">STDOUT:</div>
                                           <pre style="margin: 0; background-color: #0f172a; color: #38bdf8; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; overflow-x: auto; white-space: pre-wrap;">${trimmedStdout}</pre>`;
                    content.appendChild(stdoutDiv);
                }

                if (variables && Object.keys(variables).length > 0) {
                    const varDiv = document.createElement("div");
                    varDiv.style.marginTop = "10px";
                    let varRows = "";
                    for (const [k, v] of Object.entries(variables)) {
                        const prettyVal = tryFormatJson(v);
                        varRows += `<tr><td style="padding: 4px 8px; font-weight: bold; font-family: monospace; border: 1px solid #cbd5e1; vertical-align: top;">${k}</td><td style="padding: 4px 8px; font-family: monospace; border: 1px solid #cbd5e1; vertical-align: top; white-space: pre-wrap;">${prettyVal}</td></tr>`;
                    }
                    varDiv.innerHTML = `<div style="font-weight: bold; font-size: 12px; color: #475569; margin-bottom: 4px;">Variables:</div>
                                         <table style="border-collapse: collapse; width: 100%; border: 1px solid #cbd5e1; font-size: 12px; background-color: #ffffff;">${varRows}</table>`;
                    content.appendChild(varDiv);
                }
            }

            // Collapse the tool call block automatically after it finishes running
            toolBlock.classList.add("collapsed");

            const key = `${stageNum}_${toolIndex}`;
            delete streamingArgs[key];

            autoScrollFeedOrWorkspace(stageNum, agentRole);
        }
    }

    function endCurrentBlock(stageNum, agentRole) {
        const compoundKey = `${stageNum}_${agentRole}`;
        const active = activeBlocks[compoundKey];
        if (active) {
            if (active.type === 'thinking' && active.block) {
                active.block.classList.add("collapsed");
            }
            delete activeBlocks[compoundKey];
        }
    }

    function openSettingsModal() {
        const overlay = document.getElementById("settingsOverlay");
        if (overlay) overlay.style.display = "flex";
    }

    function closeSettingsModal() {
        const overlay = document.getElementById("settingsOverlay");
        if (overlay) overlay.style.display = "none";
    }

    function handleSettingsOverlayMouseDown(event) {
        if (event.target && event.target.id === "settingsOverlay") {
            closeSettingsModal();
        }
    }

    function getGenerationSettings() {
        const maxIters = parseInt(document.getElementById("maxItersInput")?.value || 6, 10);
        const thinkingBudget = parseInt(document.getElementById("thinkingBudgetInput")?.value || 2048, 10);
        const temp = parseFloat(document.getElementById("temperatureInput")?.value || 0.6);
        const generateSummaries = document.getElementById("generateSummariesCheckbox") ? document.getElementById("generateSummariesCheckbox").checked : true;

        return {
            maxIterations: maxIters,
            thinkingBudget: thinkingBudget,
            temperature: temp,
            generateSummaries: generateSummaries
        };
    }

    function showActiveBoardroomModal() {
        const overlay = document.getElementById("activeBoardroomOverlay");
        if (overlay) overlay.style.display = "flex";
    }

    function hideActiveBoardroomModal() {
        const overlay = document.getElementById("activeBoardroomOverlay");
        if (overlay) overlay.style.display = "none";
    }

    async function checkBoardroomStatus() {
        try {
            const res = await fetch('/api/boardroom/status');
            if (res.ok) {
                const data = await res.json();
                if (data.isRunning) {
                    showActiveBoardroomModal();
                    setControlsRunningState(true);
                } else {
                    hideActiveBoardroomModal();
                    setControlsRunningState(false);
                }
            }
        } catch (e) {
            console.error("Error checking boardroom status:", e);
        }
    }

    async function killActiveBoardroom() {
        const killBtn = document.getElementById("killBoardroomBtn");
        if (killBtn) killBtn.disabled = true;

        try {
            const res = await fetch('/api/boardroom/stop', { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                if (data.stopped || !data.isRunning) {
                    hideActiveBoardroomModal();
                    setControlsRunningState(false);
                    if (killBtn) killBtn.disabled = false;
                    return;
                }
            }
        } catch (e) {
            console.error("Error stopping boardroom:", e);
        }

        let attempts = 0;
        const pollInterval = setInterval(async () => {
            attempts++;
            try {
                const res = await fetch('/api/boardroom/status');
                if (res.ok) {
                    const data = await res.json();
                    if (!data.isRunning || attempts >= 20) {
                        clearInterval(pollInterval);
                        hideActiveBoardroomModal();
                        setControlsRunningState(false);
                        if (killBtn) killBtn.disabled = false;
                    }
                }
            } catch (e) {
                if (attempts >= 20) {
                    clearInterval(pollInterval);
                    hideActiveBoardroomModal();
                    setControlsRunningState(false);
                    if (killBtn) killBtn.disabled = false;
                }
            }
        }, 300);
    }

    function stopSimulation() {
        const stopBtn = document.getElementById("stopBtn");
        if (stopBtn) {
            stopBtn.disabled = true;
        }
        if (typeof window.sendWsMessage === "function") {
            window.sendWsMessage({ action: "stop" });
        }
        fetch('/api/boardroom/stop', { method: 'POST' }).catch(() => {});
    }

    function startSimulation() {
        if (!serverIsLoaded && !window.serverIsLoaded) {
            alert("Server not connect. Please load and connect to a server first.");
            return;
        }

        if (typeof getPayloadFn !== "function") {
            console.error("BoardroomCore: getPayload function is not registered.");
            return;
        }

        const payload = getPayloadFn();
        if (!payload) return; // Validation failed inside child handler

        setControlsRunningState(true);

        const panesContainer = document.getElementById("panesContainer");
        if (panesContainer) panesContainer.innerHTML = "";

        const modeSelect = document.getElementById("modeSelect");
        if (modeSelect) {
            initStagesBar(modeSelect.value);
        }

        startSimulationTimer();
        if (typeof BoardroomExporter !== "undefined" && BoardroomExporter.handleSimStart) {
            BoardroomExporter.handleSimStart();
        }

        // Clear existing state
        activeAgentPanes = {};
        activeBlocks = {};
        streamingArgs = {};
        currentStageNumber = 0;
        highestStageNumber = 0;
        updateSummariesBtnState();
        resetSourcesUI();

        const sidebarContent = document.getElementById("sidebarContent");
        if (sidebarContent) sidebarContent.innerHTML = "";

        emit("simulationStarted", payload);

        if (typeof window.sendWsMessage === "function") {
            window.sendWsMessage(payload);
        } else {
            console.error("BoardroomCore: sendWsMessage is not available on window.");
        }
    }

    function on(eventType, callback) {
        if (!customEventListeners[eventType]) {
            customEventListeners[eventType] = [];
        }
        customEventListeners[eventType].push(callback);
    }

    function emit(eventType, data) {
        if (customEventListeners[eventType]) {
            customEventListeners[eventType].forEach(cb => {
                try { cb(data); } catch (e) { console.error(`Error in ${eventType} listener:`, e); }
            });
        }
    }

    function handleSimulationEvent(payload) {
        if (!payload || !payload.type) return;

        const { type, agentRole, agentColor, phase, stageNum } = payload;
        const activeStage = (stageNum !== undefined && stageNum !== null) ? stageNum : currentStageNumber;

        switch (type) {
            case "metricsUpdate":
                setServerConnectedState(true);
                if (typeof updateCostUI === "function" && payload.costData) {
                    updateCostUI(payload.costData);
                }
                break;

            case "costUpdate":
                setServerConnectedState(true);
                if (typeof updateCostUI === "function") {
                    updateCostUI(payload);
                }
                break;

            case "serverStatus":
                applyServerStatus(payload);
                break;

            case "stageStart":
                setupStageLayout(payload.stageNum, payload.stageName, payload.agents);
                break;

            case "agentRunStart":
                if (activeStage === "qa") {
                    const compoundKey = `qa_${agentRole}`;
                    delete activeBlocks[compoundKey];
                    delete activeAgentPanes[compoundKey];
                }
                getOrCreateAgentPane(activeStage, agentRole, agentColor, payload.displayName);
                break;

            case "reasoningStart":
                startReasoningBlock(activeStage, agentRole);
                break;

            case "reasoningToken":
                appendReasoningToken(activeStage, agentRole, payload.token);
                break;

            case "reasoningEnd":
                endCurrentBlock(activeStage, agentRole);
                break;

            case "contentStart":
                startContentBlock(activeStage, agentRole, phase);
                break;

            case "contentToken":
                appendContentToken(activeStage, agentRole, payload.token, phase);
                break;

            case "contentEnd":
                endCurrentBlock(activeStage, agentRole);
                break;

            case "toolCallStreamStart": {
                const tIndex = (payload.index !== undefined) ? payload.index : payload.toolIndex;
                startToolCallStream(activeStage, agentRole, tIndex, payload.toolName);
                break;
            }

            case "toolCallStreamToken": {
                const tIndex = (payload.index !== undefined) ? payload.index : payload.toolIndex;
                appendToolCallStreamToken(activeStage, agentRole, tIndex, payload.token);
                break;
            }

            case "toolCallStart": {
                if (payload.toolName && payload.toolName.startsWith("confirmBoardroomDecision")) {
                    emit("decisionStarted", payload);
                }
                const tIndex = (payload.toolIndex !== undefined) ? payload.toolIndex : payload.index;
                startToolBlock(activeStage, agentRole, payload.toolName, payload.args, payload.callId, tIndex);
                break;
            }

            case "toolCallProgress":
                updateToolBlockProgress(activeStage, payload.callId, payload.progress);
                break;

            case "toolCallEnd":
                if (payload.toolName && payload.toolName.startsWith("confirmBoardroomDecision")) {
                    emit("decisionConfirmed", payload);
                }
                endToolBlock(
                    activeStage,
                    agentRole,
                    payload.toolName,
                    payload.callId,
                    payload.status,
                    payload.result,
                    payload.stdout,
                    payload.variables,
                    (payload.toolIndex !== undefined ? payload.toolIndex : payload.index)
                );
                break;

            case "agentRunEnd":
                endCurrentBlock(activeStage, agentRole);
                break;

            case "rateLimit":
                if (agentRole) {
                    appendRateLimitToAgentFeed(activeStage, agentRole, payload);
                }
                break;

            case "newSource":
                if (payload.source) {
                    addSourceRecord(payload.source);
                }
                emit("newSource", payload);
                break;

            case "resetSources":
                resetSourcesUI();
                emit("resetSources", payload);
                break;

            case "qaComplete":
                emit("qaComplete", payload);
                break;

            case "simComplete":
                setControlsRunningState(false);
                stopSimulationTimer();

                const qaTab = document.getElementById("stageStep-qa");
                if (qaTab) {
                    qaTab.style.display = "block";
                    qaTab.classList.remove("disabled");
                }

                if (payload.totalTime) {
                    const timerEl = document.getElementById("simulationTimer");
                    if (timerEl) {
                        timerEl.innerText = payload.totalTime;
                    }
                }

                if (typeof BoardroomExporter !== "undefined" && BoardroomExporter.handleSimComplete) {
                    BoardroomExporter.handleSimComplete(payload);
                }

                // Mark all stages as completed in the stages bar
                const items = document.querySelectorAll(".stage-item");
                items.forEach(item => {
                    item.classList.remove("active");
                    item.classList.add("completed");
                });

                emit("simComplete", payload);
                break;

            case "simStopped":
                hideActiveBoardroomModal();
                setControlsRunningState(false);
                stopSimulationTimer();
                document.querySelectorAll(".stage-item").forEach(item => item.classList.remove("active"));
                emit("simStopped", payload);
                break;

            case "error":
                hideActiveBoardroomModal();
                alert("Simulation Error: " + payload.message);
                setControlsRunningState(false);
                stopSimulationTimer();
                document.querySelectorAll(".stage-item").forEach(item => item.classList.remove("active"));
                emit("error", payload);
                break;
        }

        // Emit to custom listeners
        emit(payload.type, payload);
    }

    function init({ defaultStages, getPayload, initialMode }) {
        defaultStagesConfig = defaultStages || {};
        getPayloadFn = getPayload;

        const mode = initialMode || (document.getElementById("modeSelect") ? document.getElementById("modeSelect").value : "fast");
        initStagesBar(mode, defaultStagesConfig);

        const tempInput = document.getElementById("temperatureInput");
        const tempText = document.getElementById("temperatureInputText");
        if (tempInput && tempText) {
            tempInput.addEventListener("input", () => {
                tempText.innerText = parseFloat(tempInput.value).toFixed(2);
            });
        }

        setupAutoScroll(document.getElementById("sidebarContent"));
        initSimulatedDateControl();
        if (typeof connectWebsocket === "function") {
            connectWebsocket();
        } else if (typeof window.connectWebsocket === "function") {
            window.connectWebsocket();
        }
        checkServerStatus();
        checkBoardroomStatus();
    }

    function initSimulatedDateControl() {
        const dateInput = document.getElementById("simulatedDateInput");
        const checkbox = document.getElementById("simulatedDateCheckbox");
        const container = document.getElementById("simulatedDateContainer");

        if (dateInput) {
            const yesterdayStr = (typeof window.getYesterdayDateString === 'function')
                ? window.getYesterdayDateString()
                : (() => {
                    const d = new Date();
                    d.setDate(d.getDate() - 1);
                    return d.toISOString().split('T')[0];
                })();
            dateInput.setAttribute("max", yesterdayStr);
        }

        if (checkbox && container) {
            checkbox.addEventListener("change", function() {
                container.style.display = this.checked ? "flex" : "none";
            });
        }
    }

    function getSimulatedDate() {
        const checkbox = document.getElementById("simulatedDateCheckbox");
        const dateInput = document.getElementById("simulatedDateInput");
        if (checkbox && checkbox.checked) {
            const val = dateInput ? dateInput.value : null;
            if (!val) {
                alert("Please select a valid simulated evaluation date.");
                return undefined;
            }
            return val;
        }
        return null;
    }

    // Dev Test Demo function
    function runTestDemo(animated = false) {
        console.log("Running Test Demo Simulation (animated=" + animated + ")...");
        const events = [
            {
                type: "stageStart",
                stageNum: 1,
                stageName: "Macro Environment Analysis",
                agents: [{ role: "Macro Analyst", color: "cyan", name: "Macro Analyst" }]
            },
            {
                type: "agentRunStart",
                agentRole: "Macro Analyst",
                agentColor: "cyan",
                phase: "raw",
                stageNum: 1
            },
            {
                type: "reasoningStart",
                stageNum: 1,
                agentRole: "Macro Analyst"
            },
            {
                type: "reasoningToken",
                stageNum: 1,
                agentRole: "Macro Analyst",
                token: "Here is some reasoning text... \n\nI will now perform a tool call:"
            },
            {
                type: "reasoningEnd",
                stageNum: 1,
                agentRole: "Macro Analyst"
            },
            {
                type: "rateLimit",
                stageNum: 1,
                agentRole: "Macro Analyst",
                agentColor: "cyan",
                waitTime: 8.0,
                message: "Received 429 \"Too Many Requests\". Waiting for 8.0 seconds..."
            },
            {
                type: "toolCallStart",
                stageNum: 1,
                agentRole: "Macro Analyst",
                toolName: "exampleToolCall",
                args: JSON.stringify({ query: "lorem ipsum", limit: 5, filter: "active" }),
                callId: "call_example_001",
                toolIndex: 0
            },
            {
                type: "toolCallEnd",
                stageNum: 1,
                agentRole: "Macro Analyst",
                toolName: "exampleToolCall",
                callId: "call_example_001",
                status: "success",
                result: JSON.stringify({ status: "ok", results: ["lorem", "ipsum", "dolor"] }),
                stdout: "",
                variables: {},
                toolIndex: 0
            },
            {
                type: "contentStart",
                stageNum: 1,
                agentRole: "Macro Analyst",
                phase: "raw"
            },
            {
                type: "contentToken",
                stageNum: 1,
                agentRole: "Macro Analyst",
                phase: "raw",
                token: `# Heading 1\n## Heading 2\n### Heading 3\n\nHere is **bold**, *italic*, and a table:\n\n| Sector | Conviction | Weight |\n| :--- | :--- | :--- |\n| Technology | **High** | 35% |\n| Healthcare | Medium | 20% |\n| Energy | Defensive | 15% |\n\n--- \nComplete demonstration.`
            },
            {
                type: "contentEnd",
                stageNum: 1,
                agentRole: "Macro Analyst"
            },
            {
                type: "agentRunEnd",
                stageNum: 1,
                agentRole: "Macro Analyst"
            }
        ];

        if (!animated) {
            events.forEach(evt => handleSimulationEvent(evt));
            console.log("✓ Test Demo rendered immediately into Stage 1!");
        } else {
            let idx = 0;
            function playNext() {
                if (idx < events.length) {
                    const evt = events[idx++];
                    handleSimulationEvent(evt);
                    setTimeout(playNext, 150);
                } else {
                    console.log("✓ Animated Test Demo completed!");
                }
            }
            playNext();
        }
    }

    return {
        init,
        on,
        emit,
        startSimulation,
        stopSimulation,
        handleSimulationEvent,
        showStageWorkspace,
        setupStageLayout,
        initStagesBar,
        switchSidebarTab,
        openSidebar,
        toggleSidebar,
        openSettingsModal,
        closeSettingsModal,
        handleSettingsOverlayMouseDown,
        getGenerationSettings,
        checkBoardroomStatus,
        killActiveBoardroom,
        checkServerStatus,
        setServerConnectedState,
        setControlsRunningState,
        setCustomStageLayoutHandler: (fn) => { customStageLayoutHandler = fn; },
        createAgentPane,
        getOrCreateAgentPane,
        getCurrentStageNumber: () => currentStageNumber,
        getHighestStageNumber: () => highestStageNumber,
        updateSummariesBtnState,
        setupAutoScroll,
        autoScroll,
        autoScrollFeedOrWorkspace,
        scrollFeedToBottom,
        parseMarkdown,
        formatOrdinalDate,
        registerRole,
        runTestDemo,
        openSourceModal,
        closeSourceModal,
        showSourceCitation,
        ensureSourcesWorkspace,
        resetSourcesUI,
        getSources: () => capturedSources,
        initSimulatedDateControl,
        getSimulatedDate
    };
})();

// Global convenience bindings
window.BoardroomCore = BoardroomCore;
window.handleSimulationEvent = (p) => BoardroomCore.handleSimulationEvent(p);
window.setServerConnectedState = (c) => BoardroomCore.setServerConnectedState(c);
window.setControlsRunningState = (s) => BoardroomCore.setControlsRunningState(s);
window.openSettingsModal = BoardroomCore.openSettingsModal;
window.closeSettingsModal = BoardroomCore.closeSettingsModal;
window.handleSettingsOverlayMouseDown = BoardroomCore.handleSettingsOverlayMouseDown;
window.killActiveBoardroom = BoardroomCore.killActiveBoardroom;
window.switchSidebarTab = BoardroomCore.switchSidebarTab;
window.openSidebar = BoardroomCore.openSidebar;
window.toggleSidebar = BoardroomCore.toggleSidebar;
window.startSimulation = BoardroomCore.startSimulation;
window.stopSimulation = BoardroomCore.stopSimulation;
window.showStageWorkspace = BoardroomCore.showStageWorkspace;
window.openSourceModal = BoardroomCore.openSourceModal;
window.closeSourceModal = BoardroomCore.closeSourceModal;
window.showSourceCitation = BoardroomCore.showSourceCitation;
window.runTestDemo = BoardroomCore.runTestDemo;
window.testDemo = BoardroomCore.runTestDemo;
window.testMacroStage = BoardroomCore.runTestDemo;
window.initSimulatedDateControl = BoardroomCore.initSimulatedDateControl;
window.getSimulatedDate = BoardroomCore.getSimulatedDate;

