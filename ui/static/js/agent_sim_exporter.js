/**
 * agent_sim_exporter.js
 * Standalone Self-Contained HTML Report Exporter for Agent-Driven Portfolio Simulation.
 * Captures all simulation parameters, all historical sessions & milestones,
 * post-session metrics, active holdings, activity logs, dialogue feeds,
 * and live interactive Chart.js charts with automatic Base64 image fallback.
 */

const AgentSimExporter = (function () {
    let lastSimTotalTime = "";

    function getCleanInitialCapital() {
        const val = document.getElementById("initialCapitalInput")?.value;
        const num = parseFloat(String(val).replace(/[^0-9.]/g, ""));
        return isNaN(num) || num <= 0 ? 100000 : num;
    }

    function formatOrdinalDate(dateInput) {
        if (!dateInput) dateInput = new Date().toISOString().split('T')[0];
        const months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ];

        let year, monthIdx, day;
        if (typeof dateInput === "string" && /^\d{4}-\d{2}-\d{2}/.test(dateInput)) {
            const parts = dateInput.split('T')[0].split('-');
            year = parseInt(parts[0], 10);
            monthIdx = parseInt(parts[1], 10) - 1;
            day = parseInt(parts[2], 10);
        } else {
            const d = (dateInput instanceof Date) ? dateInput : new Date(dateInput);
            const valid = isNaN(d.getTime()) ? new Date() : d;
            year = valid.getFullYear();
            monthIdx = valid.getMonth();
            day = valid.getDate();
        }

        let suffix = "th";
        const lastTwo = day % 100;
        if (lastTwo < 11 || lastTwo > 13) {
            const lastDigit = day % 10;
            if (lastDigit === 1) suffix = "st";
            else if (lastDigit === 2) suffix = "nd";
            else if (lastDigit === 3) suffix = "rd";
        }
        return `${day}${suffix} ${months[monthIdx] || ""} ${year}`;
    }

    function formatOrdinalDateTime(dateObj) {
        const d = dateObj || new Date();
        const datePart = formatOrdinalDate(d);
        const hours = String(d.getHours()).padStart(2, "0");
        const minutes = String(d.getMinutes()).padStart(2, "0");
        return `${datePart}, ${hours}:${minutes}`;
    }

    function extractSimulationMetadata() {
        const startDateInput = document.getElementById("setupStartDate");
        const endDateInput = document.getElementById("setupEndDate");
        const capital = getCleanInitialCapital();

        const startDate = startDateInput?.value || "2024-09-01";
        const endDate = endDateInput?.value || "2026-09-01";

        const title = `Agent Portfolio Simulation Report`;
        const todayStr = new Date().toISOString().split("T")[0];
        const filename = `Agent_Sim_Report_${startDate}_to_${endDate}_${todayStr}.html`;

        const params = [];
        params.push({ key: "Date Produced", value: formatOrdinalDateTime(new Date()) });
        params.push({ key: "Simulation Period", value: `${formatOrdinalDate(startDate)} - ${formatOrdinalDate(endDate)}` });
        params.push({ key: "Initial Capital", value: `$${capital.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` });

        const timestepSelect = document.getElementById("timestepSelect");
        if (timestepSelect) {
            params.push({ key: "Timestep", value: timestepSelect.value });
        }

        const rebalanceSlider = document.getElementById("rebalanceAmountSlider");
        if (rebalanceSlider) {
            const rebalLabels = {
                1: "Level 1: Very Light",
                2: "Level 2: Mild",
                3: "Level 3: Moderate",
                4: "Level 4: Substantial",
                5: "Level 5: Aggressive",
                6: "Level 6: Maximum"
            };
            params.push({ key: "Rebalance", value: rebalLabels[rebalanceSlider.value] || `Level ${rebalanceSlider.value}` });
        }

        const biasCb = document.getElementById("allocationBiasCheckbox");
        if (biasCb && biasCb.checked) {
            const slider = document.getElementById("allocationBiasSlider");
            const biasLabels = {
                1: "Maximum Growth Bias",
                2: "Moderate Growth Bias",
                3: "Minor Growth Bias",
                4: "Minor Defensive Bias",
                5: "Moderate Defensive Bias",
                6: "Maximum Defensive Bias"
            };
            params.push({ key: "Allocation Bias", value: biasLabels[slider?.value] || "Growth Bias" });
        } else {
            params.push({ key: "Allocation Bias", value: "Balanced Strategy" });
        }

        const secInput = document.getElementById("targetSectorCountInput");
        const secVal = secInput?.value?.trim();
        params.push({ key: "Target Sectors", value: secVal ? `${secVal} sectors` : "Dynamic (Auto)" });

        const stInput = document.getElementById("targetStockCountInput");
        const stVal = stInput?.value?.trim();
        params.push({ key: "Target Stocks", value: stVal ? `${stVal} stocks` : "Dynamic (Auto)" });

        return { title, filename, params, startDate, endDate };
    }

    async function collectAllStylesheets() {
        let fullCss = "";
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

        // Add report specific classes & overrides (Copied exactly from boardroom_exporter.js)
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
            .chart-fallback-img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                margin: 0 auto;
                display: block;
            }
        `;

        return fullCss;
    }

    function freezeCanvases(originalRoot, targetClone) {
        const snapshots = {};
        const origCanvases = Array.from(originalRoot.querySelectorAll("canvas"));

        origCanvases.forEach(origCanvas => {
            if (!origCanvas.id) return;
            try {
                const dataUrl = origCanvas.toDataURL("image/png");
                if (dataUrl && dataUrl.length > 100) {
                    snapshots[origCanvas.id] = dataUrl;

                    const cloneCanvas = targetClone.querySelector(`#${origCanvas.id}`);
                    if (cloneCanvas && cloneCanvas.parentNode) {
                        const fallbackImg = document.createElement("img");
                        fallbackImg.id = `${origCanvas.id}_fallback`;
                        fallbackImg.src = dataUrl;
                        fallbackImg.alt = origCanvas.id;
                        fallbackImg.className = "chart-fallback-img";
                        fallbackImg.style.cssText = "display: none; width: 100%; height: 100%; object-fit: contain;";
                        cloneCanvas.parentNode.insertBefore(fallbackImg, cloneCanvas.nextSibling);
                    }
                }
            } catch (err) {
                console.warn("Could not capture canvas snapshot:", origCanvas.id, err);
            }
        });

        return snapshots;
    }

    function serializeSimulationState() {
        const exportedSessions = {};
        if (typeof sessionMgr !== "undefined" && sessionMgr.sessions) {
            Object.values(sessionMgr.sessions).forEach(sess => {
                if (!sess || !sess.id) return;
                exportedSessions[sess.id] = {
                    id: sess.id,
                    label: sess.label,
                    date: sess.date,
                    dateFormatted: sess.dateFormatted,
                    type: sess.type,
                    pace: sess.pace,
                    stages: sess.stages || [],
                    isCompleted: sess.isCompleted,
                    currentStageNum: sess.currentStageNum,
                    baselinePositions: sess.baselinePositions || [],
                    confirmedSector: sess.confirmedSector || null,
                    confirmedPortfolio: sess.confirmedPortfolio || null,
                    sessionMetrics: sess.sessionMetrics || null
                };
            });
        }

        const ptsPort = (typeof fullPortfolioPoints !== "undefined" && Array.isArray(fullPortfolioPoints)) ? fullPortfolioPoints : [];
        const ptsSp = (typeof fullSp500Points !== "undefined" && Array.isArray(fullSp500Points)) ? fullSp500Points : [];
        const liveState = (typeof lastLiveSimulationState !== "undefined") ? lastLiveSimulationState : null;
        const liveId = (typeof sessionMgr !== "undefined") ? sessionMgr.liveSessionId : null;

        return {
            sessions: exportedSessions,
            fullPortfolioPoints: ptsPort,
            fullSp500Points: ptsSp,
            lastLiveSimulationState: liveState,
            liveSessionId: liveId
        };
    }

    function buildOfflineScript(serializedData, snapshots) {
        return `
        <script>
            var simulationData = ${JSON.stringify(serializedData)};
            var canvasSnapshots = ${JSON.stringify(snapshots)};
            var activeSessionId = "milestone_0";
            var activeStageNum = 1;
            var currentChartTf = "all";
            var perfChartInstance = null;
            var doughnutChartInstances = {};

            function escapeHtml(str) {
                if (!str) return "";
                return String(str)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            function formatFriendlyDate(dateInput) {
                if (!dateInput) return "--";
                if (typeof dateInput === 'string') {
                    var clean = dateInput.trim();
                    if (/^[A-Za-z]{3}\\s+\\d{2}\\s+[A-Za-z]{3}\\s+\\d{4}$/.test(clean)) return clean;
                    var ddmmyyyy = clean.match(/^(\\d{2})-(\\d{2})-(\\d{4})$/);
                    if (ddmmyyyy) dateInput = ddmmyyyy[3] + '-' + ddmmyyyy[2] + '-' + ddmmyyyy[1];
                }
                var d = new Date(dateInput);
                if (isNaN(d.getTime())) return String(dateInput);
                var days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
                var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                var dayStr = String(d.getUTCDate()).padStart(2, '0');
                return days[d.getUTCDay()] + ' ' + dayStr + ' ' + months[d.getUTCMonth()] + ' ' + d.getUTCFullYear();
            }

            function switchSession(mId) {
                if (!mId || !simulationData.sessions[mId]) return;
                activeSessionId = mId;

                // 1. Hide all session workspaces and reveal target
                document.querySelectorAll('.milestone-workspace').forEach(function(ws) {
                    ws.style.display = 'none';
                });
                var targetWs = document.getElementById('sessionWorkspace-' + mId);
                if (targetWs) targetWs.style.display = 'flex';

                var session = simulationData.sessions[mId];
                var isLastSession = (mId === simulationData.liveSessionId);

                // 2. Highlight selector and labels in red if viewing a past historical session
                var select = document.getElementById('milestoneSelect');
                if (select) {
                    select.value = mId;
                    if (!isLastSession) {
                        select.style.borderColor = "#dc2626";
                        select.style.color = "#b91c1c";
                        select.style.backgroundColor = "#fef2f2";
                        select.style.fontWeight = "700";
                    } else {
                        select.style.borderColor = "#cbd5e1";
                        select.style.color = "#0f172a";
                        select.style.backgroundColor = "#ffffff";
                        select.style.fontWeight = "600";
                    }
                }

                // 3. Update Left Sidebar Metrics
                var metrics = session.sessionMetrics || (isLastSession ? simulationData.lastLiveSimulationState : null);
                if (metrics) {
                    renderSidebarMetrics(metrics, !isLastSession);
                }

                // 4. Render Stage Tabs for this session
                renderStagesBar(session.stages || [], mId);

                // 5. Select first stage in this session
                var firstStage = (session.stages && session.stages.length > 0) ? session.stages[0].num : 1;
                switchStage(firstStage, mId);

                // 6. Refresh Performance Chart slice
                refreshPerformanceChart(session.date);

                // 7. Ensure doughnut charts for this session are rendered or fallback shown
                renderSessionDoughnuts(session);
            }

            function switchStage(stageNum, mId) {
                if (!mId) mId = activeSessionId;
                activeStageNum = stageNum;

                var ws = document.getElementById('sessionWorkspace-' + mId);
                if (ws) {
                    ws.querySelectorAll('.stage-workspace').forEach(function(sWs) {
                        sWs.style.display = 'none';
                    });
                    var targetStageEl = document.getElementById('session-' + mId + '-stage-' + stageNum);
                    if (targetStageEl) targetStageEl.style.display = 'flex';
                }

                document.querySelectorAll('.stage-item').forEach(function(tab) {
                    var sNum = parseInt(tab.getAttribute('data-stage-num'), 10);
                    tab.classList.remove('viewing');
                    if (sNum === stageNum) {
                        tab.classList.add('viewing');
                    }
                });

                // Ensure doughnut charts resize or initialize if switching to Stage 3 or Stage 6
                if (stageNum === 3 || stageNum === 6) {
                    var session = simulationData.sessions[mId];
                    if (session) {
                        renderSessionDoughnuts(session);
                        if (stageNum === 3 && doughnutChartInstances['sec_' + mId]) {
                            try { doughnutChartInstances['sec_' + mId].resize(); } catch(e){}
                        }
                        if (stageNum === 6 && doughnutChartInstances['port_' + mId]) {
                            try { doughnutChartInstances['port_' + mId].resize(); } catch(e){}
                        }
                    }
                }
            }

            function renderStagesBar(stages, mId) {
                var bar = document.getElementById('stagesScrollContainer');
                if (!bar) return;
                bar.innerHTML = "";

                stages.forEach(function(stage) {
                    var div = document.createElement('div');
                    div.className = 'stage-item completed';
                    div.setAttribute('data-stage-num', stage.num);
                    div.id = 'stageTab-' + mId + '-' + stage.num;
                    div.innerText = 'Stage ' + stage.num + ': ' + stage.name;
                    div.onclick = function() {
                        switchStage(stage.num, mId);
                    };
                    bar.appendChild(div);
                });
            }

            function renderSidebarMetrics(data, isPast) {
                if (!data) return;

                var dateLabel = document.getElementById("metricDateLabel");
                var cashLabel = document.getElementById("metricCashLabel");
                var stockLabel = document.getElementById("metricStockLabel");
                var totalLabel = document.getElementById("metricTotalLabel");
                [dateLabel, cashLabel, stockLabel, totalLabel].forEach(function(lbl) {
                    if (lbl) {
                        if (isPast) {
                            lbl.style.color = "#dc2626";
                            lbl.style.fontWeight = "700";
                        } else {
                            lbl.style.color = "";
                            lbl.style.fontWeight = "";
                        }
                    }
                });

                var dateDisplay = document.getElementById("currentSimDateDisplay");
                if (dateDisplay) {
                    dateDisplay.textContent = data.currentDateFriendly || formatFriendlyDate(data.currentDate || data.date);
                }

                if (data.cashValue !== undefined) {
                    document.getElementById("metricCashVal").textContent = '$' + Number(data.cashValue).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                }
                if (data.stockValue !== undefined) {
                    document.getElementById("metricStockVal").textContent = '$' + Number(data.stockValue).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                }
                if (data.totalValue !== undefined) {
                    document.getElementById("metricTotalVal").textContent = '$' + Number(data.totalValue).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                }

                var fundRetEl = document.getElementById("metricFundReturn");
                if (fundRetEl && data.totalReturnPct !== undefined) {
                    var ret = Number(data.totalReturnPct);
                    var retDollar = Number(data.totalReturnDollar || 0);
                    var sign = ret >= 0 ? '+' : '';
                    fundRetEl.textContent = sign + ret.toFixed(2) + '% ($' + Math.abs(retDollar).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ')';
                    fundRetEl.className = 'metric-change ' + (ret >= 0 ? 'positive' : 'negative');
                }

                var fundRetPctEl = document.getElementById("metricFundReturnPct");
                if (fundRetPctEl && data.totalReturnPct !== undefined) {
                    var r = Number(data.totalReturnPct);
                    fundRetPctEl.textContent = (r >= 0 ? '+' : '') + r.toFixed(2) + '%';
                    fundRetPctEl.style.color = r >= 0 ? '#10b981' : '#dc2626';
                }

                var alphaEl = document.getElementById("metricAlpha");
                if (alphaEl && data.alphaPct !== undefined) {
                    var a = Number(data.alphaPct);
                    var spRet = Number(data.sp500ReturnPct || 0);
                    alphaEl.innerHTML = '<span style="font-weight: 800;">' + (a >= 0 ? '+' : '') + a.toFixed(2) + '%</span> <span style="font-size: 11px; font-weight: 500; color: #64748b;">(' + (spRet >= 0 ? '+' : '') + spRet.toFixed(2) + '%)</span>';
                    alphaEl.style.color = a >= 0 ? '#10b981' : '#dc2626';
                }

                if (data.sharpeRatio !== undefined) {
                    var sharpeEl = document.getElementById("metricSharpe");
                    if (sharpeEl) {
                        var sr = Number(data.sharpeRatio).toFixed(2);
                        var spSr = (data.sp500SharpeRatio !== undefined && data.sp500SharpeRatio !== null) ? Number(data.sp500SharpeRatio).toFixed(2) : null;
                        if (spSr !== null) {
                            sharpeEl.innerHTML = '<span style="font-weight: 800;">' + sr + '</span> <span style="font-size: 11px; font-weight: 500; color: #64748b;">(vs ' + spSr + ' for S&P 500)</span>';
                        } else {
                            sharpeEl.textContent = sr;
                        }
                    }
                }

                if (data.maxDrawdownPct !== undefined) {
                    var ddEl = document.getElementById("metricDrawdown");
                    if (ddEl) {
                        var dd = Number(data.maxDrawdownPct);
                        var spDd = (data.sp500MaxDrawdownPct !== undefined && data.sp500MaxDrawdownPct !== null) ? Number(data.sp500MaxDrawdownPct) : null;
                        if (spDd !== null) {
                            ddEl.innerHTML = '<span style="font-weight: 800; color: ' + (dd <= spDd ? '#10b981' : '#dc2626') + ';">' + dd.toFixed(2) + '%</span> <span style="font-size: 11px; font-weight: 500; color: #64748b;">(vs ' + spDd.toFixed(2) + '% for S&P 500)</span>';
                        } else {
                            ddEl.textContent = dd.toFixed(2) + '%';
                            ddEl.style.color = dd > 0 ? '#dc2626' : '#10b981';
                        }
                    }
                }

                if (data.positions) {
                    renderHoldingsTable(data.positions);
                }

                if (data.rebalanceHistory || data.systemLogs) {
                    renderActivityFeed(data.rebalanceHistory, data.systemLogs);
                }
            }

            function renderHoldingsTable(positions) {
                var tbody = document.getElementById("holdingsTableBody");
                var countBadge = document.getElementById("holdingsCountBadge");
                if (!tbody) return;

                if (countBadge) countBadge.textContent = positions.length + ' stock' + (positions.length === 1 ? '' : 's');

                if (!positions || positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #94a3b8; padding: 16px 0; font-size: 12px;">No positions</td></tr>';
                    return;
                }

                tbody.innerHTML = positions.map(function(p, idx) {
                    var rowBg = idx % 2 === 0 ? "#ffffff" : "#f8fafc";
                    var isPos = p.pnlPct >= 0;
                    var pnlColor = isPos ? "#10b981" : "#dc2626";
                    var sign = isPos ? "+" : "";
                    var sharesStr = Number(p.shares) % 1 === 0 ? Number(p.shares).toFixed(0) : Number(p.shares).toFixed(1);
                    var compName = p.companyName || "";

                    return '<tr style="background: ' + rowBg + '; border-bottom: 1px solid #f1f5f9;">' +
                        '<td style="padding: 5px 3px; text-align: left;">' +
                            '<div class="pos-ticker">' + escapeHtml(p.ticker) + '</div>' +
                            '<div class="pos-company-sub" title="' + escapeHtml(compName) + '">' + escapeHtml(compName) + '</div>' +
                        '</td>' +
                        '<td style="padding: 5px 3px; text-align: right; font-weight: 600; font-size: 11px; color: #334155; white-space: nowrap;">' + sharesStr + '</td>' +
                        '<td style="padding: 5px 3px; text-align: right; font-size: 11px; color: #475569; white-space: nowrap;">$' + Number(p.avgPrice).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + '</td>' +
                        '<td style="padding: 5px 3px; text-align: right; font-weight: 600; font-size: 11px; color: #0f172a; white-space: nowrap;">$' + Number(p.currentPrice).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + '</td>' +
                        '<td style="padding: 5px 3px; text-align: right; font-weight: 700; font-size: 11px; color: #0f172a; white-space: nowrap;">$' + Number(p.currentValue).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }) + '</td>' +
                        '<td style="padding: 5px 3px; text-align: right; font-weight: 600; font-size: 11px; color: #2563eb; white-space: nowrap;">' + Number(p.weightPct).toFixed(1) + '%</td>' +
                        '<td style="padding: 5px 3px; text-align: right; white-space: nowrap;">' +
                            '<div style="font-weight: 700; font-size: 11px; color: ' + pnlColor + '; line-height: 1.15;">' + sign + Number(p.pnlPct).toFixed(1) + '%</div>' +
                            '<div style="font-weight: 600; font-size: 10px; color: ' + pnlColor + '; line-height: 1.1;">' + sign + '$' + Math.abs(p.pnlDollar).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }) + '</div>' +
                        '</td>' +
                    '</tr>';
                }).join('');
            }

            function renderActivityFeed(history, systemLogs) {
                var container = document.getElementById("rebalanceTimelineList");
                if (!container) return;

                var events = [];
                if (history && history.length > 0) {
                    history.forEach(function(item) {
                        events.push({
                            kind: 'rebalance',
                            date: item.date,
                            dateFormatted: formatFriendlyDate(item.dateFormatted || item.date),
                            type: item.type,
                            summary: item.summary
                        });
                    });
                }

                if (systemLogs && systemLogs.length > 0) {
                    systemLogs.forEach(function(log) {
                        events.push({
                            kind: 'systemLog',
                            date: log.dateStr || log.date,
                            dateFormatted: formatFriendlyDate(log.dateStr || log.date),
                            title: log.title || 'System Log',
                            ticker: log.ticker || '',
                            message: log.message || ''
                        });
                    });
                }

                if (events.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: #94a3b8; padding: 12px 0; font-size: 11px;">Logs are empty</div>';
                    return;
                }

                events.sort(function(a, b) {
                    var timeA = a.date ? new Date(a.date).getTime() : 0;
                    var timeB = b.date ? new Date(b.date).getTime() : 0;
                    return timeB - timeA;
                });

                container.innerHTML = events.map(function(ev) {
                    if (ev.kind === 'rebalance') {
                        var typeDisplay = ev.type === 'inception' ? 'INCEPTION' : (ev.type === 'skipped' ? 'SKIPPED' : 'REBALANCED');
                        return '<div class="timeline-item ' + ev.type + '">' +
                            '<span class="timeline-date">' + ev.dateFormatted + ' (' + typeDisplay + ')</span>' +
                            '<span class="timeline-summary">' + escapeHtml(ev.summary || 'Milestone review processed.') + '</span>' +
                        '</div>';
                    } else {
                        var badgeClass = 'badge-action-log';
                        var titleLower = (ev.title || "").toLowerCase();
                        if (titleLower.includes('buy')) badgeClass = 'badge-action-buy';
                        else if (titleLower.includes('sell')) badgeClass = 'badge-action-sell';
                        else if (titleLower.includes('fail') || titleLower.includes('error') || titleLower.includes('cancel')) badgeClass = 'badge-action-cancel';

                        return '<div class="log-pill"><div>' +
                            '<span class="log-date-tag">' + ev.dateFormatted + '</span>' +
                            '<span class="' + badgeClass + '">' + escapeHtml(ev.title) + '</span>' +
                            (ev.ticker ? '<span class="log-ticker-tag">' + escapeHtml(ev.ticker) + '</span>' : '') +
                            '<span class="log-message-body">' + escapeHtml(ev.message) + '</span>' +
                        '</div></div>';
                    }
                }).join('');
            }

            function setChartTimeframe(tf, btn) {
                currentChartTf = tf.toLowerCase();
                document.querySelectorAll('.tf-btn').forEach(function(b) { b.classList.remove('active'); });
                if (btn) btn.classList.add('active');
                refreshPerformanceChart();
            }

            function refreshPerformanceChart(upToDate) {
                var canvas = document.getElementById('agentSimChartCanvas');
                var fallbackImg = document.getElementById('agentSimChartCanvas_fallback');

                if (typeof Chart === 'undefined') {
                    if (canvas) canvas.style.display = 'none';
                    if (fallbackImg) fallbackImg.style.display = 'block';
                    return;
                }

                var portPoints = simulationData.fullPortfolioPoints || [];
                var spPoints = simulationData.fullSp500Points || [];

                if (upToDate) {
                    portPoints = portPoints.filter(function(p) { return p.x <= upToDate; });
                    spPoints = spPoints.filter(function(p) { return p.x <= upToDate; });
                }

                if (currentChartTf !== 'all' && portPoints.length > 0) {
                    var lastDate = new Date(portPoints[portPoints.length - 1].x);
                    var cutoff = new Date(lastDate);
                    if (currentChartTf === '1m') cutoff.setMonth(cutoff.getMonth() - 1);
                    else if (currentChartTf === '3m') cutoff.setMonth(cutoff.getMonth() - 3);
                    else if (currentChartTf === '1y') cutoff.setFullYear(cutoff.getFullYear() - 1);
                    else if (currentChartTf === '3y') cutoff.setFullYear(cutoff.getFullYear() - 3);

                    var cutoffStr = cutoff.toISOString().split('T')[0];
                    portPoints = portPoints.filter(function(p) { return p.x >= cutoffStr; });
                    spPoints = spPoints.filter(function(p) { return p.x >= cutoffStr; });
                }

                if (!perfChartInstance && canvas) {
                    try {
                        perfChartInstance = new Chart(canvas.getContext('2d'), {
                            type: 'line',
                            data: {
                                labels: portPoints.map(function(p) { return p.x; }),
                                datasets: [
                                    {
                                        label: 'Agent Portfolio',
                                        data: portPoints.map(function(p) { return p.y; }),
                                        borderColor: '#2563eb',
                                        backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                        borderWidth: 2,
                                        fill: true,
                                        tension: 0.25,
                                        pointRadius: 0
                                    },
                                    {
                                        label: 'S&P 500 Benchmark',
                                        data: spPoints.map(function(p) { return p.y; }),
                                        borderColor: '#f59e0b',
                                        borderWidth: 2,
                                        borderDash: [5, 5],
                                        fill: false,
                                        tension: 0.25,
                                        pointRadius: 0
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                animation: false,
                                plugins: {
                                    legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 10, weight: 'bold' } } },
                                    tooltip: {
                                        enabled: true,
                                        callbacks: {
                                            label: function(ctx) { return ' ' + ctx.dataset.label + ': $' + Number(ctx.raw).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
                                        }
                                    }
                                },
                                scales: {
                                    x: { grid: { display: false }, ticks: { maxTicksLimit: 5, font: { size: 9 }, color: '#94a3b8' } },
                                    y: {
                                        grid: { color: '#f1f5f9' },
                                        ticks: {
                                            font: { size: 9 },
                                            color: '#94a3b8',
                                            callback: function(v) { return '$' + (v >= 1e6 ? (v/1e6).toFixed(1)+'M' : (v >= 1e3 ? (v/1e3).toFixed(0)+'k' : v)); }
                                        }
                                    }
                                }
                            }
                        });
                        if (canvas) canvas.style.display = 'block';
                        if (fallbackImg) fallbackImg.style.display = 'none';
                    } catch (e) {
                        console.warn("Chart.js line chart error:", e);
                        if (canvas) canvas.style.display = 'none';
                        if (fallbackImg) fallbackImg.style.display = 'block';
                    }
                } else if (perfChartInstance) {
                    perfChartInstance.data.labels = portPoints.map(function(p) { return p.x; });
                    perfChartInstance.data.datasets[0].data = portPoints.map(function(p) { return p.y; });
                    perfChartInstance.data.datasets[1].data = spPoints.map(function(p) { return p.y; });
                    perfChartInstance.update();
                }
            }

            function renderSessionDoughnuts(session) {
                var mId = session.id;
                var secCanvas = document.getElementById('sectorChartCanvas-' + mId);
                var secFallback = document.getElementById('sectorChartCanvas-' + mId + '_fallback');
                var portCanvas = document.getElementById('portChartCanvas-' + mId);
                var portFallback = document.getElementById('portChartCanvas-' + mId + '_fallback');

                if (typeof Chart === 'undefined') {
                    if (secCanvas) secCanvas.style.display = 'none';
                    if (secFallback) secFallback.style.display = 'block';
                    if (portCanvas) portCanvas.style.display = 'none';
                    if (portFallback) portFallback.style.display = 'block';
                    return;
                }

                var colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#64748b", "#14b8a6", "#f97316", "#a855f7", "#3b82f6", "#e11d48"];

                // Sector Doughnut
                if (secCanvas && session.confirmedSector) {
                    if (!doughnutChartInstances['sec_' + mId]) {
                        try {
                            var secAlloc = session.confirmedSector.confirmedAllocation || session.confirmedSector;
                            var secDict = secAlloc.sectorAllocations || {};
                            var secLabels = [];
                            var secVals = [];
                            for (var k in secDict) {
                                var sObj = secDict[k];
                                var sName = (typeof sObj === 'object' && sObj.sector) ? sObj.sector : k;
                                var sPct = (typeof sObj === 'object' && sObj.allocationPct !== undefined) ? parseFloat(sObj.allocationPct) : parseFloat(sObj);
                                if (sPct > 0) {
                                    secLabels.push(sName);
                                    secVals.push(sPct);
                                }
                            }
                            if (secLabels.length > 0) {
                                doughnutChartInstances['sec_' + mId] = new Chart(secCanvas.getContext('2d'), {
                                    type: 'doughnut',
                                    data: {
                                        labels: secLabels,
                                        datasets: [{ data: secVals, backgroundColor: colors.slice(0, secLabels.length), borderWidth: 2, borderColor: '#fff' }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10.5, weight: '600' } } } },
                                        cutout: '58%'
                                    }
                                });
                                secCanvas.style.display = 'block';
                                if (secFallback) secFallback.style.display = 'none';
                            }
                        } catch (e) {
                            console.warn("Sector chart error:", e);
                            if (secCanvas) secCanvas.style.display = 'none';
                            if (secFallback) secFallback.style.display = 'block';
                        }
                    } else {
                        try {
                            doughnutChartInstances['sec_' + mId].resize();
                        } catch(e) {}
                    }
                }

                // Portfolio Doughnut
                if (portCanvas && session.confirmedPortfolio) {
                    if (!doughnutChartInstances['port_' + mId]) {
                        try {
                            var pObj = session.confirmedPortfolio.confirmedPortfolio || session.confirmedPortfolio;
                            var positions = pObj.positions || [];
                            var indAgg = {};
                            positions.forEach(function(p) {
                                var ind = p.industry || p.sector || "General";
                                indAgg[ind] = (indAgg[ind] || 0) + (parseFloat(p.weightPct) || 0);
                            });
                            var portLabels = Object.keys(indAgg);
                            var portVals = Object.values(indAgg);
                            if (portLabels.length > 0) {
                                doughnutChartInstances['port_' + mId] = new Chart(portCanvas.getContext('2d'), {
                                    type: 'doughnut',
                                    data: {
                                        labels: portLabels,
                                        datasets: [{ data: portVals, backgroundColor: colors.slice(0, portLabels.length), borderWidth: 2, borderColor: '#fff' }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10.5, weight: '600' } } } },
                                        cutout: '58%'
                                    }
                                });
                                portCanvas.style.display = 'block';
                                if (portFallback) portFallback.style.display = 'none';
                            }
                        } catch (e) {
                            console.warn("Portfolio chart error:", e);
                            if (portCanvas) portCanvas.style.display = 'none';
                            if (portFallback) portFallback.style.display = 'block';
                        }
                    } else {
                        try {
                            doughnutChartInstances['port_' + mId].resize();
                        } catch(e) {}
                    }
                }
            }

            document.addEventListener('DOMContentLoaded', function() {
                // Collapsible blocks toggle
                document.body.addEventListener('click', function(e) {
                    var header = e.target.closest('.collapsible-header');
                    if (header) {
                        var block = header.closest('.collapsible-block');
                        if (block) block.classList.toggle('collapsed');
                    }
                });

                // Populate milestone dropdown options
                var select = document.getElementById('milestoneSelect');
                if (select) {
                    select.innerHTML = '';
                    var sessList = Object.values(simulationData.sessions || {});
                    sessList.sort(function(a, b) {
                        var numA = parseInt(String(a.id).replace(/\\D/g, '') || '0', 10);
                        var numB = parseInt(String(b.id).replace(/\\D/g, '') || '0', 10);
                        return numA - numB;
                    });

                    sessList.forEach(function(s) {
                        var opt = document.createElement('option');
                        opt.value = s.id;
                        opt.textContent = s.label || s.id;
                        select.appendChild(opt);
                    });

                    select.onchange = function() {
                        switchSession(this.value);
                    };
                }

                // Initialise viewing state
                var defaultSessionId = simulationData.liveSessionId || (sessList.length > 0 ? sessList[sessList.length - 1].id : "milestone_0");
                switchSession(defaultSessionId);
            });
        <\/script>
        `;
    }

    async function generateSelfContainedHtml() {
        const meta = extractSimulationMetadata();
        const inlinedCss = await collectAllStylesheets();

        let serverName = "Unknown";
        let modelName = "";
        let timeTaken = lastSimTotalTime;

        if (!timeTaken) {
            const timerEl = document.getElementById("simulationTimer");
            if (timerEl && timerEl.innerText && timerEl.innerText !== "0 mins 0.00 secs") {
                timeTaken = timerEl.innerText.trim();
            }
        }
        if (!timeTaken) timeTaken = "Completed";

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
            modelHtml = `<div><span class="report-meta-label">Model:</span><span class="report-meta-value" title="${escapeHtml(modelName)}">${escapeHtml(modelName)}</span></div>`;
        }

        // Top Header
        const paramsHtml = meta.params.map(p => `
            <span class="report-param-item">
                <strong>${escapeHtml(p.key)}:</strong> ${escapeHtml(p.value)}
            </span>
        `).join("");

        const headerHtml = `
        <header class="report-header">
            <div class="report-header-brand">
                <h1>CapitalAgents</h1>
            </div>
            <div class="report-header-center">
                <h2 class="report-header-title">${escapeHtml(meta.title)}</h2>
                <div class="report-params-list">
                    ${paramsHtml}
                </div>
            </div>
            <div class="report-header-right">
                <div><span class="report-meta-label">Server:</span><span class="report-meta-value">${escapeHtml(serverName)}</span></div>
                ${modelHtml}
                <div><span class="report-meta-label">Time:</span><span class="report-meta-value">${escapeHtml(timeTaken)}</span></div>
            </div>
        </header>
        `;

        // Stages Bar Clone
        const origStagesBar = document.getElementById("stagesBar");
        let stagesBarHtml = "";
        if (origStagesBar) {
            const stagesClone = origStagesBar.cloneNode(true);
            const stagesActions = stagesClone.querySelector("#stagesActions");
            if (stagesActions) {
                stagesActions.innerHTML = ""; // Clean actions in export
            }
            stagesBarHtml = stagesClone.outerHTML;
        }

        // Main Layout Clone with Left Sidebar and Workspaces
        const origLayout = document.querySelector(".main-layout");
        let mainLayoutHtml = "";
        let snapshots = {};

        if (origLayout) {
            const layoutClone = origLayout.cloneNode(true);

            // Remove settings overlays, QA elements if any
            layoutClone.querySelectorAll("#settingsOverlay, .qa-delete-btn, .qa-user-pane, .modal-close-btn").forEach(el => el.remove());

            // Collapse all thinking blocks by default
            layoutClone.querySelectorAll(".collapsible-block").forEach(block => {
                block.classList.add("collapsed");
            });

            // Freeze canvases into base64 snapshots and attach companion fallback img tags
            snapshots = freezeCanvases(origLayout, layoutClone);

            mainLayoutHtml = layoutClone.outerHTML;
        }

        const serializedData = serializeSimulationState();

        const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${escapeHtml(meta.title)} - CapitalAgents</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"><\/script>
    <style>
        ${inlinedCss}
    </style>
    <noscript>
        <style>
            canvas { display: none !important; }
            .chart-fallback-img { display: block !important; }
        </style>
    </noscript>
</head>
<body>
    ${headerHtml}
    ${stagesBarHtml}
    ${mainLayoutHtml}
    ${buildOfflineScript(serializedData, snapshots)}
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
            console.error("Error exporting agent sim report:", e);
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
        } else {
            const timerEl = document.getElementById("simulationTimer");
            if (timerEl && timerEl.innerText && timerEl.innerText !== "0 mins 0.00 secs") {
                lastSimTotalTime = timerEl.innerText.trim();
            }
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
        generateSelfContainedHtml,
        extractSimulationMetadata
    };
})();

window.AgentSimExporter = AgentSimExporter;
