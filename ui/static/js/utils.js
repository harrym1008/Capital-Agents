/**
 * CapitalAgents - Shared Frontend Utilities
 */

(function(window) {
    'use strict';

    const Utils = {};

    // --- Formatting Helpers ---

    Utils.formatTokenCount = function(val) {
        if (val === undefined || val === null || isNaN(val) || val <= 0) return "0";
        const num = Math.abs(parseFloat(val));
        if (num >= 1e9) return Number((num / 1e9).toPrecision(3)) + "b";
        if (num >= 1e6) return Number((num / 1e6).toPrecision(3)) + "m";
        if (num >= 1e3) return Number((num / 1e3).toPrecision(3)) + "k";
        return Number(num.toPrecision(3)).toString();
    };

    Utils.formatTokSpeed = function(val) {
        if (val === undefined || val === null || isNaN(val) || val <= 0) return "0.0 tok/s";
        if (val < 1000) {
            return parseFloat(val).toFixed(1) + " tok/s";
        } else {
            return Math.round(val).toString() + " tok/s";
        }
    };

    Utils.formatCacheHitRate = function(val) {
        if (val === undefined || val === null || isNaN(val)) return "0.0%";
        return parseFloat(val).toFixed(1) + "%";
    };

    Utils.formatCost = function(val) {
        if (val === undefined || val === null || isNaN(val)) return "$0.0000";
        const num = parseFloat(val);
        if (num < 10) return "$" + num.toFixed(4);
        return "$" + num.toFixed(2);
    };

    Utils.formatPrice = function(val, decimals = 2) {
        if (val === undefined || val === null || isNaN(val)) return "$0.00";
        return "$" + Number(val).toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    };

    Utils.formatDollarInput = function(input, decimalPlaces = 2) {
        if (!input) return;
        let val = input.value.replace(/[^0-9.]/g, '');
        const parts = val.split('.');
        if (parts.length > 2) {
            val = parts[0] + '.' + parts.slice(1).join('');
        }
        if (parts.length === 2 && parts[1].length > decimalPlaces) {
            val = parts[0] + '.' + parts[1].slice(0, decimalPlaces);
        }
        const wholeParts = val.split('.');
        wholeParts[0] = wholeParts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        input.value = wholeParts.join('.');
    };

    // --- Date Helpers ---

    Utils.getYesterdayDateString = function() {
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        return yesterday.toISOString().split('T')[0];
    };

    // --- Terminal Log Autoscroll Helper ---

    Utils.appendTerminalLog = function(elementOrId, line, threshold = 40) {
        const logsEl = typeof elementOrId === 'string' ? document.getElementById(elementOrId) : elementOrId;
        if (!logsEl) return;
        const isAtBottom = (logsEl.scrollHeight - logsEl.scrollTop - logsEl.clientHeight) <= threshold;
        logsEl.textContent += (line || "") + "\n";
        if (isAtBottom) {
            logsEl.scrollTop = logsEl.scrollHeight;
        }
    };

    // --- Modal Overlay Dismissal Helper ---

    Utils.setupModalOverlayDismissal = function(overlayEl, onCloseCallback) {
        if (!overlayEl) return;
        overlayEl.addEventListener('mousedown', function(event) {
            if (event.target === overlayEl) {
                if (typeof onCloseCallback === 'function') {
                    onCloseCallback();
                } else {
                    overlayEl.style.display = 'none';
                }
            }
        });
    };

    // Export to global window object
    window.AppUtils = Utils;

    // Direct global convenience bindings for backward compatibility
    window.formatTokenCount = Utils.formatTokenCount;
    window.formatTokSpeed = Utils.formatTokSpeed;
    window.formatCacheHitRate = Utils.formatCacheHitRate;
    window.formatCost = Utils.formatCost;
    window.formatDollarInput = Utils.formatDollarInput;
    window.appendTerminalLog = Utils.appendTerminalLog;
    window.getYesterdayDateString = Utils.getYesterdayDateString;

})(window);
