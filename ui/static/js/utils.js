// CapitalAgents - Shared Frontend Utilities

(function(window) {
    'use strict';

    const Utils = {};

    // Format raw token count into human-readable shorthand (K, mn, bn, tn)
    Utils.formatTokenCount = function(val) {
        if (val == null || isNaN(val) || val <= 0) return "0";
        const num = Math.abs(parseFloat(val));
        if (num >= 1e12) return Number((num / 1e12).toPrecision(3)) + "tn";
        if (num >= 1e9) return Number((num / 1e9).toPrecision(3)) + "bn";
        if (num >= 1e6) return Number((num / 1e6).toPrecision(3)) + "mn";
        if (num >= 1e3) return Number((num / 1e3).toPrecision(3)) + "K";
        return Number(num.toPrecision(3)).toString();
    };

    // Format generation speed in tokens per second
    Utils.formatTokSpeed = function(val) {
        if (val == null || isNaN(val) || val <= 0) return "0.00 tok/s";
        if (val < 1000) {
            return parseFloat(val).toFixed(1) + " tok/s";
        } else {
            return Math.round(val).toString() + " tok/s";
        }
    };

    // Format cache hit percentage to one decimal place
    Utils.formatCacheHitRate = function(val) {
        if (val == null || isNaN(val)) return "0.0%";
        return parseFloat(val).toFixed(1) + "%";
    };

    // Format money cost values with dynamic precision
    Utils.formatCost = function(val) {
        if (val == null || isNaN(val)) return "$0.0000";
        const num = parseFloat(val);
        if (num < 10) return "$" + num.toFixed(4);
        return "$" + num.toFixed(2);
    };

    // Format numeric dollar input with thousand separators and decimal restriction
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

    // Calculate the number of decimal places in a numeric value
    Utils.countDecimalPlaces = function(num) {
        if (!num || Math.floor(num) === num) return 0;
        const str = num.toString();
        if (str.indexOf('e-') !== -1) {
            return parseInt(str.split('e-')[1], 10);
        }
        if (str.indexOf('.') !== -1) {
            return str.split('.')[1].length;
        }
        return 0;
    };

    // Format large currency amounts using standard magnitude suffixes
    Utils.formatLargeCurrency = function(val, stepSize) {
        if (val === 0) return '$0';
        const absVal = Math.abs(val);
        let unit = '';
        let div = 1;

        if (absVal >= 1e12) { unit = 'tn'; div = 1e12; }
        else if (absVal >= 1e9) { unit = 'bn'; div = 1e9; }
        else if (absVal >= 1e6) { unit = 'mn'; div = 1e6; }
        else if (absVal >= 1e3) { unit = 'K'; div = 1e3; }

        const num = val / div;

        let dp = 0;
        if (stepSize && stepSize > 0) {
            dp = Utils.countDecimalPlaces(stepSize / div);
        } else {
            dp = Utils.countDecimalPlaces(num);
        }
        dp = Math.max(0, Math.min(6, dp));

        let minDp = 0;
        if (div === 1 && stepSize && stepSize < 1) {
            minDp = 2;
        }

        const formattedStr = num.toLocaleString('en-US', {
            minimumFractionDigits: minDp,
            maximumFractionDigits: dp
        });

        return '$' + formattedStr + unit;
    };

    // Retrieve yesterday's date formatted as YYYY-MM-DD
    Utils.getYesterdayDateString = function() {
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        return yesterday.toISOString().split('T')[0];
    };

    // Append log line to terminal element with autoscroll support
    Utils.appendTerminalLog = function(elementOrId, line, threshold = 40) {
        const logsEl = typeof elementOrId === 'string' ? document.getElementById(elementOrId) : elementOrId;
        if (!logsEl) return;
        const isAtBottom = (logsEl.scrollHeight - logsEl.scrollTop - logsEl.clientHeight) <= threshold;
        logsEl.textContent += (line || "") + "\n";
        if (isAtBottom) {
            logsEl.scrollTop = logsEl.scrollHeight;
        }
    };

    // Export to global window object
    window.AppUtils = Utils;

    // Direct global convenience bindings for backwards compatibility
    window.formatTokenCount = Utils.formatTokenCount;
    window.formatTokSpeed = Utils.formatTokSpeed;
    window.formatCacheHitRate = Utils.formatCacheHitRate;
    window.formatCost = Utils.formatCost;
    window.formatDollarInput = Utils.formatDollarInput;
    window.appendTerminalLog = Utils.appendTerminalLog;
    window.getYesterdayDateString = Utils.getYesterdayDateString;

})(window);
