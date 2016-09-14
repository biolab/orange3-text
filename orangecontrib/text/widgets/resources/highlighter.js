/**
 * Adopted from https://jsfiddle.net/julmot/ova17daa/
 */
var DEBUG = false;

var mark = function(pattern) {
    var flags = 'gmi';
    try {
        var regex = new RegExp(pattern, flags);

        // Mark regex inside the context if regex is valid
        $(".mark-area").unmark();
        $(".mark-area").markRegExp(regex);
    } catch(e) {    // skip invalid regexes
        if (DEBUG) {console.log('[INFO] Skipping marking for invalid regex pattern: ' + pattern)}
    }
};
