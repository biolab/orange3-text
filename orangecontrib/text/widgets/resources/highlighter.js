/**
 * Adopted from https://jsfiddle.net/julmot/ova17daa/
 */

var mark = function(pattern) {

    // Create regex
    var flags = 'gmi';
    var regex = new RegExp(pattern, flags);

    // Mark the regex inside the context
    $(".mark-area").unmark();
    $(".mark-area").markRegExp(regex);
};
