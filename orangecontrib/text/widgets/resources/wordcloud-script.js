// Options for WordCloud
var OPTIONS = {
    list: [],
    fontFamily: 'sans-serif',
    fontWeight: 300,
    color: 'UNUSED',
    backgroundColor: 'white',
    minSize: 7,
    weightFactor: 1,
    gridSize: 12,
    clearCanvas: true,
    minRotation: -Math.PI/10,
    maxRotation: Math.PI/10,
    rotateRatio: 1,
    rotationSteps: 3
};

// Redraw wordcloud when the window size changes
function redrawWordCloud() {
    WordCloud(document.getElementById('canvas'), OPTIONS);
}
window.addEventListener('resize', redrawWordCloud);

// Select words or clear selection
window.addEventListener('click', function(event) {
    var span = event.target;
    if (span.tagName == 'SPAN') {
        // Allow multiselection only if modifier key pressed
        if (! (event.ctrlKey || event.shiftKey)) {
            clearSelection();
        }
        span.className = span.className !== 'selected' ? 'selected': '';
    } else if (span.tagName == 'BODY') {
        clearSelection();
    }
    // Signal selection back to Qt
    var words = [],
        spans = document.getElementsByTagName('span');
    for (var i=0; i < spans.length; ++i) {
        var span = spans[i];
        if (span.className === 'selected')
            words.push(span.innerHTML);
    }
    SELECTED_WORDS = words;
    pybridge.update_selection(words)
});
function clearSelection() {
    var spans = document.getElementsByTagName('span');
    for (var i=0; i < spans.length; ++i) {
        spans[i].className = '';
    }
}

// Mark words in SELECTED_WORDS list selected
var SELECTED_WORDS = [];
function selectWords() {
    var lookup = {};
    var spans = document.getElementsByTagName('span');
    for (var i=0; i<SELECTED_WORDS.length; ++i) {
        lookup[SELECTED_WORDS[i]] = true;
    }
    for (var i=0; i<spans.length; ++i) {
        spans[i].className = lookup[spans[i].innerHTML] ? 'selected' : '';
    }
}
document.getElementById('canvas').addEventListener('wordcloudstop', selectWords);
