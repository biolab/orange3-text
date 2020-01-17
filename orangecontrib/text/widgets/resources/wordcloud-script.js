// Options for WordCloud
var OPTIONS = {
    list: [],
    // according to www good selection that covers all systems fonts - same than used by QT
    fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen-Sans, Ubuntu, Cantarell, Helvetica Neue, sans-serif',
    color: 'UNUSED',
    backgroundColor: 'white',
    clearCanvas: true,
    minRotation: -Math.PI/10,
    maxRotation: Math.PI/10,
    rotateRatio: 1,
    rotationSteps: 3,
    weightFactor: weight_factor,
    gridSize: 7,
    shrinkToFit: true,
    drawOutOfBound: false,
    shuffle: false
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

function combined_width() {
    /*
    This function calculates combine width of the word cloud assuming that
    cloud has a shape of ellipse which height is 0.65 % of width. It returns
    smaller of width or ~1.5 * height.
     */
    var width = document.getElementById("canvas").clientWidth;
    var height = document.getElementById("canvas").clientHeight;
    // 0.65 is the ratio between width and height of ellipsis
    return Math.min(width, 1.0 / 0.65 * height)
}

function weight_factor(size) {
    const recalculated_width = combined_width();
    // with this parameter we partially bring in the average word size
    // combined with lenght (many big characters mean decrease size more)
    size = size * (1/2 + 1/2 * 9000/textAreaEstimation);

    // in basis with resizing from 300 to 700 the font size increases for
    // this factor
    const factor = 0.85;
    if(recalculated_width < 300){
        return size;
    } else if(recalculated_width <= 700){
        return size * (1 + ((recalculated_width - 300) / 400)) * factor;
    } else {
        return size * 2 * factor;
    }
}
