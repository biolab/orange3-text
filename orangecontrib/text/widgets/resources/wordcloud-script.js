document.body.style.cursor = 'default';
document.body.style.webkitUserSelect = 'none';

// Options for WordCloud
var OPTIONS = {
    list: [],
    fontFamily: 'sans-serif',
    fontWeight: 'bold',
    color: 'UNUSED',
    backgroundColor: 'white',
    minSize: 6,
    weightFactor: 1,
    clearCanvas: true,
    minRotation: -Math.PI/10,
    maxRotation: Math.PI/10,
    rotateRatio: .5,
};

// Redraw wordcloud when the window size changes
function redrawWordCloud() {
    WordCloud(document.getElementById('canvas'), OPTIONS);
}
window.onresize = redrawWordCloud;

// Select words or clear selection
window.addEventListener('click', function(event) {
    var span = event.target;
    if (span.tagName == 'SPAN') {
        // Allow multiselection only if modifier key pressed
        if (! (event.ctrlKey || event.shiftKey)) {
            clearSelection();
        }
        cls = pybridge.word_clicked(span.innerHTML);
        console.log(cls);
        span.className = cls;
    } else if (span.tagName == 'BODY') {
        clearSelection();
    }
});
function clearSelection() {
    pybridge.word_clicked('');
    var spans = document.getElementsByTagName('span');
    for (var i=0; i < spans.length; ++i) {
        spans[i].className = '';
    }
}

// Mark words in SELECTED_WORDS list selected
var SELECTED_WORDS = [];
document.getElementById('canvas')
    .addEventListener('wordclouddrawn', selectWords);
function selectWords() {
    var lookup = {};
    var spans = document.getElementsByTagName('span');
    for (var i=0; i<SELECTED_WORDS.length; ++i) {
        lookup[SELECTED_WORDS[i]] = true;
    }
    for (var i=0; i<spans.length; ++i) {
        spans[i].className = lookup[spans[i].innerHTML] === true ? 'selected' : '';
    }
}
