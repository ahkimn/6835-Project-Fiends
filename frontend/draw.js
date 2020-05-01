var app = new Vue({
    el: '#draw',
    data: {
        currentPage: 0,
        history: [],
        pages: [],
        color: '#13c5f7',
        popups: {
            showColor: false,
            showSize: false,
            showWelcome: true,
            showSave: false,
            showOptions: false
        },
        options: {
            restrictY: false,
            restrictX: false
        },
        save: {
            name: '',
            saveItems: []
        },
        size: 12,
        colors: [
            '#d4f713',
            '#13f7ab',
            '#13f3f7',
            '#13c5f7',
            '#138cf7',
            '#1353f7',
            '#2d13f7',
            '#7513f7',
            '#a713f7',
            '#d413f7',
            '#f713e0',
            '#f71397',
            '#f7135b',
            '#f71313',
            '#f76213',
            '#f79413',
            '#f7e013'],
        sizes: [6, 12, 24, 48],
        weights: [ 2, 4, 6 ]
    },
    methods: {
        removeHistoryItem: ()=>{
            app.history.splice(app.history.length-2, 1);
            draw.redraw();
        },
        removeAllHistory: ()=>{
            app.history = [];
            draw.redraw();
        },
        saveItem: ()=>{
            if(app.save.name.length > 2){
                var historyItem = {
                    history: app.history.slice(),
                    name: app.save.name
                };

                app.save.saveItems.push(historyItem);
                app.save.name = "";
            }
        },
        loadSave: (item)=>{
            app.history = item.history.slice();
            draw.redraw();
        },
        addPage: () => {

            if (!app.pages.length || app.currentPage === app.pages.length - 1) {

                var pageItem = app.history.slice();
                if (!app.pages) {
                    app.pages.push(pageItem);
                } else {
                    app.pages[app.currentPage] = pageItem;
                }
                app.history = [];
                draw.redraw();
                app.pages.push(app.history.slice());
                app.currentPage += 1;
            }
        },
        previousPage: () => {
            if (app.currentPage > 0) {
                app.pages[app.currentPage] = app.history.slice();
                app.currentPage -= 1;
                app.history = app.pages[app.currentPage].slice();
                draw.redraw();
            }
        },
        nextPage: () => {
            if (app.currentPage < app.pages.length - 1) {
                app.pages[app.currentPage] = app.history.slice();
                app.currentPage += 1;
                app.history = app.pages[app.currentPage].slice();
                draw.redraw();
            }
        },
        test: () => {
            // Broken... tried to simulate mouse movement. Supposed to be a demo button.
            function wait(ms)
            {
                var d = new Date();
                var d2 = null;
                do { d2 = new Date(); }
                while(d2-d < ms);
            }

            var csv = 'data/sample.csv';
            d3.csv(csv).then(function(data) {
                console.log(data);
                for (var i = 0; i < data.length; i++) {
                    var coords = data[i];
                    jQuery(document.elementFromPoint(coords.x, coords.y)).mousedown();
                    jQuery(document.elementFromPoint(coords.x, coords.y)).mousemove();
                    jQuery(document.elementFromPoint(coords.x, coords.y)).mouseup();
                    wait(coords.t);
                }
            });
        }
    }
});

class Draw {
    constructor(){
        this.c = document.getElementById('canvas');
        this.ctx = this.c.getContext('2d');

        this.mouseDown = false;
        this.mouseX = 0;
        this.mouseY = 0;

        this.tempHistory = [];

        this.setSize();

        this.listen();

        this.redraw();
    }

    listen(){
        this.c.addEventListener('mousedown', (e)=>{
            this.mouseDown = true;
            this.mouseX = e.offsetX;
            this.mouseY = e.offsetY;
            this.setDummyPoint();
        });

        this.c.addEventListener('mouseup', ()=>{
            if(this.mouseDown){
                this.setDummyPoint();
            }
            this.mouseDown = false;
        });

        this.c.addEventListener('mouseleave', ()=>{
            if(this.mouseDown){
                this.setDummyPoint();
            }
            this.mouseDown = false;
        });

        this.c.addEventListener('mousemove', (e)=>{
            this.moveMouse(e);

            if(this.mouseDown){
                this.mouseX = this.mouseX;
                this.mouseY = this.mouseY;

                if(!app.options.restrictX){
                    this.mouseX = e.offsetX;
                }

                if(!app.options.restrictY){
                    this.mouseY = e.offsetY;
                }

                var item = {
                    isDummy: false,
                    x: this.mouseX,
                    y: this.mouseY,
                    c: app.color,
                    r: app.size
                };

                app.history.push(item);
                this.draw(item, app.history.length);
            }
        });

        window.addEventListener('resize', ()=>{
            this.setSize();
            this.redraw();
        });
    }

    setSize(){
        this.c.width = window.innerWidth;
        this.c.height = window.innerHeight - 60;
    }

    moveMouse(e){
        let x = e.offsetX;
        let y = e.offsetY;

        var cursor = document.getElementById('cursor');

        cursor.style.transform = `translate(${x - 10}px, ${y - 10}px)`;
    }

    getDummyItem(){
        var lastPoint = app.history[app.history.length-1];

        return {
            isDummy: true,
            x: lastPoint.x,
            y: lastPoint.y,
            c: null,
            r: null
        };
    }

    setDummyPoint(){
        var item = this.getDummyItem();
        app.history.push(item);
        this.draw(item, app.history.length);
    }

    redraw(){
        this.ctx.clearRect(0, 0, this.c.width, this.c.height);
        this.drawBgDots();

        if(!app.history.length){
            return true;
        }

        app.history.forEach((item, i)=>{
            this.draw(item, i);
        });
    }

    drawBgDots(){
        var gridSize = 50;
        this.ctx.fillStyle = 'rgba(0, 0, 0, .2)';

        for(var i = 0; i*gridSize < this.c.width; i++){
            for(var j = 0; j*gridSize < this.c.height; j++){
                if(i > 0 && j > 0){
                    this.ctx.beginPath();
                    this.ctx.rect(i * gridSize, j * gridSize, 2, 2);
                    this.ctx.fill();
                    this.ctx.closePath();
                }
            }
        }
    }

    draw(item, i){
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin="round";

        var prevItem = app.history[i-2];

        if(i < 2){
            return false;
        }

        if(!item.isDummy && !app.history[i-1].isDummy && !prevItem.isDummy){
            this.ctx.strokeStyle = item.c;
            this.ctx.lineWidth = item.r;

            this.ctx.beginPath();
            this.ctx.moveTo(prevItem.x, prevItem.y);
            this.ctx.lineTo(item.x, item.y);
            this.ctx.stroke();
            this.ctx.closePath();
        } else if (!item.isDummy) {
            this.ctx.strokeStyle = item.c;
            this.ctx.lineWidth = item.r;

            this.ctx.beginPath();
            this.ctx.moveTo(item.x, item.y);
            this.ctx.lineTo(item.x, item.y);
            this.ctx.stroke();
            this.ctx.closePath();
        }
    }
}

var draw = new Draw();

// Speech Recognition
var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
var SpeechGrammarList = SpeechGrammarList || webkitSpeechGrammarList;
var SpeechRecognitionEvent = SpeechRecognitionEvent || webkitSpeechRecognitionEvent;

var commands = [ 'add page', 'next page', 'previous page', 'clear page'];
var grammar = '#JSGF V1.0; grammar commands; public <command> = ' + commands.join(' | ') + ' ;';

var recognition = new SpeechRecognition();
var speechRecognitionList = new SpeechGrammarList();

speechRecognitionList.addFromString(grammar, 1);


recognition.grammars = speechRecognitionList;
recognition.continuous = true;
recognition.lang = 'en-US';
recognition.interimResults = false;
recognition.maxAlternatives = 1;

recognition.start();

recognition.onresult = function(event) {
    var current = event.resultIndex;

    var command = event.results[current][0].transcript.trim();
    switch (command) {
        case 'add page':
            app.addPage();
            break;
        case 'next page':
            app.nextPage();
            break;
        case 'previous page':
            app.previousPage();
            break;
        case 'clear page':
            app.removeAllHistory();
            break;
        default:
            console.log('invalid command');
    }
};
