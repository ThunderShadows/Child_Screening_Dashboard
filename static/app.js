/**
 * CECI Game Engine & Testing UI
 * Interactive cognitive games with telemetry capture for CECI scoring.
 */

// ── State ──
let state = {
    ageGroup: null,
    currentSession: 0,
    totalSessions: 3,
    sessions: [],          // Collected session data for prediction
    gameQueue: [],          // Games to play in current session
    currentGameIndex: 0,
    // Trial-level telemetry
    trials: [],
    trialStart: 0,
    currentTrial: 0,
    totalTrials: 0,
    correct: 0,
    streak: 0,
    hesitations: 0,
};

// ── Game Definitions (mirror the backend) ──
const GAMES = {
    "0-2": [
        { id: "tap_the_animal", name: "Tap the Animal", domain: "Auditory Recognition", trials: 10,
          desc: "Listen to the sound and tap the correct animal!" },
        { id: "peek_a_boo_memory", name: "Peek-a-Boo Memory", domain: "Working Memory", trials: 8,
          desc: "Remember where the toy is hidden!" },
        { id: "shape_sorter", name: "Shape Sorter", domain: "Visual Spatial", trials: 10,
          desc: "Tap the matching shape!" },
    ],
    "3-5": [
        { id: "color_pattern_match", name: "Color Pattern", domain: "Pattern Recognition", trials: 12,
          desc: "What comes next in the pattern?" },
        { id: "memory_card_flip", name: "Memory Cards", domain: "Working Memory", trials: 12,
          desc: "Find the matching pairs!" },
        { id: "counting_garden", name: "Counting Garden", domain: "Numerical Cognition", trials: 12,
          desc: "Count the items and pick the right number!" },
        { id: "story_sequence", name: "Story Sequence", domain: "Sequential Reasoning", trials: 10,
          desc: "Put the pictures in the right order!" },
    ],
    "6-9": [
        { id: "digit_span_recall", name: "Digit Span", domain: "Working Memory", trials: 12,
          desc: "Remember and repeat the number sequence!" },
        { id: "word_category_sort", name: "Word Sort", domain: "Verbal Reasoning", trials: 15,
          desc: "Sort the word into the correct category!" },
        { id: "spatial_puzzle", name: "Spatial Puzzle", domain: "Visual Spatial", trials: 12,
          desc: "Which piece completes the pattern?" },
        { id: "math_race", name: "Math Race", domain: "Numerical Cognition", trials: 15,
          desc: "Solve the math problem!" },
        { id: "go_nogo", name: "Go / No-Go", domain: "Inhibitory Control", trials: 20,
          desc: "Tap GREEN shapes, ignore RED shapes!" },
    ],
};

// ── Helpers ──
const $ = (id) => document.getElementById(id);
const shuffle = (arr) => arr.sort(() => Math.random() - 0.5);
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

// ── Screen Navigation ──
function showScreen(name) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    $('screen-' + name).classList.add('active');
}

// ── Age Group Selection ──
function selectAgeGroup(ag) {
    state.ageGroup = ag;
    state.currentSession = 0;
    state.sessions = [];
    state.totalSessions = 3;

    const labels = { "0-2": "Age 0-2", "3-5": "Age 3-5", "6-9": "Age 6-9" };
    $('breadcrumb-age').textContent = labels[ag];
    $('total-sessions').textContent = state.totalSessions;

    startNextSession();
}

// ── Session Management ──
function startNextSession() {
    state.currentSession++;
    $('session-num').textContent = state.currentSession;

    // Pick 2 random games for this session
    const allGames = GAMES[state.ageGroup];
    state.gameQueue = shuffle([...allGames]).slice(0, 2);
    state.currentGameIndex = 0;

    startNextGame();
}

function startNextGame() {
    if (state.currentGameIndex >= state.gameQueue.length) {
        // All games in session done - compute session features
        endSession();
        return;
    }

    const game = state.gameQueue[state.currentGameIndex];
    state.trials = [];
    state.currentTrial = 0;
    state.totalTrials = game.trials;
    state.correct = 0;
    state.streak = 0;
    state.hesitations = 0;

    $('game-title').textContent = game.name;
    $('breadcrumb-game').textContent = game.name;
    $('game-domain').textContent = game.domain;
    $('trial-num').textContent = '1';
    $('total-trials').textContent = game.trials;
    $('trial-progress').style.width = '0%';
    $('score-display').style.display = 'flex';
    updateLiveScore();

    showScreen('game');

    // Countdown then start game
    showCountdown(() => {
        launchGame(game);
    });
}

function showCountdown(callback) {
    const canvas = $('game-canvas');
    let count = 3;
    canvas.innerHTML = `<div class="countdown">${count}</div>`;
    const interval = setInterval(() => {
        count--;
        if (count > 0) {
            canvas.innerHTML = `<div class="countdown">${count}</div>`;
        } else {
            clearInterval(interval);
            callback();
        }
    }, 700);
}

// ── Record Trial ──
function recordTrial(correct, reactionTime, hesitated = false) {
    state.trials.push({ correct, rt: reactionTime, hesitated });
    state.currentTrial++;
    if (correct) {
        state.correct++;
        state.streak++;
    } else {
        state.streak = 0;
    }
    if (hesitated) state.hesitations++;

    $('trial-num').textContent = Math.min(state.currentTrial + 1, state.totalTrials);
    $('trial-progress').style.width = `${(state.currentTrial / state.totalTrials) * 100}%`;
    updateLiveScore();
}

function updateLiveScore() {
    const total = state.trials.length || 1;
    const acc = ((state.correct / total) * 100).toFixed(0);
    const avgRt = state.trials.length > 0
        ? (state.trials.reduce((s, t) => s + t.rt, 0) / total).toFixed(1)
        : '0.0';
    $('live-accuracy').textContent = acc + '%';
    $('live-rt').textContent = avgRt + 's';
    $('live-streak').textContent = state.streak;
}

// ── End Session ──
function endSession() {
    // Compute session features from all trials across games this session
    const allTrials = state.trials;
    const n = allTrials.length;
    const correctCount = allTrials.filter(t => t.correct).length;
    const accuracy = n > 0 ? correctCount / n : 0;
    const meanRt = n > 0 ? allTrials.reduce((s, t) => s + t.rt, 0) / n : 0;
    const hesRatio = n > 0 ? allTrials.filter(t => t.hesitated).length / n : 0;
    const completionRate = n > 0 ? correctCount / n : 0;

    // Error burst rate
    let errorBursts = 0, runLen = 0;
    allTrials.forEach(t => {
        if (!t.correct) { runLen++; }
        else { if (runLen >= 3) errorBursts++; runLen = 0; }
    });
    if (runLen >= 3) errorBursts++;
    const errorBurstRate = n > 0 ? errorBursts / Math.max(1, Math.floor(n / 3)) : 0;

    // Engagement score
    const rtValues = allTrials.map(t => t.rt);
    const rtMean = meanRt;
    const rtStd = Math.sqrt(rtValues.reduce((s, r) => s + (r - rtMean) ** 2, 0) / Math.max(1, n));
    const rtCv = rtMean > 0 ? rtStd / rtMean : 0;
    const engagement = Math.max(0, Math.min(1, 1.0 - rtCv * 0.5 - hesRatio * 0.3));

    const sessionData = {
        accuracy: parseFloat(accuracy.toFixed(4)),
        mean_reaction_time: parseFloat(meanRt.toFixed(4)),
        hesitation_ratio: parseFloat(hesRatio.toFixed(4)),
        task_completion_rate: parseFloat(completionRate.toFixed(4)),
        error_burst_rate: parseFloat(errorBurstRate.toFixed(4)),
        engagement_score: parseFloat(engagement.toFixed(4)),
    };
    state.sessions.push(sessionData);

    // Show session summary
    $('session-end-title').textContent = `Session ${state.currentSession} Complete!`;
    $('end-accuracy').textContent = (accuracy * 100).toFixed(0) + '%';
    $('end-rt').textContent = meanRt.toFixed(2) + 's';
    $('end-engagement').textContent = (engagement * 100).toFixed(0) + '%';

    if (state.currentSession >= state.totalSessions) {
        $('btn-next-session').style.display = 'none';
        $('btn-view-results').style.display = 'inline-flex';
    } else {
        $('btn-next-session').style.display = 'inline-flex';
        $('btn-view-results').style.display = 'none';
    }

    showScreen('session-end');
}

// ── Predict ──
async function finishAndPredict() {
    showScreen('results');
    $('ceci-score-display').textContent = '...';
    $('ceci-score-display').classList.add('loading');

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sessions: state.sessions,
                child_id: 1,
                age_group: state.ageGroup,
            }),
        });
        const data = await res.json();
        displayResults(data);
    } catch (err) {
        $('ceci-score-display').textContent = 'Error';
        $('clinical-note').textContent = 'Failed to get prediction: ' + err.message;
    }
}

async function runSimulation() {
    const ag = $('sim-age').value;
    const profile = $('sim-profile').value;
    const nSessions = parseInt($('sim-sessions').value);

    showScreen('results');
    $('ceci-score-display').textContent = '...';
    $('ceci-score-display').classList.add('loading');

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ age_group: ag, profile, n_sessions: nSessions }),
        });
        const data = await res.json();
        displayResults(data);
    } catch (err) {
        $('ceci-score-display').textContent = 'Error';
        $('clinical-note').textContent = 'Failed to simulate: ' + err.message;
    }
}

// ── Display Results ──
function displayResults(data) {
    const score = data.ceci_score;
    const band = data.risk_band;

    // Score display
    $('ceci-score-display').classList.remove('loading');
    $('ceci-score-display').textContent = score.toFixed(2);

    // Color the score
    const colors = { green: '#22c55e', amber: '#f59e0b', red: '#ef4444' };
    $('ceci-score-display').style.color = colors[band] || '#3b82f6';
    $('ceci-score-display').style.background = 'none';
    $('ceci-score-display').style.webkitTextFillColor = 'unset';

    // Gauge glow
    const glows = { green: 'var(--shadow-glow-green)', amber: 'var(--shadow-glow-amber)', red: 'var(--shadow-glow-red)' };
    $('ceci-gauge').style.boxShadow = glows[band] || '';

    // Risk badge
    const labels = { green: 'Low Risk', amber: 'Moderate Risk - Monitor', red: 'High Risk - Refer' };
    const icons = { green: '&#9989;', amber: '&#9888;&#65039;', red: '&#128680;' };
    $('risk-badge-container').innerHTML =
        `<div class="risk-badge ${band}">${icons[band]} ${data.risk_label || labels[band]}</div>`;

    // Component bars
    const comps = data.components;
    let barsHTML = '';
    if (comps) {
        const items = [
            { label: 'PID (Cognitive)', value: comps.pid.value, color: '#ef4444', interp: comps.pid.interpretation },
            { label: 'Consistency', value: comps.consistency.value, color: '#3b82f6', interp: comps.consistency.interpretation },
            { label: 'PEff (Effort)', value: comps.effort.value, color: '#f59e0b', interp: comps.effort.interpretation },
        ];
        items.forEach(it => {
            barsHTML += `
                <div class="component-bar">
                    <div class="comp-label">${it.label}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${it.value * 100}%; background:${it.color}"></div>
                    </div>
                    <div class="comp-value" style="color:${it.color}">${(it.value * 100).toFixed(1)}%</div>
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted); margin-left:134px; margin-top:-10px;">${it.interp}</div>
            `;
        });
        // Confidence
        if (data.confidence !== undefined) {
            barsHTML += `
                <div class="component-bar" style="margin-top:8px;">
                    <div class="comp-label">Confidence</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${data.confidence * 100}%; background:#06b6d4"></div>
                    </div>
                    <div class="comp-value" style="color:#06b6d4">${(data.confidence * 100).toFixed(1)}%</div>
                </div>
            `;
        }
    }
    $('component-bars').innerHTML = barsHTML;

    // Clinical note
    $('clinical-note').textContent = data.clinical_note || '--';

    // Session history
    const sessions = data.raw_sessions || data.sessions || state.sessions;
    let rowsHTML = '';
    sessions.forEach((s, i) => {
        const acc = s.accuracy !== undefined ? s.accuracy : '--';
        const rt = s.mean_reaction_time !== undefined ? s.mean_reaction_time : '--';
        const hes = s.hesitation_ratio !== undefined ? s.hesitation_ratio : '--';
        const comp = s.task_completion_rate !== undefined ? s.task_completion_rate : '--';
        const eng = s.engagement_score !== undefined ? s.engagement_score : '--';
        rowsHTML += `<tr>
            <td>${i + 1}</td>
            <td>${typeof acc === 'number' ? (acc * 100).toFixed(1) + '%' : acc}</td>
            <td>${typeof rt === 'number' ? rt.toFixed(2) + 's' : rt}</td>
            <td>${typeof hes === 'number' ? (hes * 100).toFixed(1) + '%' : hes}</td>
            <td>${typeof comp === 'number' ? (comp * 100).toFixed(1) + '%' : comp}</td>
            <td>${typeof eng === 'number' ? (eng * 100).toFixed(1) + '%' : eng}</td>
        </tr>`;
    });
    $('session-history-body').innerHTML = rowsHTML;
}

function resetAll() {
    state = {
        ageGroup: null, currentSession: 0, totalSessions: 3, sessions: [],
        gameQueue: [], currentGameIndex: 0, trials: [], trialStart: 0,
        currentTrial: 0, totalTrials: 0, correct: 0, streak: 0, hesitations: 0,
    };
    showScreen('home');
}

// ═══════════════════════════════════════════════════════════
//  GAME IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════

function launchGame(game) {
    const launchers = {
        tap_the_animal: gameAnimalTap,
        peek_a_boo_memory: gamePeekMemory,
        shape_sorter: gameShapeSorter,
        color_pattern_match: gameColorPattern,
        memory_card_flip: gameMemoryCards,
        counting_garden: gameCounting,
        story_sequence: gameStorySequence,
        digit_span_recall: gameDigitSpan,
        word_category_sort: gameWordSort,
        spatial_puzzle: gameSpatialPuzzle,
        math_race: gameMathRace,
        go_nogo: gameGoNoGo,
    };

    const launcher = launchers[game.id];
    if (launcher) {
        launcher(game);
    } else {
        // Fallback generic game
        gameGenericChoice(game);
    }
}

// ── Helper: Next trial or next game ──
function advanceOrFinish(game) {
    if (state.currentTrial >= game.trials) {
        state.currentGameIndex++;
        setTimeout(() => startNextGame(), 600);
    }
}

// ------ GAME: Tap the Animal ------
function gameAnimalTap(game) {
    const animals = [
        { emoji: '&#128049;', name: 'Cat' },
        { emoji: '&#128054;', name: 'Dog' },
        { emoji: '&#128038;', name: 'Bird' },
        { emoji: '&#128046;', name: 'Cow' },
        { emoji: '&#128055;', name: 'Pig' },
        { emoji: '&#128056;', name: 'Frog' },
    ];

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const target = pick(animals);
        const options = shuffle([target, ...shuffle(animals.filter(a => a.name !== target.name)).slice(0, 2)]);
        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">Tap the <strong>${target.name}</strong>!</div>
            <div class="game-grid cols-3">
                ${options.map((a, i) => `<button class="game-btn" data-name="${a.name}" onclick="handleAnimalTap(this, '${target.name}')">${a.emoji}</button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleAnimalTap = (btn, targetName) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        const isCorrect = btn.dataset.name === targetName;
        const hesitated = rt > 5;
        btn.classList.add(isCorrect ? 'correct' : 'incorrect');
        recordTrial(isCorrect, rt, hesitated);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Peek-a-Boo Memory ------
function gamePeekMemory(game) {
    const toys = ['&#127912;', '&#9917;', '&#128663;', '&#128059;', '&#11088;'];

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const numCups = 3;
        const hiddenIndex = randInt(0, numCups - 1);
        const toy = pick(toys);
        const canvas = $('game-canvas');

        // Show phase
        canvas.innerHTML = `
            <div class="game-instruction">Watch where the toy goes!</div>
            <div class="game-grid cols-3">
                ${Array.from({ length: numCups }, (_, i) =>
                    `<button class="game-btn" style="font-size:2.5rem;">${i === hiddenIndex ? toy : ''}</button>`
                ).join('')}
            </div>
        `;

        // Hide phase
        setTimeout(() => {
            canvas.innerHTML = `
                <div class="game-instruction">Where was the ${toy} hidden? Tap the cup!</div>
                <div class="game-grid cols-3">
                    ${Array.from({ length: numCups }, (_, i) =>
                        `<button class="game-btn hidden-card" onclick="handlePeek(${i}, ${hiddenIndex})"></button>`
                    ).join('')}
                </div>
            `;
            state.trialStart = performance.now();
        }, 1500);
    }

    window.handlePeek = (clicked, correct) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        const isCorrect = clicked === correct;
        recordTrial(isCorrect, rt, rt > 6);
        // Show result briefly
        const btns = $('game-canvas').querySelectorAll('.game-btn');
        btns[clicked].classList.add(isCorrect ? 'correct' : 'incorrect');
        setTimeout(showTrial, 600);
    };

    showTrial();
}

// ------ GAME: Shape Sorter ------
function gameShapeSorter(game) {
    const shapes = [
        { emoji: '&#128308;', name: 'Circle' },
        { emoji: '&#128999;', name: 'Square' },
        { emoji: '&#128310;', name: 'Triangle' },
        { emoji: '&#128311;', name: 'Diamond' },
        { emoji: '&#11088;', name: 'Star' },
    ];

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const target = pick(shapes);
        const options = shuffle([target, ...shuffle(shapes.filter(s => s.name !== target.name)).slice(0, 2)]);
        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">Find the <strong>${target.name}</strong>!</div>
            <div class="stimulus">${target.emoji}</div>
            <div class="game-grid cols-3">
                ${options.map(s => `<button class="game-btn" onclick="handleShape(this, '${s.name}', '${target.name}')">${s.emoji}</button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleShape = (btn, clicked, target) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === target ? 'correct' : 'incorrect');
        recordTrial(clicked === target, rt, rt > 6);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Color Pattern Match ------
function gameColorPattern(game) {
    const colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899'];
    const colorNames = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Pink'];

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        // Create a pattern of 4 colors, ask what next (repeating AB pattern)
        const a = randInt(0, colors.length - 1);
        let b = randInt(0, colors.length - 1);
        while (b === a) b = randInt(0, colors.length - 1);
        const pattern = [a, b, a, b]; // ABAB
        const answer = a; // Next is A
        const wrongOptions = shuffle(
            Array.from({ length: colors.length }, (_, i) => i).filter(i => i !== answer)
        ).slice(0, 2);
        const options = shuffle([answer, ...wrongOptions]);

        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">What color comes next?</div>
            <div style="display:flex; gap:12px; margin:16px 0;">
                ${pattern.map(c => `<div style="width:50px; height:50px; border-radius:10px; background:${colors[c]};"></div>`).join('')}
                <div style="width:50px; height:50px; border-radius:10px; border:2px dashed rgba(255,255,255,0.3); display:flex; align-items:center; justify-content:center; font-size:1.5rem; color:var(--text-muted);">?</div>
            </div>
            <div class="game-grid cols-3">
                ${options.map(c => `<button class="game-btn" style="background:${colors[c]}; min-height:60px;" onclick="handlePattern(this, ${c}, ${answer})"></button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handlePattern = (btn, clicked, answer) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === answer ? 'correct' : 'incorrect');
        recordTrial(clicked === answer, rt, rt > 5);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Memory Card Flip ------
function gameMemoryCards(game) {
    const emojis = ['&#127803;', '&#127819;', '&#128142;', '&#9924;', '&#127752;', '&#128171;'];
    let board = [];
    let flipped = [];
    let matched = 0;
    let pairsNeeded = 4;

    function initBoard() {
        const selected = shuffle([...emojis]).slice(0, pairsNeeded);
        board = shuffle([...selected, ...selected]);
        flipped = [];
        matched = 0;
    }

    function renderBoard() {
        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">Find matching pairs! Tap two cards.</div>
            <div class="game-grid cols-4">
                ${board.map((emoji, i) => {
                    const isFlipped = flipped.includes(i);
                    const isMatched = board[i] === null;
                    if (isMatched) return `<button class="game-btn correct" style="opacity:0.5" disabled>${emoji}</button>`;
                    if (isFlipped) return `<button class="game-btn flipped">${emoji}</button>`;
                    return `<button class="game-btn hidden-card" onclick="handleCardFlip(${i})"></button>`;
                }).join('')}
            </div>
        `;
    }

    window.handleCardFlip = (index) => {
        if (flipped.length >= 2 || flipped.includes(index)) return;
        if (!state.trialStart) state.trialStart = performance.now();

        flipped.push(index);
        renderBoard();

        if (flipped.length === 2) {
            const rt = (performance.now() - state.trialStart) / 1000;
            const [a, b] = flipped;
            const isMatch = board[a] === board[b];

            setTimeout(() => {
                if (isMatch) {
                    board[a] = null;
                    board[b] = null;
                    matched++;
                    recordTrial(true, rt, rt > 5);
                } else {
                    recordTrial(false, rt, rt > 5);
                }
                flipped = [];
                state.trialStart = performance.now();

                if (matched >= pairsNeeded || state.currentTrial >= game.trials) {
                    advanceOrFinish(game);
                } else {
                    renderBoard();
                }
            }, 700);
        }
    };

    initBoard();
    state.trialStart = performance.now();
    renderBoard();
}

// ------ GAME: Counting Garden ------
function gameCounting(game) {
    const items = ['&#127800;', '&#127801;', '&#127803;', '&#127799;', '&#127807;'];

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const count = randInt(2, 8);
        const item = pick(items);
        const wrongA = Math.max(1, count + randInt(-2, -1));
        const wrongB = count + randInt(1, 2);
        const options = shuffle([count, wrongA, wrongB]);

        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">How many ${item} are there?</div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:16px 0;">
                ${Array(count).fill(`<span style="font-size:2rem;">${item}</span>`).join('')}
            </div>
            <div class="game-grid cols-3">
                ${options.map(n => `<button class="game-btn" style="font-size:1.8rem; font-weight:700;" onclick="handleCount(this, ${n}, ${count})">${n}</button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleCount = (btn, clicked, answer) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === answer ? 'correct' : 'incorrect');
        recordTrial(clicked === answer, rt, rt > 5);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Story Sequence ------
function gameStorySequence(game) {
    const stories = [
        { cards: ['&#127748;', '&#127783;&#65039;', '&#127752;'], label: 'Sun, Rain, Rainbow' },
        { cards: ['&#127834;', '&#127858;', '&#128513;'], label: 'Cook, Eat, Happy' },
        { cards: ['&#128716;', '&#9748;', '&#127876;'], label: 'Sleep, Wake, Play' },
        { cards: ['&#128218;', '&#9997;&#65039;', '&#11088;'], label: 'Read, Write, Star' },
    ];

    let seqIndex = 0;
    function showTrial() {
        if (state.currentTrial >= game.trials || seqIndex >= stories.length) {
            advanceOrFinish(game); return;
        }
        const story = stories[seqIndex % stories.length];
        const shuffled = shuffle(story.cards.map((c, i) => ({ emoji: c, order: i })));

        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">Put these in order: ${story.label}</div>
            <div class="game-grid cols-3">
                ${shuffled.map(c => `<button class="game-btn" data-order="${c.order}" onclick="handleSeq(this)">${c.emoji}</button>`).join('')}
            </div>
            <div style="margin-top:16px; font-size:0.85rem; color:var(--text-muted);">Tap in the correct order (1st, 2nd, 3rd)</div>
        `;
        state.trialStart = performance.now();
        window._seqClicks = [];
        window._seqExpected = 0;
    }

    window.handleSeq = (btn) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        const order = parseInt(btn.dataset.order);
        const isCorrect = order === window._seqExpected;
        btn.classList.add(isCorrect ? 'correct' : 'incorrect');
        btn.style.pointerEvents = 'none';
        recordTrial(isCorrect, rt, rt > 6);

        if (isCorrect) window._seqExpected++;

        if (window._seqExpected >= 3 || !isCorrect) {
            seqIndex++;
            setTimeout(showTrial, 600);
        }
    };

    showTrial();
}

// ------ GAME: Digit Span Recall ------
function gameDigitSpan(game) {
    let spanLength = 3;

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const digits = Array.from({ length: spanLength }, () => randInt(1, 9));
        const canvas = $('game-canvas');

        // Show digits one by one
        let showIndex = 0;
        canvas.innerHTML = `<div class="game-instruction">Remember these numbers!</div><div class="stimulus-text" id="digit-display">Ready...</div>`;

        const interval = setInterval(() => {
            if (showIndex < digits.length) {
                $('digit-display').textContent = digits[showIndex];
                showIndex++;
            } else {
                clearInterval(interval);
                // Input phase
                canvas.innerHTML = `
                    <div class="game-instruction">Type the numbers in order!</div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap; justify-content:center;">
                        ${Array.from({ length: 9 }, (_, i) =>
                            `<button class="game-btn" style="min-width:60px; min-height:60px; font-size:1.4rem; font-weight:700;" onclick="handleDigit(${i + 1})">${i + 1}</button>`
                        ).join('')}
                    </div>
                    <div style="margin-top:12px; font-size:1.2rem; font-weight:600;" id="digit-input"></div>
                `;
                state.trialStart = performance.now();
                window._digitTarget = digits;
                window._digitInput = [];
            }
        }, 800);
    }

    window.handleDigit = (num) => {
        window._digitInput.push(num);
        $('digit-input').textContent = window._digitInput.join(' ');

        if (window._digitInput.length >= window._digitTarget.length) {
            const rt = (performance.now() - state.trialStart) / 1000;
            const isCorrect = window._digitInput.every((d, i) => d === window._digitTarget[i]);
            recordTrial(isCorrect, rt, rt > 8);
            if (isCorrect && spanLength < 7) spanLength++;
            setTimeout(showTrial, 600);
        }
    };

    showTrial();
}

// ------ GAME: Word Category Sort ------
function gameWordSort(game) {
    const categories = [
        { name: 'Animals', words: ['Cat', 'Dog', 'Fish', 'Bird', 'Horse'] },
        { name: 'Food', words: ['Apple', 'Cake', 'Rice', 'Bread', 'Milk'] },
        { name: 'Colors', words: ['Red', 'Blue', 'Green', 'Yellow', 'Pink'] },
    ];

    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const correctCat = pick(categories);
        const word = pick(correctCat.words);
        const options = shuffle(categories.map(c => c.name));

        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">Which category does this word belong to?</div>
            <div class="stimulus-text">${word}</div>
            <div class="game-grid cols-3">
                ${options.map(cat => `<button class="game-btn" style="font-size:1rem; font-weight:600;" onclick="handleWordSort(this, '${cat}', '${correctCat.name}')">${cat}</button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleWordSort = (btn, clicked, correct) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === correct ? 'correct' : 'incorrect');
        recordTrial(clicked === correct, rt, rt > 5);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Spatial Puzzle ------
function gameSpatialPuzzle(game) {
    const patterns = [
        { grid: ['#', '.', '#', '.', '#', '.', '#', '.', '?'], answer: '#' },
        { grid: ['.', '#', '.', '#', '.', '#', '.', '#', '?'], answer: '.' },
        { grid: ['#', '#', '.', '.', '#', '#', '.', '.', '?'], answer: '#' },
    ];

    let pIdx = 0;
    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const pattern = patterns[pIdx % patterns.length];
        pIdx++;
        const options = shuffle([pattern.answer, pattern.answer === '#' ? '.' : '#']);

        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">What fills the missing spot?</div>
            <div class="game-grid cols-3" style="max-width:200px; margin:16px auto;">
                ${pattern.grid.map(cell => {
                    if (cell === '?') return `<div class="game-btn" style="border-style:dashed; background:none; cursor:default;">?</div>`;
                    const bg = cell === '#' ? 'var(--accent-blue)' : 'transparent';
                    return `<div class="game-btn" style="background:${bg}; cursor:default;"></div>`;
                }).join('')}
            </div>
            <div class="game-grid cols-2" style="max-width:200px; margin:0 auto;">
                ${options.map(opt => {
                    const bg = opt === '#' ? 'var(--accent-blue)' : 'transparent';
                    return `<button class="game-btn" style="background:${bg};" onclick="handleSpatial(this, '${opt}', '${pattern.answer}')"></button>`;
                }).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleSpatial = (btn, clicked, answer) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === answer ? 'correct' : 'incorrect');
        recordTrial(clicked === answer, rt, rt > 6);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Math Race ------
function gameMathRace(game) {
    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const ops = ['+', '-'];
        const op = pick(ops);
        let a, b, answer;
        if (op === '+') {
            a = randInt(1, 15);
            b = randInt(1, 15);
            answer = a + b;
        } else {
            a = randInt(5, 20);
            b = randInt(1, a);
            answer = a - b;
        }

        const wrong1 = answer + randInt(1, 3);
        const wrong2 = Math.max(0, answer - randInt(1, 3));
        const options = shuffle([answer, wrong1, wrong2]);

        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">Solve it fast!</div>
            <div class="stimulus-text" style="font-size:2.5rem;">${a} ${op} ${b} = ?</div>
            <div class="game-grid cols-3">
                ${options.map(n => `<button class="game-btn" style="font-size:1.5rem; font-weight:700;" onclick="handleMath(this, ${n}, ${answer})">${n}</button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleMath = (btn, clicked, answer) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === answer ? 'correct' : 'incorrect');
        recordTrial(clicked === answer, rt, rt > 5);
        setTimeout(showTrial, 500);
    };

    showTrial();
}

// ------ GAME: Go / No-Go ------
function gameGoNoGo(game) {
    const goShapes = ['&#9679;', '&#9733;', '&#9650;'];
    const nogoShapes = ['&#10006;', '&#9632;', '&#9830;'];
    let goNoGoTimeout = null;

    function showTrial() {
        if (state.currentTrial >= game.trials) {
            if (goNoGoTimeout) clearTimeout(goNoGoTimeout);
            advanceOrFinish(game);
            return;
        }

        const isGo = Math.random() > 0.35; // ~65% go trials
        const shape = isGo ? pick(goShapes) : pick(nogoShapes);
        const canvas = $('game-canvas');

        canvas.innerHTML = `
            <div class="game-instruction">${isGo ? 'TAP! (Green)' : 'HOLD BACK! (Red)'}</div>
            <div class="go-nogo-stimulus ${isGo ? 'go' : 'nogo'}" onclick="handleGoNogo(true, ${isGo})">${shape}</div>
            <div style="margin-top:16px; font-size:0.8rem; color:var(--text-muted);">
                Tap GREEN shapes &bull; Ignore RED shapes
            </div>
        `;
        state.trialStart = performance.now();
        window._gonogoHandled = false;

        // Auto-advance after 2s for no-go trials
        goNoGoTimeout = setTimeout(() => {
            if (!window._gonogoHandled) {
                window._gonogoHandled = true;
                const isCorrect = !isGo; // Correct to NOT tap on no-go
                recordTrial(isCorrect, 2.0, false);
                showTrial();
            }
        }, 2000);
    }

    window.handleGoNogo = (tapped, isGo) => {
        if (window._gonogoHandled) return;
        window._gonogoHandled = true;
        if (goNoGoTimeout) clearTimeout(goNoGoTimeout);
        const rt = (performance.now() - state.trialStart) / 1000;
        const isCorrect = isGo; // Correct to tap on go, wrong to tap on no-go
        recordTrial(isCorrect, rt, rt > 2);
        setTimeout(showTrial, 400);
    };

    showTrial();
}

// ------ Fallback Generic ------
function gameGenericChoice(game) {
    function showTrial() {
        if (state.currentTrial >= game.trials) { advanceOrFinish(game); return; }
        const correct = randInt(1, 4);
        const canvas = $('game-canvas');
        canvas.innerHTML = `
            <div class="game-instruction">${game.desc || 'Choose the correct answer!'}</div>
            <div class="game-grid cols-2">
                ${[1, 2, 3, 4].map(n => `<button class="game-btn" onclick="handleGeneric(this, ${n}, ${correct})">${n === correct ? '&#11088;' : '&#128309;'}</button>`).join('')}
            </div>
        `;
        state.trialStart = performance.now();
    }

    window.handleGeneric = (btn, clicked, answer) => {
        const rt = (performance.now() - state.trialStart) / 1000;
        btn.classList.add(clicked === answer ? 'correct' : 'incorrect');
        recordTrial(clicked === answer, rt, rt > 5);
        setTimeout(showTrial, 500);
    };

    showTrial();
}
