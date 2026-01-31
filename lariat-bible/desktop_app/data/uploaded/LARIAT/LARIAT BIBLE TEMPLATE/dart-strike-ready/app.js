// Dart Strike Main Application
class DartStrikeGame {
    constructor() {
        this.players = [];
        this.currentPlayerIndex = 0;
        this.selectedPins = new Set();
        this.gameStarted = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.addPlayer('Player 1');
        this.updateDisplay();
        this.logEvent('Welcome to Dart Strike! 🎯🎳');
    }

    setupEventListeners() {
        // Game control buttons
        document.getElementById('newGameBtn').addEventListener('click', () => this.newGame());
        document.getElementById('addPlayerBtn').addEventListener('click', () => this.showAddPlayerModal());
        document.getElementById('resetPinsBtn').addEventListener('click', () => this.resetPins());
        document.getElementById('submitThrowBtn').addEventListener('click', () => this.submitThrow());

        // Pin selection
        document.querySelectorAll('.pin').forEach(pin => {
            pin.addEventListener('click', (e) => this.togglePin(e.target));
        });

        // Modal controls
        document.getElementById('confirmAddPlayer').addEventListener('click', () => this.confirmAddPlayer());
        document.getElementById('cancelAddPlayer').addEventListener('click', () => this.hideAddPlayerModal());
        document.getElementById('newPlayerName').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.confirmAddPlayer();
        });

        // Close modal on outside click
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('addPlayerModal');
            if (e.target === modal) {
                this.hideAddPlayerModal();
            }
        });
    }

    newGame() {
        if (!confirm('Start a new game? This will reset all scores.')) return;
        
        this.gameStarted = true;
        this.currentPlayerIndex = 0;
        
        // Reset all players' scores
        this.players.forEach(player => {
            player.scorer.reset();
        });
        
        this.resetPins();
        this.updateDisplay();
        this.logEvent('🎮 New game started!');
    }

    addPlayer(name) {
        const player = {
            id: Date.now(),
            name: name,
            scorer: new BowlingScorer()
        };
        
        this.players.push(player);
        this.updatePlayersList();
        this.updateScorecard();
        this.logEvent(`👤 ${name} joined the game`);
        
        return player;
    }

    showAddPlayerModal() {
        const modal = document.getElementById('addPlayerModal');
        const input = document.getElementById('newPlayerName');
        modal.style.display = 'block';
        input.value = '';
        input.focus();
    }

    hideAddPlayerModal() {
        document.getElementById('addPlayerModal').style.display = 'none';
    }

    confirmAddPlayer() {
        const input = document.getElementById('newPlayerName');
        const name = input.value.trim();
        
        if (!name) {
            alert('Please enter a player name');
            return;
        }
        
        if (this.players.some(p => p.name.toLowerCase() === name.toLowerCase())) {
            alert('Player name already exists');
            return;
        }
        
        this.addPlayer(name);
        this.hideAddPlayerModal();
    }

    getCurrentPlayer() {
        return this.players[this.currentPlayerIndex];
    }

    togglePin(pinElement) {
        const pinNumber = parseInt(pinElement.dataset.pin);
        const currentPlayer = this.getCurrentPlayer();
        
        if (!currentPlayer || currentPlayer.scorer.gameComplete) return;
        
        // Check if pin is already knocked down
        if (pinElement.classList.contains('knocked')) {
            // Allow deselection
            pinElement.classList.remove('knocked');
            this.selectedPins.delete(pinNumber);
        } else {
            // Check remaining pins
            const remainingPins = currentPlayer.scorer.getRemainingPins();
            if (this.selectedPins.size >= remainingPins) {
                alert(`You can only knock down ${remainingPins} pin(s) on this throw`);
                return;
            }
            
            pinElement.classList.add('knocked');
            this.selectedPins.add(pinNumber);
        }
        
        this.updatePinsDownCount();
    }

    updatePinsDownCount() {
        document.getElementById('pinsDownCount').textContent = this.selectedPins.size;
    }

    resetPins() {
        // Clear all knocked pins
        document.querySelectorAll('.pin').forEach(pin => {
            pin.classList.remove('knocked');
        });
        this.selectedPins.clear();
        this.updatePinsDownCount();
        this.logEvent('🔄 Pins reset');
    }

    submitThrow() {
        const currentPlayer = this.getCurrentPlayer();
        if (!currentPlayer) return;
        
        const pinsDown = this.selectedPins.size;
        
        if (pinsDown === 0) {
            if (!confirm('Record a gutter ball (0 pins)?')) return;
        }
        
        // Record the throw
        currentPlayer.scorer.recordBall(pinsDown);
        
        // Log the throw
        const frameNum = currentPlayer.scorer.getCurrentFrameNumber();
        const ballNum = currentPlayer.scorer.getCurrentBallNumber() - 1; // -1 because we just recorded it
        
        let throwDescription = `${currentPlayer.name} - Frame ${frameNum}, Ball ${ballNum}: `;
        if (pinsDown === 10) {
            throwDescription += 'STRIKE! 🎳';
        } else if (ballNum === 2 && currentPlayer.scorer.isSpare(frameNum - 1)) {
            throwDescription += 'SPARE! 🎯';
        } else {
            throwDescription += `${pinsDown} pin${pinsDown !== 1 ? 's' : ''}`;
        }
        
        this.logEvent(throwDescription);
        
        // Check if we need to move to next player
        const shouldMoveToNextPlayer = this.shouldMoveToNextPlayer(currentPlayer);
        
        if (shouldMoveToNextPlayer) {
            this.moveToNextPlayer();
        } else {
            // Same player, next ball - reset pins if needed
            if (pinsDown === 10 || this.shouldResetPinsForSamePlayer(currentPlayer)) {
                this.resetPins();
            } else {
                // Keep knocked pins down for second ball
                this.selectedPins.clear();
                this.updatePinsDownCount();
            }
        }
        
        this.updateDisplay();
        this.checkGameComplete();
    }

    shouldMoveToNextPlayer(player) {
        const scorer = player.scorer;
        const frame = scorer.currentFrame;
        const ball = scorer.currentBall;
        
        // Game complete for this player
        if (scorer.gameComplete) return true;
        
        // In frames 1-9
        if (frame < 9 || (frame === 9 && ball === 0)) {
            // Just moved to next frame (after strike or second ball)
            return ball === 0;
        }
        
        // In 10th frame - only move when complete
        if (frame >= 10) {
            return scorer.gameComplete;
        }
        
        return false;
    }

    shouldResetPinsForSamePlayer(player) {
        const scorer = player.scorer;
        const frame = scorer.currentFrame;
        
        // In 10th frame, reset after strikes or spares
        if (frame === 9) {
            const balls = scorer.frames[9].balls;
            if (balls.length === 1 && balls[0] === 10) return true; // Strike
            if (balls.length === 2) {
                if (balls[1] === 10) return true; // Second ball strike
                if (balls[0] + balls[1] === 10) return true; // Spare
            }
        }
        
        return false;
    }

    moveToNextPlayer() {
        // Move to next player
        this.currentPlayerIndex = (this.currentPlayerIndex + 1) % this.players.length;
        
        // CRITICAL: Reset pins for the new player's turn
        this.resetPins();
        
        const newPlayer = this.getCurrentPlayer();
        this.logEvent(`🎮 ${newPlayer.name}'s turn`);
    }

    updateDisplay() {
        this.updateCurrentPlayerDisplay();
        this.updatePlayersList();
        this.updateScorecard();
        this.updatePinRack();
    }

    updateCurrentPlayerDisplay() {
        const player = this.getCurrentPlayer();
        if (!player) return;
        
        document.getElementById('currentPlayerName').textContent = player.name;
        document.getElementById('currentFrame').textContent = 
            Math.min(player.scorer.getCurrentFrameNumber(), 10);
        document.getElementById('currentBall').textContent = 
            player.scorer.getCurrentBallNumber();
    }

    updatePlayersList() {
        const container = document.getElementById('playersList');
        container.innerHTML = '';
        
        this.players.forEach((player, index) => {
            const tag = document.createElement('div');
            tag.className = 'player-tag';
            if (index === this.currentPlayerIndex) {
                tag.classList.add('active');
            }
            tag.textContent = `${player.name} (${player.scorer.getTotalScore()})`;
            container.appendChild(tag);
        });
    }

    updateScorecard() {
        const container = document.getElementById('scorecards');
        container.innerHTML = '';
        
        this.players.forEach(player => {
            const card = this.createScorecardElement(player);
            container.appendChild(card);
        });
    }

    createScorecardElement(player) {
        const card = document.createElement('div');
        card.className = 'scorecard';
        
        const title = document.createElement('h3');
        title.textContent = `${player.name} - Total: ${player.scorer.getTotalScore()}`;
        card.appendChild(title);
        
        const table = document.createElement('table');
        table.className = 'scorecard-table';
        
        // Header row
        const headerRow = document.createElement('tr');
        for (let i = 1; i <= 10; i++) {
            const th = document.createElement('th');
            th.textContent = `Frame ${i}`;
            headerRow.appendChild(th);
        }
        table.appendChild(headerRow);
        
        // Scores row
        const scoresRow = document.createElement('tr');
        for (let i = 0; i < 10; i++) {
            const td = document.createElement('td');
            const frameBox = document.createElement('div');
            frameBox.className = 'frame-box';
            
            const display = player.scorer.getFrameDisplay(i);
            
            // Frame scores
            const frameScores = document.createElement('div');
            frameScores.className = i === 9 ? 'frame-scores tenth-frame' : 'frame-scores';
            
            display.balls.forEach(ball => {
                const span = document.createElement('span');
                span.textContent = ball;
                frameScores.appendChild(span);
            });
            
            // Frame total
            const frameTotal = document.createElement('div');
            frameTotal.className = 'frame-total';
            frameTotal.textContent = display.total;
            
            frameBox.appendChild(frameScores);
            frameBox.appendChild(frameTotal);
            td.appendChild(frameBox);
            scoresRow.appendChild(td);
        }
        table.appendChild(scoresRow);
        
        card.appendChild(table);
        return card;
    }

    updatePinRack() {
        const player = this.getCurrentPlayer();
        if (!player) return;
        
        const remainingPins = player.scorer.getRemainingPins();
        const submitBtn = document.getElementById('submitThrowBtn');
        
        if (player.scorer.gameComplete) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Game Complete';
        } else {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Throw';
        }
        
        // Disable pins if no more can be knocked down
        document.querySelectorAll('.pin').forEach(pin => {
            if (!pin.classList.contains('knocked') && this.selectedPins.size >= remainingPins) {
                pin.style.opacity = '0.5';
                pin.style.cursor = 'not-allowed';
            } else {
                pin.style.opacity = '1';
                pin.style.cursor = 'pointer';
            }
        });
    }

    checkGameComplete() {
        const allComplete = this.players.every(p => p.scorer.gameComplete);
        
        if (allComplete && this.players.length > 0) {
            this.showGameResults();
        }
    }

    showGameResults() {
        // Find winner
        let winner = this.players[0];
        let maxScore = winner.scorer.getTotalScore();
        
        this.players.forEach(player => {
            const score = player.scorer.getTotalScore();
            if (score > maxScore) {
                maxScore = score;
                winner = player;
            }
        });
        
        const resultsDiv = document.createElement('div');
        resultsDiv.className = 'game-over';
        resultsDiv.innerHTML = `
            <h2>🎊 Game Complete! 🎊</h2>
            <p>Winner: ${winner.name} with ${maxScore} points!</p>
            <h3>Final Scores:</h3>
            ${this.players.map(p => 
                `<p>${p.name}: ${p.scorer.getTotalScore()}</p>`
            ).join('')}
        `;
        
        document.querySelector('.container').insertBefore(
            resultsDiv, 
            document.querySelector('.scorecard-section')
        );
        
        this.logEvent(`🏆 Game Over! ${winner.name} wins with ${maxScore} points!`);
    }

    logEvent(message) {
        const logContainer = document.getElementById('gameLog');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        
        const time = new Date().toLocaleTimeString();
        entry.textContent = `[${time}] ${message}`;
        
        logContainer.insertBefore(entry, logContainer.firstChild);
        
        // Keep only last 20 entries
        while (logContainer.children.length > 20) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }
}

// Initialize game when page loads
let game;
document.addEventListener('DOMContentLoaded', () => {
    game = new DartStrikeGame();
});
