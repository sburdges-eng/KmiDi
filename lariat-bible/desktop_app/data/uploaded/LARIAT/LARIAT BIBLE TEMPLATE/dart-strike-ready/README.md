# Dart Strike 🎯🎳

A hybrid darts and bowling scoring game that combines the precision of darts with the scoring system of bowling.

## Features

### ✅ Core Functionality
- **Traditional Bowling Scoring**: Full 10-frame bowling with strikes, spares, and 10th frame rules
- **Visual Pin Rack**: Interactive pin selection system
- **Multi-Player Support**: Add unlimited players to compete
- **Automatic Pin Reset**: Pins automatically reset between players' turns
- **Smart Pin Management**: Pins reset or stay based on bowling rules:
  - Reset after strikes
  - Reset between different players' turns
  - Stay knocked down for second ball (unless strike)
  - Reset after spares

### 🎮 Game Features
- **Live Scoring**: Real-time score calculation with cumulative totals
- **Game Log**: Track every throw and event
- **Visual Feedback**: Active player highlighting and animations
- **10th Frame Logic**: Proper handling of bonus balls
- **Winner Declaration**: Automatic game completion and winner announcement

## Installation & Running

### Option 1: Using Node.js
```bash
# Install dependencies (optional, for http-server)
npm install

# Start the server
npm start
```
Then open http://localhost:8080 in your browser

### Option 2: Using Python
```bash
# Start with Python's built-in server
python3 -m http.server 8080
```
Then open http://localhost:8080 in your browser

### Option 3: Direct File
Simply open `index.html` directly in a web browser

## How to Play

1. **Starting a Game**
   - Click "New Game" to begin
   - Add players using "Add Player" button
   - Game starts with Player 1

2. **Making Throws**
   - Click on pins to mark them as knocked down
   - Knocked pins turn red and rotate
   - Click "Submit Throw" to record your score
   - Pins automatically reset for the next player

3. **Scoring System**
   - **Strike (X)**: Knock down all 10 pins with first ball = 10 + next 2 balls
   - **Spare (/)**: Knock down all 10 pins with 2 balls = 10 + next 1 ball
   - **Open Frame**: Less than 10 pins = actual pin count
   - **10th Frame**: Get up to 3 balls if you score a strike or spare

4. **Pin Reset Rules**
   - **Automatic Reset Between Players**: When switching to a new player, pins always reset
   - **Strike Reset**: Pins reset after a strike for the same player
   - **Spare Reset**: Pins stay down for second ball, reset after spare
   - **10th Frame Reset**: Special rules apply for bonus balls
   - **Manual Reset**: Use "Reset Pins" button anytime

## Key Files

- `index.html` - Main game interface
- `app.js` - Game logic and player management
- `bowling-scorer.js` - Bowling scoring calculations
- `styles.css` - Visual styling and animations

## Technical Details

### Pin Reset Implementation
The game implements intelligent pin reset logic:

```javascript
// Automatic reset when switching players
moveToNextPlayer() {
    this.currentPlayerIndex = (this.currentPlayerIndex + 1) % this.players.length;
    this.resetPins(); // Always reset for new player
}

// Smart reset for same player
shouldResetPinsForSamePlayer(player) {
    // Reset after strikes or in 10th frame special cases
}
```

### Scoring Engine
- Modular `BowlingScorer` class handles all scoring logic
- Proper lookahead for strikes and spares
- Accurate 10th frame handling with bonus balls

## Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers (responsive design)

## Future Enhancements
- Save/Load games
- Player statistics tracking
- Tournament mode
- Sound effects
- Dart board integration for actual dart throwing
- Online multiplayer

## Troubleshooting

**Pins not resetting?**
- Check browser console for errors
- Try manual "Reset Pins" button
- Refresh the page to restart

**Scoring seems off?**
- Remember bowling scoring is cumulative
- Strikes/spares aren't scored until bonus balls are thrown
- Check the game log for throw history

## License
MIT License - Feel free to modify and use!

## Credits
Created for The Lariat Restaurant by Sean
Combines the best of darts precision with bowling scoring excitement!
