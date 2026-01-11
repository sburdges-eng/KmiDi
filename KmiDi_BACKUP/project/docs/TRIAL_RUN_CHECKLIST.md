# ✅ Dart Strike Trial Run Checklist

Use this checklist to verify everything works correctly during your trial runs.

---

## 🏁 Pre-Flight Checks

- [ ] Xcode project created successfully
- [ ] All `.swift` files added to project
- [ ] Project builds without errors (⌘B)
- [ ] App launches in simulator (⌘R)

---

## 🎮 Game Setup Testing

### Player Management
- [ ] Can add player with name "Sean"
- [ ] Can add second player "Player 2"
- [ ] Player names display in list
- [ ] Can remove players using trash icon
- [ ] Duplicate names show error message
- [ ] Empty names show error message
- [ ] Can add up to 8 players
- [ ] "Start Game" button activates when players added
- [ ] "Start Game" button disabled when no players

### Game Start
- [ ] Tapping "Start Game" transitions to game view
- [ ] First player's name displays correctly
- [ ] "Frame: 1" shows
- [ ] "Throw: 1" shows
- [ ] All 10 pins show in bowling formation
- [ ] All pins start in "standing" state (blue/white)

---

## 🎯 Pin Interaction Testing

### Basic Pin Tapping
- [ ] Tapping pin changes color (blue → red)
- [ ] Tapping again toggles back (red → blue)
- [ ] "Pins Knocked" counter updates correctly
- [ ] Can select multiple pins
- [ ] Can deselect multiple pins
- [ ] Counter shows "X/10" format

### Pin Reset Testing (CRITICAL)

#### Test 1: Normal Frame Reset
1. [ ] Select 5 pins, submit throw 1
2. [ ] Verify: Knocked pins stay red
3. [ ] Verify: Standing pins stay blue
4. [ ] Select 3 more pins, submit throw 2
5. [ ] **✅ VERIFY: All pins reset to blue (standing)**
6. [ ] Verify: Next player's turn

#### Test 2: Strike Reset
1. [ ] Select all 10 pins
2. [ ] Submit throw
3. [ ] **✅ VERIFY: All pins immediately reset**
4. [ ] Verify: Moved to next frame
5. [ ] Verify: "Throw: 1" for new frame

#### Test 3: Player Turn Reset
1. [ ] Complete Player 1's frame (2 throws)
2. [ ] **✅ VERIFY: Pins reset when switching to Player 2**
3. [ ] Verify: Player 2's name shows
4. [ ] Verify: "Frame: 1" for Player 2

#### Test 4: Manual Reset
1. [ ] Knock down several pins
2. [ ] Press "Reset Pins" button
3. [ ] **✅ VERIFY: All pins return to standing**
4. [ ] Verify: Counter resets to 0/10

---

## 📊 Scoring Testing

### Strike Scoring
1. [ ] Knock down all 10 pins on first throw
2. [ ] Verify: "X" appears in scorecard
3. [ ] Play next 2 throws
4. [ ] Verify: Strike frame score = 10 + next 2 throws

### Spare Scoring
1. [ ] Knock down 7 pins on first throw
2. [ ] Knock down 3 pins on second throw (total 10)
3. [ ] Verify: "/" appears in scorecard
4. [ ] Play next throw
5. [ ] Verify: Spare frame score = 10 + next throw

### Open Frame Scoring
1. [ ] Knock down 7 pins on first throw
2. [ ] Knock down 2 pins on second throw (total 9)
3. [ ] Verify: Shows "7" and "2" in scorecard
4. [ ] Verify: Frame score = 9

### Perfect Game (Optional Advanced Test)
1. [ ] Bowl 12 consecutive strikes
2. [ ] Verify: Final score = 300

---

## 🎳 10th Frame Testing

### 10th Frame with Strike
1. [ ] Reach frame 10
2. [ ] Bowl a strike
3. [ ] **✅ VERIFY: Pins reset for bonus throws**
4. [ ] Verify: Can bowl 2 more times
5. [ ] Verify: All 3 throws count

### 10th Frame with Spare
1. [ ] Reach frame 10
2. [ ] Bowl spare (e.g., 7 then 3)
3. [ ] **✅ VERIFY: Pins reset for bonus throw**
4. [ ] Verify: Can bowl 1 more time
5. [ ] Verify: All 3 throws count

### 10th Frame Open
1. [ ] Reach frame 10
2. [ ] Bowl open frame (e.g., 7 then 2)
3. [ ] Verify: Game ends (no bonus throw)
4. [ ] Verify: Only 2 throws counted

---

## 📱 Scorecard Testing

### Display
- [ ] Tap "Scorecard" button
- [ ] Verify: All players show
- [ ] Verify: All frames display (1-10)
- [ ] Verify: Frame 10 has 3 boxes
- [ ] Verify: Cumulative scores show
- [ ] Verify: Current frame highlighted
- [ ] Verify: Strikes show "X"
- [ ] Verify: Spares show "/"
- [ ] Verify: Gutter balls show "-"

### Navigation
- [ ] Can close scorecard with "Done"
- [ ] Returns to game view
- [ ] Game state preserved

---

## 💾 Persistence Testing

### Auto-Save
1. [ ] Start a game and play a few frames
2. [ ] Press Home button (⌘⇧H in simulator)
3. [ ] App goes to background
4. [ ] Verify: Console shows "✅ Game saved successfully"

### Resume Game
1. [ ] Kill app completely (swipe up in app switcher)
2. [ ] Relaunch app
3. [ ] Verify: "Resume Game?" alert appears
4. [ ] Verify: Shows last played date/time
5. [ ] Tap "Resume"
6. [ ] **✅ VERIFY: Game state restored perfectly**
7. [ ] Verify: Same players, scores, and frame

### New Game (with saved game)
1. [ ] Launch app with saved game
2. [ ] Tap "New Game" in alert
3. [ ] Verify: Returns to player setup
4. [ ] Verify: No players listed
5. [ ] Verify: Can start fresh game

---

## 🏁 Game Completion Testing

### Game End
1. [ ] Play through all 10 frames for all players
2. [ ] Verify: "Game Complete!" alert shows
3. [ ] Verify: Shows winner's name and score
4. [ ] Verify: Can view scorecard
5. [ ] Verify: Can start new game

### Winner Detection
- [ ] Highest score player declared winner
- [ ] Tie games handled correctly
- [ ] Final scores accurate

---

## 🐛 Bug Testing

### Edge Cases
- [ ] Test with 1 player (should work)
- [ ] Test with 8 players (maximum)
- [ ] Test rapid pin tapping (no crashes)
- [ ] Test submitting with 0 pins (should work)
- [ ] Test all strikes game (300 points)
- [ ] Test all gutter balls game (0 points)

### Memory/Performance
- [ ] No lag when tapping pins
- [ ] Smooth scrolling in scorecard
- [ ] No memory warnings in console
- [ ] App responds quickly

---

## 📱 Device Testing (Optional)

If testing on real iPhone:

- [ ] Deploy to physical device
- [ ] Trust developer certificate
- [ ] App installs successfully
- [ ] Touch interactions feel natural
- [ ] Screen fits properly
- [ ] Rotation works (if enabled)

---

## ✅ Final Sign-Off

### Core Features
- [ ] Pin reset works perfectly ✨
- [ ] Scoring calculates correctly
- [ ] Multi-player works smoothly
- [ ] Game saves and resumes
- [ ] Scorecard displays accurately

### User Experience
- [ ] Interface is clean and intuitive
- [ ] No confusing behaviors
- [ ] Buttons respond properly
- [ ] Text is readable
- [ ] Layout looks professional

### Performance
- [ ] No crashes during testing
- [ ] No freezing or lag
- [ ] Memory usage reasonable
- [ ] Battery drain acceptable

---

## 📝 Issues Found

Use this space to note any problems:

**Issue 1:**
- Problem:
- Steps to reproduce:
- Expected behavior:
- Actual behavior:

**Issue 2:**
- Problem:
- Steps to reproduce:
- Expected behavior:
- Actual behavior:

---

## 🎉 Trial Run Complete!

If all items are checked, your Dart Strike iOS app is **ready to go**!

**Pass Criteria**: ✅ 90%+ of items checked with no critical bugs

**Critical Items** (Must Pass):
- ✅ Pins reset after 2 throws
- ✅ Pins reset after strikes
- ✅ Pins reset when changing players
- ✅ Scoring calculates correctly
- ✅ Game saves and resumes

---

**Date Tested**: _______________
**Tester**: _______________
**Result**: ⭐⭐⭐⭐⭐

**Ready for launch**: YES ☐  NO ☐
