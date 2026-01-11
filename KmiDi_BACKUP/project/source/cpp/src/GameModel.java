package com.dartstrike.models;

import javafx.beans.property.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

import java.util.ArrayList;
import java.util.List;

/**
 * Game Model
 * Core game logic with traditional bowling scoring and pin reset
 * ✅ PIN RESET BUG FIXED
 */
public class GameModel {
    private final ObservableList<Player> players;
    private final IntegerProperty currentPlayerIndex;
    private final ObservableList<Pin> pins;
    private final BooleanProperty gameStarted;
    private final BooleanProperty gameComplete;
    private final List<GameStateListener> listeners;
    
    private static final int MAX_PINS = 10;
    
    public GameModel() {
        this.players = FXCollections.observableArrayList();
        this.currentPlayerIndex = new SimpleIntegerProperty(0);
        this.pins = FXCollections.observableArrayList();
        this.gameStarted = new SimpleBooleanProperty(false);
        this.gameComplete = new SimpleBooleanProperty(false);
        this.listeners = new ArrayList<>();
        
        resetPins();
    }
    
    // ===== GAME SETUP =====
    
    public void startNewGame(List<String> playerNames) {
        players.clear();
        for (String name : playerNames) {
            players.add(new Player(name));
        }
        currentPlayerIndex.set(0);
        resetPins();
        gameStarted.set(true);
        gameComplete.set(false);
        notifyListeners();
    }
    
    public void addPlayer(String name) {
        players.add(new Player(name));
    }
    
    // ===== PIN MANAGEMENT =====
    
    public void resetPins() {
        pins.clear();
        for (int i = 1; i <= MAX_PINS; i++) {
            pins.add(new Pin(i));
        }
        notifyListeners();
    }
    
    public void togglePin(int pinId) {
        for (Pin pin : pins) {
            if (pin.getId() == pinId) {
                pin.toggle();
                notifyListeners();
                break;
            }
        }
    }
    
    public int getPinsKnockedDown() {
        return (int) pins.stream().filter(pin -> !pin.isStanding()).count();
    }
    
    // ===== THROW SUBMISSION =====
    
    public void submitThrow() {
        if (currentPlayerIndex.get() >= players.size()) return;
        
        int pinsDown = getPinsKnockedDown();
        Player player = players.get(currentPlayerIndex.get());
        int frameIndex = player.getCurrentFrameIndex();
        
        if (frameIndex >= 10) {
            // Game complete for this player
            checkGameComplete();
            return;
        }
        
        Frame frame = player.getFrames().get(frameIndex);
        
        // Handle 10th frame special rules
        if (frameIndex == 9) {
            handle10thFrame(player, frame, pinsDown);
        } else {
            handleRegularFrame(player, frame, pinsDown);
        }
        
        calculateScores();
        notifyListeners();
    }
    
    private void handleRegularFrame(Player player, Frame frame, int pinsDown) {
        int frameIndex = player.getCurrentFrameIndex();
        
        if (player.getCurrentThrow() == 1) {
            frame.setFirstThrow(pinsDown);
            
            if (pinsDown == 10) {
                // Strike - move to next frame
                moveToNextFrame(player);
                resetPins();  // ✅ RESET AFTER STRIKE
            } else {
                // Not a strike - continue to second throw
                player.setCurrentThrow(2);
            }
        } else {
            // Second throw
            frame.setSecondThrow(pinsDown);
            moveToNextFrame(player);
            resetPins();  // ✅ RESET AFTER FRAME COMPLETE
        }
    }
    
    private void handle10thFrame(Player player, Frame frame, int pinsDown) {
        if (player.getCurrentThrow() == 1) {
            frame.setFirstThrow(pinsDown);
            player.setCurrentThrow(2);
            
            if (pinsDown == 10) {
                resetPins();  // Strike - reset for bonus throws
            }
        } else if (player.getCurrentThrow() == 2) {
            frame.setSecondThrow(pinsDown);
            
            // Check if player gets a third throw
            Integer firstThrow = frame.getFirstThrow();
            Integer secondThrow = frame.getSecondThrow();
            
            if (firstThrow == 10 || (firstThrow + secondThrow == 10)) {
                // Strike or spare - get bonus throw
                player.setCurrentThrow(3);
                resetPins();
            } else {
                // No bonus throw - move to next player
                moveToNextFrame(player);
                resetPins();
            }
        } else {
            // Third throw (bonus)
            frame.setThirdThrow(pinsDown);
            moveToNextFrame(player);
            resetPins();
        }
    }
    
    private void moveToNextFrame(Player player) {
        player.setCurrentFrameIndex(player.getCurrentFrameIndex() + 1);
        player.setCurrentThrow(1);
        
        // Move to next player
        moveToNextPlayer();
    }
    
    private void moveToNextPlayer() {
        currentPlayerIndex.set(currentPlayerIndex.get() + 1);
        
        if (currentPlayerIndex.get() >= players.size()) {
            // All players finished this round
            currentPlayerIndex.set(0);
            
            // Check if all players are done
            if (players.stream().allSatisfy(Player::isGameComplete)) {
                gameComplete.set(true);
            }
        }
        
        // Skip players who are done
        while (currentPlayerIndex.get() < players.size() && 
               players.get(currentPlayerIndex.get()).isGameComplete()) {
            currentPlayerIndex.set(currentPlayerIndex.get() + 1);
            if (currentPlayerIndex.get() >= players.size()) {
                currentPlayerIndex.set(0);
            }
        }
        
        resetPins();  // ✅ RESET WHEN SWITCHING PLAYERS
    }
    
    private void checkGameComplete() {
        if (players.stream().allSatisfy(Player::isGameComplete)) {
            gameComplete.set(true);
        }
    }
    
    // ===== SCORE CALCULATION =====
    
    public void calculateScores() {
        for (Player player : players) {
            int cumulativeScore = 0;
            
            for (int frameIndex = 0; frameIndex < 10; frameIndex++) {
                Frame frame = player.getFrames().get(frameIndex);
                
                if (frameIndex == 9) {
                    // 10th frame scoring
                    Integer first = frame.getFirstThrow();
                    Integer second = frame.getSecondThrow();
                    Integer third = frame.getThirdThrow();
                    
                    int frameScore = (first != null ? first : 0) +
                                   (second != null ? second : 0) +
                                   (third != null ? third : 0);
                    cumulativeScore += frameScore;
                    frame.setScore(cumulativeScore);
                } else {
                    // Regular frame scoring
                    Integer frameScore = calculateFrameScore(player, frameIndex);
                    if (frameScore != null) {
                        cumulativeScore += frameScore;
                        frame.setScore(cumulativeScore);
                    } else {
                        frame.setScore(null);
                    }
                }
            }
        }
    }
    
    private Integer calculateFrameScore(Player player, int frameIndex) {
        Frame frame = player.getFrames().get(frameIndex);
        Integer firstThrow = frame.getFirstThrow();
        
        if (firstThrow == null) return null;
        
        // Strike
        if (firstThrow == 10) {
            int nextFrame = frameIndex + 1;
            if (nextFrame >= 10) return null;
            
            Frame next = player.getFrames().get(nextFrame);
            Integer nextFirst = next.getFirstThrow();
            if (nextFirst == null) return null;
            
            if (nextFirst == 10) {
                // Next frame is also a strike
                int nextNextFrame = frameIndex + 2;
                if (nextNextFrame < 10) {
                    Integer nextNextFirst = player.getFrames().get(nextNextFrame).getFirstThrow();
                    if (nextNextFirst == null) return null;
                    return 10 + nextFirst + nextNextFirst;
                } else {
                    // Next frame is 10th frame
                    Integer nextSecond = next.getSecondThrow();
                    if (nextSecond == null) return null;
                    return 10 + nextFirst + nextSecond;
                }
            } else {
                // Next frame is not a strike
                Integer nextSecond = next.getSecondThrow();
                if (nextSecond == null) return null;
                return 10 + nextFirst + nextSecond;
            }
        }
        
        // Spare
        Integer secondThrow = frame.getSecondThrow();
        if (secondThrow == null) return null;
        
        if (firstThrow + secondThrow == 10) {
            int nextFrame = frameIndex + 1;
            if (nextFrame >= 10) return null;
            
            Frame next = player.getFrames().get(nextFrame);
            Integer nextFirst = next.getFirstThrow();
            if (nextFirst == null) return null;
            return 10 + nextFirst;
        }
        
        // Open frame
        return firstThrow + secondThrow;
    }
    
    // ===== CURRENT PLAYER HELPER =====
    
    public Player getCurrentPlayer() {
        if (currentPlayerIndex.get() < players.size()) {
            return players.get(currentPlayerIndex.get());
        }
        return null;
    }
    
    // ===== GAME STATE =====
    
    public void resetGame() {
        players.clear();
        currentPlayerIndex.set(0);
        resetPins();
        gameStarted.set(false);
        gameComplete.set(false);
        notifyListeners();
    }
    
    // ===== LISTENERS =====
    
    public interface GameStateListener {
        void onGameStateChanged();
    }
    
    public void addListener(GameStateListener listener) {
        listeners.add(listener);
    }
    
    public void removeListener(GameStateListener listener) {
        listeners.remove(listener);
    }
    
    private void notifyListeners() {
        for (GameStateListener listener : listeners) {
            listener.onGameStateChanged();
        }
    }
    
    // ===== PROPERTIES =====
    
    public ObservableList<Player> getPlayers() {
        return players;
    }
    
    public IntegerProperty currentPlayerIndexProperty() {
        return currentPlayerIndex;
    }
    
    public ObservableList<Pin> getPins() {
        return pins;
    }
    
    public BooleanProperty gameStartedProperty() {
        return gameStarted;
    }
    
    public BooleanProperty gameCompleteProperty() {
        return gameComplete;
    }
}
