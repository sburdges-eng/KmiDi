// Bowling Scorer Module - Handles all bowling scoring logic
class BowlingScorer {
    constructor() {
        this.frames = [];
        this.currentFrame = 0;
        this.currentBall = 0;
        this.gameComplete = false;
        this.initializeFrames();
    }

    initializeFrames() {
        // Initialize 10 frames
        for (let i = 0; i < 10; i++) {
            this.frames.push({
                frameNumber: i + 1,
                balls: [],
                score: null,
                cumulative: null
            });
        }
    }

    reset() {
        this.frames = [];
        this.currentFrame = 0;
        this.currentBall = 0;
        this.gameComplete = false;
        this.initializeFrames();
    }

    // Record a ball thrown
    recordBall(pinsDown) {
        if (this.gameComplete) return false;

        const frame = this.frames[this.currentFrame];
        
        // Handle 10th frame specially
        if (this.currentFrame === 9) {
            return this.recordTenthFrameBall(pinsDown);
        }

        // Regular frames (1-9)
        frame.balls.push(pinsDown);

        // Strike - move to next frame
        if (pinsDown === 10) {
            this.currentFrame++;
            this.currentBall = 0;
        } 
        // Second ball thrown
        else if (frame.balls.length === 2) {
            this.currentFrame++;
            this.currentBall = 0;
        }
        // First ball thrown, not a strike
        else {
            this.currentBall = 1;
        }

        this.calculateScores();
        return true;
    }

    recordTenthFrameBall(pinsDown) {
        const frame = this.frames[9];
        frame.balls.push(pinsDown);

        // Determine if game is complete
        if (frame.balls.length === 1) {
            // First ball
            this.currentBall = 1;
        } else if (frame.balls.length === 2) {
            // Second ball
            if (frame.balls[0] === 10 || (frame.balls[0] + frame.balls[1] === 10)) {
                // Strike or spare - get third ball
                this.currentBall = 2;
            } else {
                // No strike or spare - game over
                this.gameComplete = true;
            }
        } else if (frame.balls.length === 3) {
            // Third ball - game over
            this.gameComplete = true;
        }

        this.calculateScores();
        return true;
    }

    // Calculate all frame scores
    calculateScores() {
        let cumulativeScore = 0;

        for (let i = 0; i < 10; i++) {
            const frame = this.frames[i];
            
            if (frame.balls.length === 0) {
                break; // No more balls thrown
            }

            let frameScore = null;

            if (i < 9) {
                // Frames 1-9
                if (this.isStrike(i)) {
                    // Strike scoring
                    const nextTwo = this.getNextTwoBalls(i);
                    if (nextTwo.length === 2) {
                        frameScore = 10 + nextTwo[0] + nextTwo[1];
                    }
                } else if (this.isSpare(i)) {
                    // Spare scoring
                    const nextOne = this.getNextBall(i);
                    if (nextOne !== null) {
                        frameScore = 10 + nextOne;
                    }
                } else if (frame.balls.length === 2) {
                    // Open frame
                    frameScore = frame.balls[0] + frame.balls[1];
                }
            } else {
                // 10th frame
                if (frame.balls.length >= 2) {
                    if (frame.balls[0] === 10 || (frame.balls[0] + frame.balls[1] === 10)) {
                        // Strike or spare in 10th - need all 3 balls
                        if (frame.balls.length === 3) {
                            frameScore = frame.balls.reduce((a, b) => a + b, 0);
                        }
                    } else {
                        // Open frame in 10th
                        frameScore = frame.balls[0] + frame.balls[1];
                    }
                }
            }

            if (frameScore !== null) {
                frame.score = frameScore;
                cumulativeScore += frameScore;
                frame.cumulative = cumulativeScore;
            }
        }
    }

    isStrike(frameIndex) {
        return this.frames[frameIndex].balls[0] === 10;
    }

    isSpare(frameIndex) {
        const frame = this.frames[frameIndex];
        return frame.balls.length === 2 && 
               frame.balls[0] + frame.balls[1] === 10;
    }

    getNextBall(frameIndex) {
        if (frameIndex === 9) return null;
        
        const nextFrame = this.frames[frameIndex + 1];
        if (nextFrame.balls.length > 0) {
            return nextFrame.balls[0];
        }
        return null;
    }

    getNextTwoBalls(frameIndex) {
        const balls = [];
        
        if (frameIndex === 9) return balls;
        
        const nextFrame = this.frames[frameIndex + 1];
        
        if (nextFrame.balls.length >= 2) {
            // Next frame has at least 2 balls
            balls.push(nextFrame.balls[0]);
            balls.push(nextFrame.balls[1]);
        } else if (nextFrame.balls.length === 1 && this.isStrike(frameIndex + 1)) {
            // Next frame is a strike, need to look ahead one more
            balls.push(10);
            if (frameIndex + 2 < 10) {
                const frameAfterNext = this.frames[frameIndex + 2];
                if (frameAfterNext.balls.length > 0) {
                    balls.push(frameAfterNext.balls[0]);
                }
            }
        }
        
        return balls;
    }

    getTotalScore() {
        let total = 0;
        for (const frame of this.frames) {
            if (frame.cumulative !== null) {
                total = frame.cumulative;
            }
        }
        return total;
    }

    getCurrentFrameNumber() {
        return Math.min(this.currentFrame + 1, 10);
    }

    getCurrentBallNumber() {
        return this.currentBall + 1;
    }

    getRemainingPins() {
        if (this.gameComplete) return 0;
        
        // 10th frame special handling
        if (this.currentFrame === 9) {
            const frame = this.frames[9];
            if (frame.balls.length === 0) return 10;
            if (frame.balls.length === 1) {
                if (frame.balls[0] === 10) return 10; // Strike, reset pins
                return 10 - frame.balls[0];
            }
            if (frame.balls.length === 2) {
                if (frame.balls[0] === 10) {
                    if (frame.balls[1] === 10) return 10; // Two strikes
                    return 10 - frame.balls[1];
                }
                if (frame.balls[0] + frame.balls[1] === 10) return 10; // Spare, reset pins
                return 0; // Game should be over
            }
            return 0;
        }
        
        // Regular frames
        if (this.currentBall === 0) {
            return 10;
        } else {
            const frame = this.frames[this.currentFrame];
            return 10 - frame.balls[0];
        }
    }

    getFrameDisplay(frameIndex) {
        const frame = this.frames[frameIndex];
        const display = {
            balls: [],
            total: frame.cumulative || ''
        };

        if (frameIndex < 9) {
            // Regular frames
            if (frame.balls.length === 0) {
                display.balls = ['', ''];
            } else if (this.isStrike(frameIndex)) {
                display.balls = ['', 'X'];
            } else if (this.isSpare(frameIndex)) {
                display.balls = [frame.balls[0].toString(), '/'];
            } else if (frame.balls.length === 1) {
                display.balls = [frame.balls[0].toString(), ''];
            } else {
                display.balls = [frame.balls[0].toString(), frame.balls[1].toString()];
            }
        } else {
            // 10th frame
            for (let i = 0; i < 3; i++) {
                if (i < frame.balls.length) {
                    const ball = frame.balls[i];
                    if (ball === 10) {
                        display.balls.push('X');
                    } else if (i === 1 && frame.balls[0] + ball === 10 && frame.balls[0] !== 10) {
                        display.balls.push('/');
                    } else if (i === 2 && frame.balls[1] !== 10 && frame.balls[1] + ball === 10) {
                        display.balls.push('/');
                    } else {
                        display.balls.push(ball.toString());
                    }
                } else {
                    display.balls.push('');
                }
            }
        }

        return display;
    }
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BowlingScorer;
}
