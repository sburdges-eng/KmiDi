# Compression Deep Dive Guide

Understanding and using compression to control dynamics and add punch.

---

## What Compression Does

Compression reduces the volume of loud sounds, making the overall dynamic range smaller.

**Results:**
- Quiet parts become more audible
- Loud parts are controlled
- More consistent level
- Can add punch, glue, or character

---

## The Basic Parameters

### Threshold

**What it does:** Sets the level where compression starts.

- Signals below threshold: Unchanged
- Signals above threshold: Compressed

**Setting it:**
- Lower threshold = more compression
- Watch the gain reduction meter
- Start with threshold that catches peaks

### Ratio

**What it does:** How much compression is applied.

| Ratio | Meaning | Use For |
|-------|---------|---------|
| 2:1 | Gentle, 2dB over becomes 1dB | Subtle control |
| 3:1 | Moderate | Vocals, guitars |
| 4:1 | Standard | Most sources |
| 6:1 | Aggressive | Drums, punchy sounds |
| 10:1+ | Limiting | Peaks, protection |
| ∞:1 | Brick wall | Limiting |

### Attack

**What it does:** How fast compression kicks in.

| Attack | Effect | Use For |
|--------|--------|---------|
| Fast (0-10ms) | Catches transients, reduces punch | Control, smoothing |
| Medium (10-30ms) | Lets transients through, then controls | Vocals, balance |
| Slow (30-100ms+) | Transients pass, body compressed | Punch, impact |

**Key insight:** Slow attack = more punch (transients pass through).

### Release

**What it does:** How fast compression lets go.

| Release | Effect | Use For |
|---------|--------|---------|
| Fast (50-100ms) | Quick recovery, pumping possible | Drums, pumping effect |
| Medium (100-300ms) | Natural, transparent | Most sources |
| Slow (300ms+) | Smooth, sustained reduction | Gentle control |
| Auto | Adjusts to source | Convenience |

**Tip:** Release should recover before the next transient hits.

### Makeup Gain

**What it does:** Adds volume back after compression.

- Compression reduces level
- Makeup gain compensates
- Match bypassed level for fair comparison

### Knee

**What it does:** Controls how gradually compression kicks in around the threshold.

| Knee | Behavior | Use For |
|------|----------|---------|
| Hard (0dB) | Abrupt start at threshold | Aggressive, obvious compression |
| Soft (1-3dB) | Gradual transition | Natural, transparent compression |
| Medium (3-6dB) | Smooth curve | Most mixing tasks |

**Key insight:** Soft knee = more musical, less obvious compression.

---

## Gain Reduction Amounts

### How Much Is Right?

| Amount | Result | Use For |
|--------|--------|---------|
| 1-3 dB | Subtle control | Gentle shaping |
| 3-6 dB | Noticeable control | Most mixing tasks |
| 6-10 dB | Heavy compression | Drums, aggressive sound |
| 10+ dB | Extreme | Effects, pumping |

### The Sweet Spot

For most mixing:

- 3-6 dB gain reduction
- Compression working, but not obvious
- Sound is controlled, not squashed

---

## Compression Techniques

### Basic Compression (Control)

**Goal:** Even out dynamics.

**Settings:**
- Ratio: 3:1 to 4:1
- Attack: Medium (15-30ms)
- Release: Medium (100-200ms) or Auto
- Gain reduction: 3-6 dB on peaks

### Punchy Compression

**Goal:** Add impact and punch.

**Settings:**
- Ratio: 4:1 to 6:1
- Attack: SLOW (30-50ms) — lets transients through
- Release: Medium-fast (50-150ms)
- Gain reduction: 4-8 dB

**Key:** Slow attack preserves transients.

### Glue Compression (Bus)

**Goal:** Make elements sound cohesive.

**Settings:**
- Ratio: 2:1 to 4:1
- Attack: Medium-slow (10-30ms)
- Release: Auto or medium
- Gain reduction: 1-3 dB (gentle)

**Use on:** Drum bus, mix bus, groups.

### Parallel Compression

**Goal:** Add power without losing dynamics.

**Method:**
1. Original track stays uncompressed
2. Send to aux with heavy compression
3. Blend compressed signal underneath

**Aux Settings:**
- Ratio: 8:1 to 10:1
- Fast attack
- Medium release
- Smash it (10+ dB reduction)
- Blend to taste

**Result:** Dynamics preserved + power added.

### Serial Compression

**Goal:** Transparent control using multiple stages.

**Method:**
1. First compressor: Light (2-3 dB)
2. Second compressor: Light (2-3 dB)
3. Total: 4-6 dB (but more transparent)

**Why it works:** Each compressor works less hard = less artifacts.

### Sidechain Compression

**Goal:** Make one element duck when another plays (e.g., bass ducking when kick hits).

**Method:**
1. Insert compressor on target track (e.g., bass)
2. Set sidechain input to source track (e.g., kick)
3. Adjust threshold and ratio to taste
4. Fast attack (1-5ms) and release (50-150ms)

**Common Uses:**
- Bass ducking for kick drum
- Synth pads ducking for vocals
- Reverb ducking for dry signal
- Creating rhythmic pumping effects

**Settings:**
- Ratio: 4:1 to 8:1
- Attack: Fast (1-5ms)
- Release: Fast-medium (50-150ms) — match tempo
- Gain reduction: 3-6 dB on duck

**Tip:** Use sidechain for clarity, not just effects.

### Multiband Compression

**Goal:** Compress different frequency ranges independently.

**Method:**
1. Split signal into bands (low, mid, high)
2. Apply different compression to each band
3. Blend bands back together

**Common Uses:**
- Control low-end without affecting highs
- Tame harsh frequencies in vocals
- Balance frequency balance dynamically
- De-essing (high-frequency compression)

**Settings by Band:**
- **Low:** Ratio 4:1, slow attack (30-50ms), medium release
- **Mid:** Ratio 3:1, medium attack (10-30ms), medium release
- **High:** Ratio 2:1 to 4:1, fast attack (1-10ms), fast release

**Tip:** Use sparingly — can sound unnatural if overdone.

### De-essing

**Goal:** Reduce harsh "s" and "sh" sounds in vocals.

**Method:**
1. Use multiband compressor or dedicated de-esser
2. Target high frequencies (5-10 kHz)
3. Fast attack, fast release
4. Listen for sibilance reduction without dulling vocals

**Settings:**
- Frequency: 5-10 kHz (where sibilance lives)
- Ratio: 3:1 to 6:1
- Attack: Very fast (0.1-1ms)
- Release: Fast (10-50ms)
- Gain reduction: 3-6 dB on "s" sounds only

**Tip:** Solo the de-esser to hear what it's catching.

---

## Compression by Instrument

### Vocals

**Goal:** Consistent level, present but natural.

**Settings:**
- Ratio: 3:1 to 4:1
- Attack: Medium (10-30ms) — let consonants through
- Release: Medium (100-200ms) or Auto
- Gain reduction: 3-6 dB on peaks

**Tips:**
- Fast attack kills consonants
- Consider serial compression
- Automate volume first for extreme dynamics

### Drums (Kick)

**Goal:** Punch and control.

**Settings:**
- Ratio: 4:1 to 6:1
- Attack: Slow (30-50ms) — preserve transient
- Release: Fast (50-100ms) — recover before next hit
- Gain reduction: 4-8 dB

### Drums (Snare)

**Goal:** Snap and body.

**Settings:**
- Ratio: 4:1 to 6:1
- Attack: Medium-slow (20-40ms) — preserve crack
- Release: Medium (100-150ms)
- Gain reduction: 4-8 dB

### Drums (Bus)

**Goal:** Glue the kit together.

**Settings:**
- Ratio: 2:1 to 4:1
- Attack: Medium (10-30ms)
- Release: Auto or matched to tempo
- Gain reduction: 2-4 dB

**Tip:** Parallel compression on drum bus = power.

### Bass

**Goal:** Consistent low end.

**Settings:**
- Ratio: 4:1 to 6:1
- Attack: Medium-fast (10-20ms)
- Release: Medium (100-200ms)
- Gain reduction: 4-8 dB

**Tip:** Bass often needs more compression than you think.

### Acoustic Guitar

**Goal:** Even dynamics, present but natural.

**Settings:**
- Ratio: 3:1 to 4:1
- Attack: Medium (15-30ms)
- Release: Medium (150-250ms)
- Gain reduction: 3-6 dB

### Electric Guitar

**Goal:** Control peaks, add sustain.

**Settings:**
- Ratio: 3:1 to 4:1
- Attack: Medium (10-30ms)
- Release: Medium (100-200ms)
- Gain reduction: 3-6 dB

**Note:** Distorted guitars are already compressed by the amp.

### Piano/Keys

**Goal:** Even dynamics across range.

**Settings:**
- Ratio: 2:1 to 3:1 (gentle)
- Attack: Medium (20-40ms)
- Release: Medium-slow (200-400ms)
- Gain reduction: 2-4 dB

---

## Mix Bus Compression

### Why Use It

- Glues the mix together
- Adds cohesion
- Can add punch and energy

### Settings

- Ratio: 2:1 to 4:1
- Attack: Medium-slow (10-30ms)
- Release: Auto or matched to tempo
- Gain reduction: 1-3 dB (SUBTLE)

### Important

- Use from early in mixing
- Mix INTO the compressor
- Don't add at the end and crank it

---

## Compression Order in Signal Chain

**Where to place compression matters.**

### Typical Order (Top to Bottom)

1. **EQ (Pre-compression)** - Remove problem frequencies first
2. **Compression** - Control dynamics
3. **EQ (Post-compression)** - Shape tone after compression
4. **Effects** - Reverb, delay, modulation

### Why Order Matters

- **EQ before compression:** Removing mud before compression = cleaner compression
- **Compression before reverb:** Compressed signal into reverb = more consistent reverb tail
- **EQ after compression:** Compression can change frequency balance; post-EQ fixes it

### Exceptions

- **De-esser:** Usually after main compression (catches sibilance that compression brings up)
- **Parallel compression:** On separate aux, so order doesn't matter
- **Sidechain:** Compressor position matters less than sidechain routing

**Rule of thumb:** Fix problems first (EQ), then control dynamics (compression), then add character (effects).

---

## Reading Gain Reduction Meters

**Understanding what the meters tell you.**

### What to Watch For

| Meter Reading | Meaning | Action |
|---------------|---------|--------|
| 0 dB | No compression | Lower threshold or check routing |
| 1-3 dB | Light compression | Good for subtle control |
| 3-6 dB | Moderate compression | Sweet spot for most mixing |
| 6-10 dB | Heavy compression | Check if it sounds good |
| 10+ dB | Extreme compression | Usually too much unless intentional |
| Constant reduction | Always compressing | Raise threshold |
| No recovery | Release too slow | Speed up release |

### Visual Patterns

- **Peak reduction:** Compressor catching peaks (good)
- **Constant reduction:** Threshold too low (usually bad)
- **Pumping:** Release too fast (may be intentional)
- **No movement:** Compressor not working (check routing/threshold)

**Tip:** Watch meters while listening — meters confirm what your ears hear.

---

## Creative Compression Techniques

**Using compression as an effect, not just control.**

### Pumping Effect

**Goal:** Rhythmic pumping/breathing effect.

**Method:**
- Fast attack (1-5ms)
- Fast release (50-150ms) — match tempo
- High ratio (8:1+)
- Heavy gain reduction (8-12 dB)
- Sidechain to kick or snare

**Use for:** EDM, electronic music, creative effects.

### Crushing Compression

**Goal:** Extreme, lo-fi, character-heavy sound.

**Method:**
- Very fast attack (0.1-1ms)
- Fast release (10-50ms)
- High ratio (10:1+)
- Extreme gain reduction (15-20 dB)
- Often on parallel aux

**Use for:** Drums, creative effects, lo-fi aesthetics.

### Upward Compression

**Goal:** Bring up quiet parts without affecting loud parts.

**Method:**
- Use expander/gate in reverse, or specialized upward compressor
- Raises quiet signals instead of lowering loud ones
- Preserves transients

**Use for:** Bringing out room tone, ambience, quiet details.

### Compression as Distortion

**Goal:** Use compression artifacts as tone.

**Method:**
- Extreme settings (fast attack, high ratio)
- Drive into compression hard
- Use character compressors (FET, tube)
- Embrace the distortion

**Use for:** Aggressive sounds, creative tone shaping.

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Too much compression | Sounds squashed, lifeless | Reduce ratio or raise threshold |
| Attack too fast | Kills transients, no punch | Slow down attack |
| Release too fast | Pumping artifacts | Slow down release |
| Release too slow | Compression doesn't recover | Speed up release |
| Only compressing | Level still uneven | Use volume automation too |
| Not A/B comparing | Can't tell if it's better | Bypass and compare |
| Compressing everything | Mix loses dynamics | Use compression selectively |
| Ignoring makeup gain | Compressed signal too quiet | Add makeup gain to match level |
| Wrong compressor type | Doesn't fit the source | Match compressor character to source |
| Sidechain too obvious | Pumping is distracting | Reduce ratio or adjust release |

---

## Troubleshooting Compression Issues

### Problem: Compression Sounds Obvious/Unnatural

**Causes:**
- Too much gain reduction
- Attack too fast
- Release not matching source

**Solutions:**
- Reduce ratio or raise threshold
- Slow down attack to preserve transients
- Match release to source rhythm
- Try serial compression (lighter stages)
- Use softer knee

### Problem: No Punch/Impact

**Causes:**
- Attack too fast (killing transients)
- Too much compression overall
- Wrong compressor type

**Solutions:**
- Slow down attack (30-50ms for drums)
- Reduce gain reduction
- Try FET or VCA compressor (faster, punchier)
- Use parallel compression instead

### Problem: Pumping/Breathing Artifacts

**Causes:**
- Release too fast
- Too much gain reduction
- Release not matching tempo

**Solutions:**
- Slow down release
- Reduce gain reduction
- Match release to song tempo
- Use auto-release if available

### Problem: Compression Not Working

**Causes:**
- Threshold too high
- Wrong routing
- Bypass engaged
- Sidechain not configured

**Solutions:**
- Lower threshold until meter moves
- Check signal routing
- Verify compressor is active
- Check sidechain input if using sidechain

### Problem: Vocals Sound Dull

**Causes:**
- Attack too fast (killing consonants)
- Too much compression
- Wrong compressor type

**Solutions:**
- Slow attack to 15-30ms
- Reduce gain reduction
- Try opto compressor (smoother)
- Use serial compression

### Problem: Mix Sounds Small/Flat

**Causes:**
- Too much compression on mix bus
- Everything over-compressed
- No dynamics left

**Solutions:**
- Reduce mix bus compression (1-3 dB max)
- Re-evaluate individual track compression
- Use parallel compression for power
- Leave some dynamics intact

---

## Compressor Types (Character)

### VCA (Clean)

- Fast, precise
- Transparent
- Good for: Drums, mix bus
- Logic: Compressor (Platinum Digital)

### Opto (Smooth)

- Slow, smooth
- Natural sound
- Good for: Vocals, bass, guitars
- Logic: Compressor (Vintage Opto)

### FET (Aggressive)

- Fast, punchy
- Adds color/grit
- Good for: Drums, aggressive vocals
- Logic: Compressor (Vintage FET)

### Tube/Variable-Mu (Warm)

- Slow, warm
- Gentle, musical
- Good for: Mix bus, mastering
- Logic: Compressor (Vintage VCA)

---

## Logic Pro Compressor Modes

| Mode | Character | Use For |
|------|-----------|---------|
| Platinum Digital | Clean, precise | Surgical control |
| Vintage VCA | Punchy, snappy | Drums, bus |
| Vintage FET | Aggressive, fast | Drums, vocals |
| Vintage Opto | Smooth, slow | Vocals, bass |
| Studio VCA | Modern, clean | General purpose |
| Studio FET | Modern aggressive | Parallel compression |

---

## Compression by Genre

### Electronic/EDM

**Characteristics:** Heavy compression, pumping effects, sidechain ducking.

- **Kick:** Heavy compression (6:1, slow attack, fast release)
- **Bass:** Sidechain to kick, heavy compression
- **Synths:** Moderate compression, often sidechained
- **Mix Bus:** Moderate compression with pumping

### Hip-Hop

**Characteristics:** Punchy drums, controlled vocals, tight low end.

- **Kick:** Aggressive compression (6:1, slow attack)
- **Snare:** Heavy compression for snap
- **Vocals:** Serial compression, de-essing important
- **808/Bass:** Heavy compression, sidechain to kick

### Rock

**Characteristics:** Dynamic, punchy, natural-sounding compression.

- **Drums:** Moderate compression, preserve transients
- **Vocals:** Moderate compression (3:1-4:1)
- **Guitars:** Light compression (distortion already compresses)
- **Mix Bus:** Light glue compression

### Pop

**Characteristics:** Consistent levels, polished, controlled.

- **Vocals:** Serial compression, de-essing essential
- **Drums:** Moderate compression, parallel for power
- **Everything:** More compression than rock, less than EDM
- **Mix Bus:** Light compression for glue

### Jazz/Acoustic

**Characteristics:** Minimal compression, preserve dynamics.

- **Everything:** Light compression (2:1-3:1)
- **Vocals:** Gentle compression, preserve natural dynamics
- **Drums:** Light compression, preserve transients
- **Mix Bus:** Very light or none

---

## Quick Reference

### Starting Points

| Source | Ratio | Attack | Release | GR |
|--------|-------|--------|---------|-----|
| Vocals | 3:1 | 15-30ms | Auto | 3-6dB |
| Kick | 4:1 | 30-50ms | 50-100ms | 4-8dB |
| Snare | 4:1 | 20-40ms | 100-150ms | 4-8dB |
| Bass | 4:1 | 10-20ms | 100-200ms | 4-8dB |
| Acoustic | 3:1 | 15-30ms | 150-250ms | 3-6dB |
| Drum Bus | 3:1 | 10-30ms | Auto | 2-4dB |
| Mix Bus | 2:1 | 20-30ms | Auto | 1-3dB |

### Advanced Techniques Quick Reference

| Technique | Ratio | Attack | Release | GR | Use For |
|-----------|-------|--------|---------|-----|---------|
| Parallel | 8:1+ | Fast | Medium | 10+ dB | Power without losing dynamics |
| Sidechain | 4:1-8:1 | Fast (1-5ms) | Fast (50-150ms) | 3-6 dB | Ducking, clarity |
| Multiband Low | 4:1 | Slow (30-50ms) | Medium | 3-6 dB | Control low end |
| Multiband High | 2:1-4:1 | Fast (1-10ms) | Fast | 3-6 dB | De-essing, harshness |
| Pumping Effect | 8:1+ | Fast (1-5ms) | Fast (50-150ms) | 8-12 dB | Rhythmic effects |
| Crushing | 10:1+ | Very fast (0.1-1ms) | Fast (10-50ms) | 15-20 dB | Lo-fi, character |

---

## Related
- [[EQ Deep Dive Guide]]
- [[Mixing Workflow Checklist]]
- [[Vocal Production Guide]]

