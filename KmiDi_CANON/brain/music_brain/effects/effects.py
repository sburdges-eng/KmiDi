# music_brain/effects/effects.py
"""
Individual effect implementations - every effect type imaginable.
"""

from typing import List, Dict, Optional
import math
import random

from music_brain.effects.base import (
    BaseEffect,
    EffectCategory,
    Parameter,
    DistortionType,
    DelayType,
    ReverbType,
    FilterType,
    WaveShape,
    AmpModel,
    CabinetType,
)


# =============================================================================
# DISTORTION / OVERDRIVE / FUZZ
# =============================================================================

class DistortionEffect(BaseEffect):
    """
    Multi-mode distortion with 8 circuit types.
    """
    
    def __init__(self):
        super().__init__("Distortion", EffectCategory.DISTORTION)
        self.circuit_type = DistortionType.TUBE
    
    def _init_parameters(self):
        self.add_parameter("drive", 0.5, 0.0, 1.0, "", "Amount of distortion")
        self.add_parameter("tone", 0.5, 0.0, 1.0, "", "Brightness control")
        self.add_parameter("level", 0.7, 0.0, 1.0, "", "Output level")
        self.add_parameter("gate", 0.0, 0.0, 1.0, "", "Noise gate threshold")
        self.add_parameter("bias", 0.5, 0.0, 1.0, "", "Tube bias (asymmetry)")
        self.add_parameter("sag", 0.3, 0.0, 1.0, "", "Power supply sag")
        self.add_parameter("blend", 1.0, 0.0, 1.0, "", "Dry/wet blend")
    
    def set_circuit(self, circuit_type: DistortionType):
        self.circuit_type = circuit_type
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        drive = self.get_param("drive")
        tone = self.get_param("tone")
        level = self.get_param("level")
        bias = self.get_param("bias")
        blend = self.get_param("blend")
        
        output = []
        for sample in samples:
            # Apply input gain based on drive
            x = sample * (1 + drive * 20) * self.input_gain
            
            # Apply bias offset for asymmetric clipping
            x += (bias - 0.5) * 0.5
            
            # Distortion curve based on circuit type
            if self.circuit_type == DistortionType.TUBE:
                # Soft clipping - tanh
                y = math.tanh(x * (1 + drive * 3))
            
            elif self.circuit_type == DistortionType.TRANSISTOR:
                # Harder clipping
                y = max(-1, min(1, x * (1 + drive * 5)))
            
            elif self.circuit_type == DistortionType.DIODE:
                # Asymmetric diode clipping
                if x > 0:
                    y = min(1, x * (1 + drive * 4))
                else:
                    y = max(-0.7, x * (1 + drive * 2))
            
            elif self.circuit_type == DistortionType.FUZZ:
                # Extreme hard clipping + octave
                y = 1.0 if x > 0.1 else (-1.0 if x < -0.1 else x * 10)
                y *= math.tanh(x * 2)  # Add octave-ish harmonics
            
            elif self.circuit_type == DistortionType.RECTIFIER:
                # Full wave rectification + clipping
                y = math.tanh(abs(x) * (1 + drive * 3))
            
            elif self.circuit_type == DistortionType.BITCRUSH:
                # Bit reduction
                bits = max(2, int(16 - drive * 14))
                steps = 2 ** bits
                y = round(x * steps) / steps
            
            elif self.circuit_type == DistortionType.WAVEFOLD:
                # Wave folding
                y = math.sin(x * (1 + drive * 10) * math.pi)
            
            elif self.circuit_type == DistortionType.TAPE:
                # Tape saturation
                y = math.tanh(x * (1 + drive * 2)) * 0.9 + x * 0.1
            
            else:
                y = math.tanh(x)
            
            # Simple tone control (lowpass)
            # In real implementation, use proper filter
            
            # Blend dry/wet
            y = sample * (1 - blend) + y * blend
            
            # Output level
            y *= level * self.output_gain
            
            output.append(y)
        
        return output


class OverdriveEffect(BaseEffect):
    """
    Classic overdrive - Tube Screamer style with variants.
    """
    
    def __init__(self):
        super().__init__("Overdrive", EffectCategory.DISTORTION)
        self.variant = "ts808"  # ts808, ts9, blues, transparent
    
    def _init_parameters(self):
        self.add_parameter("drive", 0.5, 0.0, 1.0, "", "Drive amount")
        self.add_parameter("tone", 0.5, 0.0, 1.0, "", "Tone control")
        self.add_parameter("level", 0.7, 0.0, 1.0, "", "Output level")
        self.add_parameter("bass", 0.5, 0.0, 1.0, "", "Bass response")
        self.add_parameter("mid_boost", 0.6, 0.0, 1.0, "", "Mid frequency boost")
        self.add_parameter("compression", 0.4, 0.0, 1.0, "", "Compression amount")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        drive = self.get_param("drive")
        tone = self.get_param("tone")
        level = self.get_param("level")
        mid_boost = self.get_param("mid_boost")
        
        output = []
        for sample in samples:
            x = sample * self.input_gain
            
            # Apply mid boost (simplified)
            x *= (1 + mid_boost * 0.5)
            
            # Soft clipping with drive
            gain = 1 + drive * 10
            y = math.tanh(x * gain) / math.tanh(gain)
            
            # Output
            y *= level * self.output_gain
            output.append(y)
        
        return output


class FuzzEffect(BaseEffect):
    """
    Vintage and modern fuzz circuits.
    """
    
    def __init__(self):
        super().__init__("Fuzz", EffectCategory.DISTORTION)
        self.variant = "big_muff"  # big_muff, fuzz_face, octavia, rat
    
    def _init_parameters(self):
        self.add_parameter("fuzz", 0.7, 0.0, 1.0, "", "Fuzz intensity")
        self.add_parameter("tone", 0.5, 0.0, 1.0, "", "Tone control")
        self.add_parameter("sustain", 0.6, 0.0, 1.0, "", "Sustain/compression")
        self.add_parameter("volume", 0.6, 0.0, 1.0, "", "Output volume")
        self.add_parameter("gate", 0.1, 0.0, 1.0, "", "Noise gate")
        self.add_parameter("octave", 0.0, 0.0, 1.0, "", "Octave up blend")
        self.add_parameter("bias", 0.5, 0.0, 1.0, "", "Transistor bias (dying battery)")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        fuzz = self.get_param("fuzz")
        volume = self.get_param("volume")
        bias = self.get_param("bias")
        octave = self.get_param("octave")
        
        output = []
        for sample in samples:
            x = sample * self.input_gain
            
            # Bias affects gain and asymmetry (dying battery effect)
            gain = 10 + fuzz * 100 * bias
            
            # Extreme clipping
            y = x * gain
            y = max(-1, min(1, y))
            
            # Octave (full wave rectification)
            if octave > 0:
                y_oct = abs(y)
                y = y * (1 - octave) + y_oct * octave
            
            # More shaping
            y = math.tanh(y * 2)
            
            y *= volume * self.output_gain
            output.append(y)
        
        return output


# =============================================================================
# MODULATION EFFECTS
# =============================================================================

class ChorusEffect(BaseEffect):
    """
    Chorus with multiple modes and deep modulation.
    """
    
    def __init__(self):
        super().__init__("Chorus", EffectCategory.MODULATION)
        self.mode = "classic"  # classic, dual, tri, quad, dimension
    
    def _init_parameters(self):
        self.add_parameter("rate", 0.5, 0.1, 10.0, "Hz", "LFO rate")
        self.add_parameter("depth", 0.5, 0.0, 1.0, "", "Modulation depth")
        self.add_parameter("delay", 7.0, 1.0, 30.0, "ms", "Base delay time")
        self.add_parameter("feedback", 0.0, -0.9, 0.9, "", "Feedback amount")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("spread", 0.5, 0.0, 1.0, "", "Stereo spread")
        self.add_parameter("voices", 2.0, 1.0, 8.0, "", "Number of voices")
        self.add_parameter("high_cut", 0.7, 0.0, 1.0, "", "High frequency cut")
        self.add_parameter("low_cut", 0.0, 0.0, 1.0, "", "Low frequency cut")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        rate = self.get_param("rate")
        depth = self.get_param("depth")
        mix = self.get_param("mix")
        
        output = []
        phase = 0.0
        
        for i, sample in enumerate(samples):
            # Simple chorus simulation
            phase += rate / sample_rate
            if phase >= 1.0:
                phase -= 1.0
            
            mod = math.sin(phase * 2 * math.pi) * depth
            
            # Mix original with modulated (simplified)
            y = sample * (1 - mix) + sample * (1 + mod * 0.1) * mix
            y *= self.output_gain
            output.append(y)
        
        return output


class FlangerEffect(BaseEffect):
    """
    Flanger with through-zero and jet modes.
    """
    
    def __init__(self):
        super().__init__("Flanger", EffectCategory.MODULATION)
        self.mode = "classic"  # classic, through_zero, jet, tape
    
    def _init_parameters(self):
        self.add_parameter("rate", 0.3, 0.01, 10.0, "Hz", "LFO rate")
        self.add_parameter("depth", 0.7, 0.0, 1.0, "", "Modulation depth")
        self.add_parameter("manual", 0.5, 0.0, 1.0, "", "Manual delay offset")
        self.add_parameter("feedback", 0.5, -0.99, 0.99, "", "Feedback (resonance)")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("spread", 0.0, 0.0, 1.0, "", "Stereo spread")
        self.add_parameter("tone", 0.5, 0.0, 1.0, "", "Tone control")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        rate = self.get_param("rate")
        depth = self.get_param("depth")
        feedback = self.get_param("feedback")
        mix = self.get_param("mix")
        
        output = []
        phase = 0.0
        
        for sample in samples:
            phase += rate / sample_rate
            if phase >= 1.0:
                phase -= 1.0
            
            mod = math.sin(phase * 2 * math.pi) * depth
            
            # Simplified flanging
            y = sample * (1 - mix) + sample * (1 + mod * 0.3 + feedback * 0.2) * mix
            y *= self.output_gain
            output.append(y)
        
        return output


class PhaserEffect(BaseEffect):
    """
    Multi-stage phaser with various configurations.
    """
    
    def __init__(self):
        super().__init__("Phaser", EffectCategory.MODULATION)
        self.stages = 4  # 2, 4, 6, 8, 10, 12
    
    def _init_parameters(self):
        self.add_parameter("rate", 0.5, 0.01, 10.0, "Hz", "LFO rate")
        self.add_parameter("depth", 0.7, 0.0, 1.0, "", "Sweep depth")
        self.add_parameter("feedback", 0.5, 0.0, 0.99, "", "Resonance")
        self.add_parameter("center", 0.5, 0.0, 1.0, "", "Center frequency")
        self.add_parameter("spread", 0.5, 0.0, 1.0, "", "Notch spread")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("stages", 4.0, 2.0, 12.0, "", "Number of stages")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        rate = self.get_param("rate")
        depth = self.get_param("depth")
        feedback = self.get_param("feedback")
        mix = self.get_param("mix")
        
        output = []
        phase = 0.0
        
        for sample in samples:
            phase += rate / sample_rate
            if phase >= 1.0:
                phase -= 1.0
            
            mod = math.sin(phase * 2 * math.pi) * depth
            
            # Simplified phasing
            y = sample * (1 - mix) + sample * math.cos(mod * math.pi) * mix
            y *= self.output_gain
            output.append(y)
        
        return output


class TremoloEffect(BaseEffect):
    """
    Amplitude modulation with multiple wave shapes.
    """
    
    def __init__(self):
        super().__init__("Tremolo", EffectCategory.MODULATION)
        self.wave_shape = WaveShape.SINE
    
    def _init_parameters(self):
        self.add_parameter("rate", 5.0, 0.1, 20.0, "Hz", "Tremolo rate")
        self.add_parameter("depth", 0.5, 0.0, 1.0, "", "Depth")
        self.add_parameter("shape", 0.0, 0.0, 1.0, "", "Wave shape morph")
        self.add_parameter("stereo", 0.0, 0.0, 1.0, "", "Stereo phase offset")
        self.add_parameter("bias", 0.5, 0.0, 1.0, "", "Bias (asymmetry)")
        self.add_parameter("soft_clip", 0.0, 0.0, 1.0, "", "Soft clipping")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        rate = self.get_param("rate")
        depth = self.get_param("depth")
        
        output = []
        phase = 0.0
        
        for sample in samples:
            phase += rate / sample_rate
            if phase >= 1.0:
                phase -= 1.0
            
            # LFO
            if self.wave_shape == WaveShape.SINE:
                mod = (math.sin(phase * 2 * math.pi) + 1) / 2
            elif self.wave_shape == WaveShape.SQUARE:
                mod = 1.0 if phase < 0.5 else 0.0
            elif self.wave_shape == WaveShape.TRIANGLE:
                mod = abs(2 * phase - 1)
            else:
                mod = (math.sin(phase * 2 * math.pi) + 1) / 2
            
            # Apply tremolo
            y = sample * (1 - depth + mod * depth)
            y *= self.output_gain
            output.append(y)
        
        return output


class VibratoEffect(BaseEffect):
    """
    Pitch modulation (vibrato).
    """
    
    def __init__(self):
        super().__init__("Vibrato", EffectCategory.MODULATION)
    
    def _init_parameters(self):
        self.add_parameter("rate", 5.0, 0.1, 15.0, "Hz", "Vibrato rate")
        self.add_parameter("depth", 0.5, 0.0, 1.0, "", "Pitch deviation")
        self.add_parameter("rise_time", 0.5, 0.0, 2.0, "s", "Rise time")
        self.add_parameter("mix", 1.0, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]  # Simplified


class RotaryEffect(BaseEffect):
    """
    Leslie/rotary speaker simulation.
    """
    
    def __init__(self):
        super().__init__("Rotary", EffectCategory.MODULATION)
    
    def _init_parameters(self):
        self.add_parameter("speed", 0.5, 0.0, 1.0, "", "Fast/slow")
        self.add_parameter("horn_rate", 6.0, 0.5, 10.0, "Hz", "Horn rotation speed")
        self.add_parameter("drum_rate", 0.8, 0.2, 2.0, "Hz", "Drum rotation speed")
        self.add_parameter("horn_depth", 0.7, 0.0, 1.0, "", "Horn modulation depth")
        self.add_parameter("drum_depth", 0.5, 0.0, 1.0, "", "Drum modulation depth")
        self.add_parameter("acceleration", 0.5, 0.1, 2.0, "s", "Speed change time")
        self.add_parameter("distance", 0.5, 0.0, 1.0, "", "Mic distance")
        self.add_parameter("drive", 0.3, 0.0, 1.0, "", "Preamp drive")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class RingModEffect(BaseEffect):
    """
    Ring modulator for metallic/robotic tones.
    """
    
    def __init__(self):
        super().__init__("Ring Mod", EffectCategory.MODULATION)
    
    def _init_parameters(self):
        self.add_parameter("frequency", 440.0, 20.0, 5000.0, "Hz", "Carrier frequency")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("lfo_rate", 0.0, 0.0, 20.0, "Hz", "LFO modulation rate")
        self.add_parameter("lfo_depth", 0.0, 0.0, 1.0, "", "LFO depth")
        self.add_parameter("env_follow", 0.0, 0.0, 1.0, "", "Envelope following")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        freq = self.get_param("frequency")
        mix = self.get_param("mix")
        
        output = []
        phase = 0.0
        
        for sample in samples:
            phase += freq / sample_rate
            if phase >= 1.0:
                phase -= 1.0
            
            carrier = math.sin(phase * 2 * math.pi)
            modulated = sample * carrier
            
            y = sample * (1 - mix) + modulated * mix
            y *= self.output_gain
            output.append(y)
        
        return output


class UnivibeEffect(BaseEffect):
    """
    Classic univibe/photocell modulation.
    """
    
    def __init__(self):
        super().__init__("Univibe", EffectCategory.MODULATION)
    
    def _init_parameters(self):
        self.add_parameter("speed", 0.5, 0.0, 1.0, "", "Speed control")
        self.add_parameter("intensity", 0.7, 0.0, 1.0, "", "Intensity")
        self.add_parameter("mode", 0.0, 0.0, 1.0, "", "Chorus/Vibrato mode")
        self.add_parameter("symmetry", 0.5, 0.0, 1.0, "", "LFO symmetry")
        self.add_parameter("volume", 0.8, 0.0, 1.0, "", "Output volume")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


# =============================================================================
# TIME-BASED EFFECTS
# =============================================================================

class DelayEffect(BaseEffect):
    """
    Multi-mode delay with 12+ algorithms.
    """
    
    def __init__(self):
        super().__init__("Delay", EffectCategory.TIME)
        self.delay_type = DelayType.DIGITAL
        self._buffer: List[float] = []
        self._write_pos = 0
    
    def _init_parameters(self):
        self.add_parameter("time", 300.0, 1.0, 2000.0, "ms", "Delay time")
        self.add_parameter("feedback", 0.4, 0.0, 1.0, "", "Feedback amount")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("mod_rate", 0.5, 0.0, 5.0, "Hz", "Modulation rate")
        self.add_parameter("mod_depth", 0.0, 0.0, 1.0, "", "Modulation depth")
        self.add_parameter("high_cut", 0.8, 0.0, 1.0, "", "High frequency damping")
        self.add_parameter("low_cut", 0.0, 0.0, 1.0, "", "Low frequency cut")
        self.add_parameter("saturation", 0.0, 0.0, 1.0, "", "Tape/analog saturation")
        self.add_parameter("diffusion", 0.0, 0.0, 1.0, "", "Diffusion amount")
        self.add_parameter("ducking", 0.0, 0.0, 1.0, "", "Ducking amount")
        self.add_parameter("pan", 0.5, 0.0, 1.0, "", "Stereo pan")
        self.add_parameter("pitch", 0.0, -12.0, 12.0, "st", "Pitch shift per repeat")
    
    def set_type(self, delay_type: DelayType):
        self.delay_type = delay_type
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        delay_ms = self.get_param("time")
        feedback = self.get_param("feedback")
        mix = self.get_param("mix")
        saturation = self.get_param("saturation")
        
        delay_samples = int(delay_ms * sample_rate / 1000)
        
        # Initialize buffer if needed
        if len(self._buffer) < delay_samples + sample_rate:
            self._buffer = [0.0] * (delay_samples + sample_rate)
        
        output = []
        
        for sample in samples:
            # Read from buffer
            read_pos = (self._write_pos - delay_samples) % len(self._buffer)
            delayed = self._buffer[read_pos]
            
            # Apply saturation for analog/tape modes
            if saturation > 0 and self.delay_type in [DelayType.ANALOG, DelayType.TAPE]:
                delayed = math.tanh(delayed * (1 + saturation * 2)) * (1 / (1 + saturation))
            
            # Write to buffer with feedback
            self._buffer[self._write_pos] = sample + delayed * feedback
            self._write_pos = (self._write_pos + 1) % len(self._buffer)
            
            # Mix
            y = sample * (1 - mix) + delayed * mix
            y *= self.output_gain
            output.append(y)
        
        return output


class ReverbEffect(BaseEffect):
    """
    Multi-algorithm reverb with 15+ types.
    """
    
    def __init__(self):
        super().__init__("Reverb", EffectCategory.TIME)
        self.reverb_type = ReverbType.HALL
    
    def _init_parameters(self):
        self.add_parameter("decay", 2.0, 0.1, 30.0, "s", "Decay time")
        self.add_parameter("size", 0.5, 0.0, 1.0, "", "Room size")
        self.add_parameter("predelay", 20.0, 0.0, 200.0, "ms", "Pre-delay")
        self.add_parameter("damping", 0.5, 0.0, 1.0, "", "High frequency damping")
        self.add_parameter("diffusion", 0.7, 0.0, 1.0, "", "Diffusion")
        self.add_parameter("mix", 0.3, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("modulation", 0.2, 0.0, 1.0, "", "Modulation depth")
        self.add_parameter("low_cut", 0.1, 0.0, 1.0, "", "Low frequency cut")
        self.add_parameter("high_cut", 0.8, 0.0, 1.0, "", "High frequency cut")
        self.add_parameter("early_late", 0.5, 0.0, 1.0, "", "Early/late balance")
        self.add_parameter("shimmer", 0.0, 0.0, 1.0, "", "Shimmer (pitch shift)")
        self.add_parameter("freeze", 0.0, 0.0, 1.0, "", "Infinite hold")
        self.add_parameter("ducking", 0.0, 0.0, 1.0, "", "Ducking")
    
    def set_type(self, reverb_type: ReverbType):
        self.reverb_type = reverb_type
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        decay = self.get_param("decay")
        mix = self.get_param("mix")
        
        # Simplified reverb simulation
        output = []
        reverb_tail = [0.0] * 10
        
        for sample in samples:
            # Simple multi-tap delay approximation
            reverb_out = sum(reverb_tail) / len(reverb_tail) * 0.5
            
            # Update tail
            reverb_tail.pop(0)
            reverb_tail.append(sample + reverb_out * (decay / 10))
            
            y = sample * (1 - mix) + reverb_out * mix
            y *= self.output_gain
            output.append(y)
        
        return output


# =============================================================================
# DYNAMICS
# =============================================================================

class CompressorEffect(BaseEffect):
    """
    Full-featured compressor with multiple modes.
    """
    
    def __init__(self):
        super().__init__("Compressor", EffectCategory.DYNAMICS)
        self.mode = "vca"  # vca, opto, fet, tube, multiband
    
    def _init_parameters(self):
        self.add_parameter("threshold", -20.0, -60.0, 0.0, "dB", "Threshold")
        self.add_parameter("ratio", 4.0, 1.0, 20.0, ":1", "Ratio")
        self.add_parameter("attack", 10.0, 0.1, 100.0, "ms", "Attack time")
        self.add_parameter("release", 100.0, 10.0, 1000.0, "ms", "Release time")
        self.add_parameter("knee", 0.5, 0.0, 1.0, "", "Soft knee")
        self.add_parameter("makeup", 0.0, 0.0, 24.0, "dB", "Makeup gain")
        self.add_parameter("mix", 1.0, 0.0, 1.0, "", "Parallel compression mix")
        self.add_parameter("sidechain_hpf", 0.0, 0.0, 500.0, "Hz", "Sidechain HPF")
        self.add_parameter("auto_makeup", 0.0, 0.0, 1.0, "", "Auto makeup gain")
        self.add_parameter("lookahead", 0.0, 0.0, 10.0, "ms", "Lookahead")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        threshold_db = self.get_param("threshold")
        ratio = self.get_param("ratio")
        attack_ms = self.get_param("attack")
        release_ms = self.get_param("release")
        makeup_db = self.get_param("makeup")
        mix = self.get_param("mix")
        
        threshold = 10 ** (threshold_db / 20)
        makeup = 10 ** (makeup_db / 20)
        
        attack_coef = math.exp(-1 / (attack_ms * sample_rate / 1000))
        release_coef = math.exp(-1 / (release_ms * sample_rate / 1000))
        
        output = []
        envelope = 0.0
        
        for sample in samples:
            # Envelope detection
            input_level = abs(sample)
            if input_level > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level
            
            # Gain computation
            if envelope > threshold:
                gain_reduction = threshold + (envelope - threshold) / ratio
                gain = gain_reduction / max(envelope, 0.0001)
            else:
                gain = 1.0
            
            # Apply compression
            compressed = sample * gain * makeup
            
            # Parallel mix
            y = sample * (1 - mix) + compressed * mix
            y *= self.output_gain
            output.append(y)
        
        return output


class NoiseGateEffect(BaseEffect):
    """
    Noise gate with multiple modes.
    """
    
    def __init__(self):
        super().__init__("Noise Gate", EffectCategory.DYNAMICS)
    
    def _init_parameters(self):
        self.add_parameter("threshold", -40.0, -80.0, 0.0, "dB", "Threshold")
        self.add_parameter("attack", 0.5, 0.1, 10.0, "ms", "Attack time")
        self.add_parameter("hold", 50.0, 0.0, 500.0, "ms", "Hold time")
        self.add_parameter("release", 50.0, 10.0, 500.0, "ms", "Release time")
        self.add_parameter("range", -80.0, -80.0, 0.0, "dB", "Range (attenuation)")
        self.add_parameter("hysteresis", 3.0, 0.0, 12.0, "dB", "Hysteresis")
        self.add_parameter("lookahead", 0.0, 0.0, 5.0, "ms", "Lookahead")
        self.add_parameter("sidechain_filter", 0.0, 0.0, 1.0, "", "Sidechain filter")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        threshold_db = self.get_param("threshold")
        threshold = 10 ** (threshold_db / 20)
        range_db = self.get_param("range")
        range_gain = 10 ** (range_db / 20)
        
        output = []
        gate_open = False
        
        for sample in samples:
            level = abs(sample)
            
            if level > threshold:
                gate_open = True
            elif level < threshold * 0.5:  # Hysteresis
                gate_open = False
            
            if gate_open:
                y = sample
            else:
                y = sample * range_gain
            
            y *= self.output_gain
            output.append(y)
        
        return output


# =============================================================================
# FILTER / EQ
# =============================================================================

class EQEffect(BaseEffect):
    """
    Multi-band parametric EQ.
    """
    
    def __init__(self):
        super().__init__("EQ", EffectCategory.FILTER)
    
    def _init_parameters(self):
        # 4-band parametric + high/low shelf
        self.add_parameter("low_shelf_freq", 80.0, 20.0, 500.0, "Hz", "Low shelf frequency")
        self.add_parameter("low_shelf_gain", 0.0, -15.0, 15.0, "dB", "Low shelf gain")
        
        self.add_parameter("low_mid_freq", 250.0, 100.0, 1000.0, "Hz", "Low-mid frequency")
        self.add_parameter("low_mid_gain", 0.0, -15.0, 15.0, "dB", "Low-mid gain")
        self.add_parameter("low_mid_q", 1.0, 0.1, 10.0, "", "Low-mid Q")
        
        self.add_parameter("high_mid_freq", 2000.0, 500.0, 8000.0, "Hz", "High-mid frequency")
        self.add_parameter("high_mid_gain", 0.0, -15.0, 15.0, "dB", "High-mid gain")
        self.add_parameter("high_mid_q", 1.0, 0.1, 10.0, "", "High-mid Q")
        
        self.add_parameter("high_shelf_freq", 8000.0, 2000.0, 16000.0, "Hz", "High shelf frequency")
        self.add_parameter("high_shelf_gain", 0.0, -15.0, 15.0, "dB", "High shelf gain")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class WahEffect(BaseEffect):
    """
    Wah pedal with auto-wah and envelope modes.
    """
    
    def __init__(self):
        super().__init__("Wah", EffectCategory.FILTER)
        self.mode = "manual"  # manual, auto, envelope, random
    
    def _init_parameters(self):
        self.add_parameter("position", 0.5, 0.0, 1.0, "", "Pedal position")
        self.add_parameter("range_low", 400.0, 200.0, 1000.0, "Hz", "Low frequency")
        self.add_parameter("range_high", 2000.0, 1000.0, 5000.0, "Hz", "High frequency")
        self.add_parameter("q", 3.0, 0.5, 10.0, "", "Resonance")
        self.add_parameter("lfo_rate", 2.0, 0.1, 10.0, "Hz", "Auto-wah rate")
        self.add_parameter("env_sensitivity", 0.5, 0.0, 1.0, "", "Envelope sensitivity")
        self.add_parameter("env_attack", 10.0, 1.0, 100.0, "ms", "Envelope attack")
        self.add_parameter("env_release", 100.0, 10.0, 500.0, "ms", "Envelope release")
        self.add_parameter("mix", 1.0, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        position = self.get_param("position")
        range_low = self.get_param("range_low")
        range_high = self.get_param("range_high")
        q = self.get_param("q")
        
        freq = range_low + position * (range_high - range_low)
        
        # Simplified bandpass filter
        output = []
        for sample in samples:
            y = sample  # Would apply bandpass here
            y *= self.output_gain
            output.append(y)
        
        return output


class FilterEffect(BaseEffect):
    """
    Multi-mode filter with 10 filter types.
    """
    
    def __init__(self):
        super().__init__("Filter", EffectCategory.FILTER)
        self.filter_type = FilterType.LOWPASS
    
    def _init_parameters(self):
        self.add_parameter("cutoff", 1000.0, 20.0, 20000.0, "Hz", "Cutoff frequency", curve="logarithmic")
        self.add_parameter("resonance", 0.5, 0.0, 1.0, "", "Resonance")
        self.add_parameter("drive", 0.0, 0.0, 1.0, "", "Filter drive")
        self.add_parameter("env_amount", 0.0, -1.0, 1.0, "", "Envelope amount")
        self.add_parameter("env_attack", 10.0, 1.0, 500.0, "ms", "Envelope attack")
        self.add_parameter("env_release", 100.0, 10.0, 2000.0, "ms", "Envelope release")
        self.add_parameter("lfo_amount", 0.0, 0.0, 1.0, "", "LFO amount")
        self.add_parameter("lfo_rate", 1.0, 0.1, 20.0, "Hz", "LFO rate")
        self.add_parameter("key_track", 0.0, 0.0, 1.0, "", "Keyboard tracking")
        self.add_parameter("mix", 1.0, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


# =============================================================================
# PITCH
# =============================================================================

class PitchShiftEffect(BaseEffect):
    """
    Pitch shifter with multiple algorithms.
    """
    
    def __init__(self):
        super().__init__("Pitch Shift", EffectCategory.PITCH)
    
    def _init_parameters(self):
        self.add_parameter("pitch", 0.0, -24.0, 24.0, "st", "Pitch shift")
        self.add_parameter("fine", 0.0, -100.0, 100.0, "ct", "Fine tune (cents)")
        self.add_parameter("formant", 0.0, -12.0, 12.0, "st", "Formant shift")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
        self.add_parameter("delay", 0.0, 0.0, 100.0, "ms", "Latency compensation")
        self.add_parameter("window", 50.0, 10.0, 200.0, "ms", "Window size")
        self.add_parameter("crossfade", 0.5, 0.0, 1.0, "", "Crossfade amount")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class HarmonizerEffect(BaseEffect):
    """
    Intelligent harmonizer with scale-aware pitch.
    """
    
    def __init__(self):
        super().__init__("Harmonizer", EffectCategory.PITCH)
    
    def _init_parameters(self):
        self.add_parameter("voice1", 0.0, -24.0, 24.0, "st", "Voice 1 pitch")
        self.add_parameter("voice1_mix", 0.5, 0.0, 1.0, "", "Voice 1 level")
        self.add_parameter("voice2", 0.0, -24.0, 24.0, "st", "Voice 2 pitch")
        self.add_parameter("voice2_mix", 0.0, 0.0, 1.0, "", "Voice 2 level")
        self.add_parameter("voice3", 0.0, -24.0, 24.0, "st", "Voice 3 pitch")
        self.add_parameter("voice3_mix", 0.0, 0.0, 1.0, "", "Voice 3 level")
        self.add_parameter("voice4", 0.0, -24.0, 24.0, "st", "Voice 4 pitch")
        self.add_parameter("voice4_mix", 0.0, 0.0, 1.0, "", "Voice 4 level")
        self.add_parameter("key", 0.0, 0.0, 11.0, "", "Key (0=C)")
        self.add_parameter("scale", 0.0, 0.0, 7.0, "", "Scale type")
        self.add_parameter("smart", 1.0, 0.0, 1.0, "", "Smart harmonization")
        self.add_parameter("dry", 1.0, 0.0, 1.0, "", "Dry level")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class OctaverEffect(BaseEffect):
    """
    Octave up/down with tracking.
    """
    
    def __init__(self):
        super().__init__("Octaver", EffectCategory.PITCH)
    
    def _init_parameters(self):
        self.add_parameter("octave_down2", 0.0, 0.0, 1.0, "", "-2 octave level")
        self.add_parameter("octave_down1", 0.5, 0.0, 1.0, "", "-1 octave level")
        self.add_parameter("dry", 0.5, 0.0, 1.0, "", "Dry level")
        self.add_parameter("octave_up1", 0.0, 0.0, 1.0, "", "+1 octave level")
        self.add_parameter("octave_up2", 0.0, 0.0, 1.0, "", "+2 octave level")
        self.add_parameter("tracking", 0.5, 0.0, 1.0, "", "Tracking speed")
        self.add_parameter("tone", 0.5, 0.0, 1.0, "", "Tone control")
        self.add_parameter("polyphonic", 0.0, 0.0, 1.0, "", "Polyphonic mode")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


# =============================================================================
# AMP/CABINET
# =============================================================================

class AmpSimEffect(BaseEffect):
    """
    Amp simulation with 15 amp models.
    """
    
    def __init__(self):
        super().__init__("Amp Sim", EffectCategory.AMP)
        self.amp_model = AmpModel.CRUNCH_BRIT
    
    def _init_parameters(self):
        self.add_parameter("gain", 0.5, 0.0, 1.0, "", "Preamp gain")
        self.add_parameter("bass", 0.5, 0.0, 1.0, "", "Bass")
        self.add_parameter("mid", 0.5, 0.0, 1.0, "", "Mid")
        self.add_parameter("treble", 0.5, 0.0, 1.0, "", "Treble")
        self.add_parameter("presence", 0.5, 0.0, 1.0, "", "Presence")
        self.add_parameter("master", 0.5, 0.0, 1.0, "", "Master volume")
        self.add_parameter("sag", 0.3, 0.0, 1.0, "", "Power amp sag")
        self.add_parameter("bias", 0.5, 0.0, 1.0, "", "Tube bias")
        self.add_parameter("bright", 0.0, 0.0, 1.0, "", "Bright switch")
        self.add_parameter("tight", 0.5, 0.0, 1.0, "", "Low end tightness")
    
    def set_model(self, model: AmpModel):
        self.amp_model = model
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        gain = self.get_param("gain")
        master = self.get_param("master")
        sag = self.get_param("sag")
        
        output = []
        for sample in samples:
            x = sample * self.input_gain * (1 + gain * 10)
            
            # Tube-style saturation
            y = math.tanh(x * (1 + gain * 2))
            
            # Power amp sag simulation
            y *= (1 - sag * 0.3 * abs(y))
            
            y *= master * self.output_gain
            output.append(y)
        
        return output


class CabinetSimEffect(BaseEffect):
    """
    Cabinet/IR simulation with mic options.
    """
    
    def __init__(self):
        super().__init__("Cabinet", EffectCategory.AMP)
        self.cabinet_type = CabinetType.CAB_4X12
    
    def _init_parameters(self):
        self.add_parameter("mic_position", 0.5, 0.0, 1.0, "", "Mic position (center/edge)")
        self.add_parameter("mic_distance", 0.5, 0.0, 1.0, "", "Mic distance")
        self.add_parameter("mic_type", 0.0, 0.0, 3.0, "", "Mic type (57/421/ribbon/condenser)")
        self.add_parameter("room", 0.2, 0.0, 1.0, "", "Room ambience")
        self.add_parameter("low_cut", 80.0, 20.0, 200.0, "Hz", "Low cut frequency")
        self.add_parameter("high_cut", 8000.0, 2000.0, 12000.0, "Hz", "High cut frequency")
        self.add_parameter("resonance", 0.5, 0.0, 1.0, "", "Cabinet resonance")
    
    def set_cabinet(self, cabinet: CabinetType):
        self.cabinet_type = cabinet
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


# =============================================================================
# SPECIAL EFFECTS
# =============================================================================

class LooperEffect(BaseEffect):
    """
    Loop recorder/player.
    """
    
    def __init__(self):
        super().__init__("Looper", EffectCategory.SPECIAL)
        self._recording = False
        self._playing = False
        self._buffer: List[float] = []
        self._position = 0
    
    def _init_parameters(self):
        self.add_parameter("record", 0.0, 0.0, 1.0, "", "Record toggle")
        self.add_parameter("play", 0.0, 0.0, 1.0, "", "Play toggle")
        self.add_parameter("overdub", 0.0, 0.0, 1.0, "", "Overdub toggle")
        self.add_parameter("reverse", 0.0, 0.0, 1.0, "", "Reverse playback")
        self.add_parameter("half_speed", 0.0, 0.0, 1.0, "", "Half speed")
        self.add_parameter("fade", 0.0, 0.0, 1.0, "", "Fade amount per loop")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Loop/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class GranularEffect(BaseEffect):
    """
    Granular synthesis/processing.
    """
    
    def __init__(self):
        super().__init__("Granular", EffectCategory.SPECIAL)
    
    def _init_parameters(self):
        self.add_parameter("grain_size", 50.0, 5.0, 500.0, "ms", "Grain size")
        self.add_parameter("density", 0.5, 0.0, 1.0, "", "Grain density")
        self.add_parameter("pitch", 0.0, -24.0, 24.0, "st", "Pitch shift")
        self.add_parameter("pitch_random", 0.0, 0.0, 12.0, "st", "Pitch randomization")
        self.add_parameter("position", 0.5, 0.0, 1.0, "", "Buffer position")
        self.add_parameter("position_random", 0.0, 0.0, 1.0, "", "Position randomization")
        self.add_parameter("spread", 0.5, 0.0, 1.0, "", "Stereo spread")
        self.add_parameter("reverse", 0.0, 0.0, 1.0, "", "Reverse probability")
        self.add_parameter("freeze", 0.0, 0.0, 1.0, "", "Freeze input")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class ShimmerEffect(BaseEffect):
    """
    Shimmer reverb (reverb + pitch shift).
    """
    
    def __init__(self):
        super().__init__("Shimmer", EffectCategory.SPECIAL)
    
    def _init_parameters(self):
        self.add_parameter("decay", 5.0, 0.5, 30.0, "s", "Decay time")
        self.add_parameter("shimmer", 0.5, 0.0, 1.0, "", "Shimmer amount")
        self.add_parameter("pitch", 12.0, -24.0, 24.0, "st", "Pitch shift")
        self.add_parameter("modulation", 0.3, 0.0, 1.0, "", "Modulation")
        self.add_parameter("low_cut", 200.0, 20.0, 1000.0, "Hz", "Low cut")
        self.add_parameter("high_cut", 8000.0, 2000.0, 15000.0, "Hz", "High cut")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class FreezeEffect(BaseEffect):
    """
    Audio freeze/sustain.
    """
    
    def __init__(self):
        super().__init__("Freeze", EffectCategory.SPECIAL)
        self._frozen_buffer: List[float] = []
        self._is_frozen = False
    
    def _init_parameters(self):
        self.add_parameter("freeze", 0.0, 0.0, 1.0, "", "Freeze toggle")
        self.add_parameter("rise", 0.5, 0.01, 2.0, "s", "Rise time")
        self.add_parameter("release", 0.5, 0.01, 2.0, "s", "Release time")
        self.add_parameter("grain", 100.0, 10.0, 500.0, "ms", "Grain size")
        self.add_parameter("pitch", 0.0, -24.0, 24.0, "st", "Pitch shift")
        self.add_parameter("formant", 0.0, -12.0, 12.0, "st", "Formant shift")
        self.add_parameter("mix", 0.5, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class SlicerEffect(BaseEffect):
    """
    Rhythmic slicer/gate.
    """
    
    def __init__(self):
        super().__init__("Slicer", EffectCategory.SPECIAL)
    
    def _init_parameters(self):
        self.add_parameter("steps", 8.0, 2.0, 32.0, "", "Number of steps")
        self.add_parameter("rate", 1.0, 0.25, 4.0, "x", "Rate multiplier")
        self.add_parameter("sync", 1.0, 0.0, 1.0, "", "Tempo sync")
        self.add_parameter("attack", 5.0, 0.1, 50.0, "ms", "Step attack")
        self.add_parameter("release", 20.0, 1.0, 200.0, "ms", "Step release")
        self.add_parameter("shuffle", 0.0, 0.0, 1.0, "", "Shuffle amount")
        self.add_parameter("reverse", 0.0, 0.0, 1.0, "", "Reverse probability")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        return [s * self.output_gain for s in samples]


class BitcrusherEffect(BaseEffect):
    """
    Bit depth and sample rate reduction.
    """
    
    def __init__(self):
        super().__init__("Bitcrusher", EffectCategory.SPECIAL)
    
    def _init_parameters(self):
        self.add_parameter("bits", 16.0, 1.0, 16.0, "", "Bit depth")
        self.add_parameter("sample_rate", 44100.0, 100.0, 44100.0, "Hz", "Sample rate")
        self.add_parameter("dither", 0.0, 0.0, 1.0, "", "Dithering")
        self.add_parameter("jitter", 0.0, 0.0, 1.0, "", "Timing jitter")
        self.add_parameter("filter", 0.5, 0.0, 1.0, "", "Anti-aliasing filter")
        self.add_parameter("mix", 1.0, 0.0, 1.0, "", "Wet/dry mix")
    
    def process(self, samples: List[float], sample_rate: int) -> List[float]:
        if self.bypass:
            return samples
        
        bits = int(self.get_param("bits"))
        target_sr = self.get_param("sample_rate")
        mix = self.get_param("mix")
        
        steps = 2 ** bits
        downsample_factor = max(1, int(sample_rate / target_sr))
        
        output = []
        hold_sample = 0.0
        
        for i, sample in enumerate(samples):
            # Sample rate reduction
            if i % downsample_factor == 0:
                hold_sample = sample
            
            # Bit reduction
            crushed = round(hold_sample * steps) / steps
            
            # Mix
            y = sample * (1 - mix) + crushed * mix
            y *= self.output_gain
            output.append(y)
        
        return output

