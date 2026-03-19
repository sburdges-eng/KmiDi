"""
KellyBrain: central music cognition executive.
Phase 2: audio_model + chord_model + emotion_model -> decide -> next_chord, mood_vector.
Phase 3: jepa (MultiJEPA) + memory + personality -> think -> decide (policy with taste).
"""


class KellyBrain:
    def __init__(self, audio_model=None, chord_model=None, emotion_model=None, jepa=None, language_model=None, memory=None, personality=None):
        self.audio = audio_model
        self.chords = chord_model
        self.emotion = emotion_model
        self.jepa = jepa
        self.language_model = language_model
        self.memory = memory
        self.personality = personality

    def think(self, features):
        """
        features: mel (B, 1, n_mels, time) or latent.
        Returns: decision dict (next_chord, mood_vector, and optionally density, brightness, etc.).
        """
        import torch
        if self.jepa is not None:
            # Phase 3: Multi-JEPA
            if isinstance(features, torch.Tensor):
                with torch.no_grad():
                    states = self.jepa(features)
            else:
                states = self.jepa(features)
            
            # Use Language Model if available to process JEPA states
            if self.language_model is not None:
                decision = self.decide_via_llm(states)
            else:
                decision = self.decide_from_states(states)
        else:
            # Phase 2: audio + chord + emotion
            if isinstance(features, torch.Tensor):
                # Mel: (B, 1, n_mels, time) -> use encoder. Latent: (B, d) or (B, seq, d) -> use directly.
                if features.dim() >= 3:
                    with torch.no_grad():
                        latent = self.audio(features)
                        z = latent.mean(dim=1) if latent.dim() == 3 else latent
                        emotions = self.emotion(z)
                        harmony = self.chords(latent)
                    decision = self.decide(emotions, harmony)
                else:
                    z = features
                    emotions = self.emotion(z)
                    if z.dim() == 1:
                        z_chord = z.unsqueeze(0).unsqueeze(0)
                    elif z.dim() == 2:
                        z_chord = z.unsqueeze(1)
                    else:
                        z_chord = z
                    harmony = self.chords(z_chord)
                    decision = self.decide(emotions, harmony)
            else:
                z = torch.as_tensor(features, dtype=torch.float32)
                if z.dim() == 0:
                    z = z.unsqueeze(0)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                emotions = self.emotion(z)
                if z.dim() == 2:
                    z_chord = z.unsqueeze(1)
                else:
                    z_chord = z
                harmony = self.chords(z_chord)
                decision = self.decide(emotions, harmony)
        if self.memory is not None:
            self.memory.store(decision)
        return decision

    def decide_via_llm(self, states):
        """Use the language model to make high-level decisions from JEPA states."""
        import torch
        # Combine states into a context vector or tokens for the LLM
        # This is a placeholder for actual tokenization/embedding logic
        context = []
        for k, v in states.items():
            if hasattr(v, "mean"):
                context.append(v.mean(dim=0).flatten())
            else:
                context.append(torch.tensor(v).flatten())
        
        context_tensor = torch.cat(context)
        
        with torch.no_grad():
            llm_output = self.language_model(context_tensor)
            
        # Parse LLM output into decision dict
        # Assuming output is a vector of probabilities or logits
        decision = self.decide_from_states(states)
        decision["llm_influence"] = llm_output.mean().item() if hasattr(llm_output, "mean") else 0.0
        return decision

    def decide_from_states(self, states):
        """Decide from Multi-JEPA states dict (emotion, harmony, rhythm, timbre, intent). Optionally condition on personality/memory."""
        import torch
        emotion = states.get("emotion")
        harmony = states.get("harmony")
        if emotion is not None and hasattr(emotion, "mean"):
            mood_vector = emotion.mean(dim=0) if emotion.dim() > 1 else emotion
            if hasattr(mood_vector, "cpu"):
                mood_vector = mood_vector.cpu()
        else:
            mood_vector = emotion
        next_chord = 0
        if harmony is not None and hasattr(harmony, "argmax"):
            h = harmony.mean(dim=1) if harmony.dim() == 3 else harmony
            next_chord = h.argmax(dim=-1)
            if isinstance(next_chord, torch.Tensor):
                next_chord = next_chord.item() if next_chord.numel() == 1 else next_chord.tolist()
        out = {"next_chord": next_chord, "mood_vector": mood_vector}
        if self.personality:
            from ..personality.taste import apply_taste
            taste = apply_taste(self.personality)
            out["chord_bias"] = taste["chord_bias"]
            out["rhythm_complexity"] = taste["rhythm_complexity"]
            out["timbre_brightness"] = taste["timbre_brightness"]
            out["note_density"] = taste["note_density"]
        return out

    def decide(self, emotions, harmony):
        """Produce next_chord and mood_vector from emotion and harmony outputs (Phase 2)."""
        import torch
        if hasattr(emotions, "cpu"):
            emotions = emotions.cpu()
        if hasattr(harmony, "cpu"):
            harmony = harmony.cpu()
        mood_vector = emotions.mean(dim=0) if hasattr(emotions, "dim") and emotions.dim() > 1 else emotions
        if hasattr(harmony, "argmax"):
            if harmony.dim() == 3:
                last_h = harmony[:, -1, :]
                next_chord = last_h.argmax(dim=-1)
            else:
                next_chord = harmony.argmax(dim=-1)
            if isinstance(next_chord, torch.Tensor):
                next_chord = next_chord.item() if next_chord.numel() == 1 else next_chord.tolist()
        else:
            next_chord = 0
        return {"next_chord": next_chord, "mood_vector": mood_vector}
