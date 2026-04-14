#pragma once
/*
 * HybridCoupling.h - AI vs Human Emotional State Mixing
 * =======================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: QuantumEmotionalField (QuantumEmotionalState operands)
 * - ML / Bridge Layers: upstream “AI” affect estimates
 * - UI / Input: human-reported or biometric-derived states
 *
 * Purpose: Blends machine-inferred and human emotional wavefunctions with
 *          tunable coupling strength.
 *
 * Features:
 * - Normalized α/β mixing coefficients
 * - Cross-influence energy term
 */

#include "engine/QuantumEmotionalField.h"
#include <complex>

namespace kelly {

class HybridCoupling {
public:
    HybridCoupling(float alpha = 0.5f, float beta = 0.5f, float kappa = 0.3f);
    
    /**
     * Create hybrid emotional state
     * Ψ_hybrid = αΨ_AI + βΨ_human
     */
    QuantumEmotionalState createHybridState(
        const QuantumEmotionalState& aiState,
        const QuantumEmotionalState& humanState
    ) const;
    
    /**
     * Calculate cross-influence term
     * ΔH = κ Re(Ψ_AI* Ψ_human)
     */
    float calculateCrossInfluence(
        const QuantumEmotionalState& aiState,
        const QuantumEmotionalState& humanState
    ) const;
    
    /**
     * Calculate coherence between AI and human states
     */
    float calculateCoherence(
        const QuantumEmotionalState& aiState,
        const QuantumEmotionalState& humanState
    ) const;
    
    /**
     * Set coupling parameters
     */
    void setParameters(float alpha, float beta, float kappa) {
        alpha_ = alpha;
        beta_ = beta;
        kappa_ = kappa;
        normalizeWeights();
    }
    
    float getAlpha() const { return alpha_; }
    float getBeta() const { return beta_; }
    float getKappa() const { return kappa_; }
    
private:
    float alpha_, beta_, kappa_;  // Coupling parameters
    
    void normalizeWeights();
};

} // namespace kelly
