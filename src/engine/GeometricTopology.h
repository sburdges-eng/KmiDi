#pragma once
/*
 * GeometricTopology.h - VAD Manifold Geometry
 * =============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: VADCalculator (VADState trajectories)
 * - Engine Layer: TemporalMemory, PredictiveTrendAnalyzer (derivatives)
 *
 * Purpose: Curvature, distances, and attractors in valence-arousal-dominance space.
 *
 * Features:
 * - Discrete curvature from velocity/acceleration
 * - Euclidean emotional distance
 * - Stability / attractor heuristics
 */

#include "engine/VADCalculator.h"
#include <vector>

namespace kelly {

class GeometricTopology {
public:
    /**
     * Calculate emotional manifold curvature
     * κ = ||Ė × Ë|| / ||Ė||³
     */
    float calculateCurvature(
        const VADState& state,
        const VADState& velocity,  // First derivative
        const VADState& acceleration  // Second derivative
    ) const;
    
    /**
     * Calculate emotional distance
     * d(E₁, E₂) = √((V₁-V₂)² + (A₁-A₂)² + (D₁-D₂)²)
     */
    float calculateDistance(const VADState& state1, const VADState& state2) const;
    
    /**
     * Find emotional attractors (stable equilibrium points)
     * ∇U_E = 0, det(∇²U_E) > 0
     */
    struct Attractor {
        VADState position;
        float stability;  // Stability measure
        std::string name;  // e.g., "Joy", "Peace", "Flow"
    };
    
    std::vector<Attractor> findAttractors(
        const std::vector<VADState>& candidateStates
    ) const;
    
    /**
     * Calculate stability of a state
     * Based on potential energy second derivative
     */
    float calculateStability(const VADState& state) const;
    
    /**
     * Calculate emotional volume in VAD space
     * Volume of region defined by states
     */
    float calculateVolume(const std::vector<VADState>& states) const;
    
private:
    // Helper functions
    VADState crossProduct(const VADState& a, const VADState& b) const;
    float dotProduct(const VADState& a, const VADState& b) const;
};

} // namespace kelly
