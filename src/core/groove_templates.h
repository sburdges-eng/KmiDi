#pragma once
/*
 * groove_templates.h - Legacy Groove Template Registry
 * =====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: GrooveEngine.h (rich templates + humanization)
 * - Core Layer: Named map of simple GrooveTemplate structs
 *
 * Purpose: Static groove presets for early prototypes.
 *
 * Features:
 * - initializeTemplates catalog
 * - Lookup by name
 */

#include <string>
#include <vector>
#include <map>

namespace kelly {

struct GrooveTemplate {
    std::string name;
    int numerator;
    int denominator;
    std::vector<std::pair<float, int>> pattern;  // time, velocity
    float swing = 0.0f;
};

class GrooveTemplates {
public:
    GrooveTemplates();
    ~GrooveTemplates() = default;

    const GrooveTemplate* getTemplate(const std::string& name) const;
    std::vector<std::string> getTemplateNames() const;

private:
    void initializeTemplates();
    std::map<std::string, GrooveTemplate> templates_;
};

} // namespace kelly
