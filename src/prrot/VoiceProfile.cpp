#include "prrot/VoiceProfile.h"
#include <algorithm>
#include <cstring>
#include <cctype>

namespace prrot {

const char* phonemeTypeToString(PhonemeType type) {
    switch (type) {
        // Vowels
        case PhonemeType::AA: return "AA";
        case PhonemeType::AE: return "AE";
        case PhonemeType::AH: return "AH";
        case PhonemeType::AO: return "AO";
        case PhonemeType::AW: return "AW";
        case PhonemeType::AY: return "AY";
        case PhonemeType::EH: return "EH";
        case PhonemeType::ER: return "ER";
        case PhonemeType::EY: return "EY";
        case PhonemeType::IH: return "IH";
        case PhonemeType::IY: return "IY";
        case PhonemeType::OW: return "OW";
        case PhonemeType::OY: return "OY";
        case PhonemeType::UH: return "UH";
        case PhonemeType::UW: return "UW";

        // Consonants - Stops
        case PhonemeType::B: return "B";
        case PhonemeType::D: return "D";
        case PhonemeType::G: return "G";
        case PhonemeType::K: return "K";
        case PhonemeType::P: return "P";
        case PhonemeType::T: return "T";

        // Consonants - Fricatives
        case PhonemeType::CH: return "CH";
        case PhonemeType::DH: return "DH";
        case PhonemeType::F: return "F";
        case PhonemeType::HH: return "HH";
        case PhonemeType::JH: return "JH";
        case PhonemeType::S: return "S";
        case PhonemeType::SH: return "SH";
        case PhonemeType::TH: return "TH";
        case PhonemeType::V: return "V";
        case PhonemeType::Z: return "Z";
        case PhonemeType::ZH: return "ZH";

        // Consonants - Nasals
        case PhonemeType::M: return "M";
        case PhonemeType::N: return "N";
        case PhonemeType::NG: return "NG";

        // Consonants - Liquids/Glides
        case PhonemeType::L: return "L";
        case PhonemeType::R: return "R";
        case PhonemeType::W: return "W";
        case PhonemeType::Y: return "Y";

        // Silence
        case PhonemeType::SIL: return "SIL";

        default:
        case PhonemeType::UNKNOWN: return "UNKNOWN";
    }
}

PhonemeType stringToPhonemeType(const std::string& symbol) {
    // Convert to uppercase for comparison
    std::string upper = symbol;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    // Map strings to phoneme types
    if (upper == "AA") return PhonemeType::AA;
    if (upper == "AE") return PhonemeType::AE;
    if (upper == "AH") return PhonemeType::AH;
    if (upper == "AO") return PhonemeType::AO;
    if (upper == "AW") return PhonemeType::AW;
    if (upper == "AY") return PhonemeType::AY;
    if (upper == "EH") return PhonemeType::EH;
    if (upper == "ER") return PhonemeType::ER;
    if (upper == "EY") return PhonemeType::EY;
    if (upper == "IH") return PhonemeType::IH;
    if (upper == "IY") return PhonemeType::IY;
    if (upper == "OW") return PhonemeType::OW;
    if (upper == "OY") return PhonemeType::OY;
    if (upper == "UH") return PhonemeType::UH;
    if (upper == "UW") return PhonemeType::UW;

    if (upper == "B") return PhonemeType::B;
    if (upper == "D") return PhonemeType::D;
    if (upper == "G") return PhonemeType::G;
    if (upper == "K") return PhonemeType::K;
    if (upper == "P") return PhonemeType::P;
    if (upper == "T") return PhonemeType::T;

    if (upper == "CH") return PhonemeType::CH;
    if (upper == "DH") return PhonemeType::DH;
    if (upper == "F") return PhonemeType::F;
    if (upper == "HH") return PhonemeType::HH;
    if (upper == "JH") return PhonemeType::JH;
    if (upper == "S") return PhonemeType::S;
    if (upper == "SH") return PhonemeType::SH;
    if (upper == "TH") return PhonemeType::TH;
    if (upper == "V") return PhonemeType::V;
    if (upper == "Z") return PhonemeType::Z;
    if (upper == "ZH") return PhonemeType::ZH;

    if (upper == "M") return PhonemeType::M;
    if (upper == "N") return PhonemeType::N;
    if (upper == "NG") return PhonemeType::NG;

    if (upper == "L") return PhonemeType::L;
    if (upper == "R") return PhonemeType::R;
    if (upper == "W") return PhonemeType::W;
    if (upper == "Y") return PhonemeType::Y;

    if (upper == "SIL" || upper == "SILENCE") return PhonemeType::SIL;

    return PhonemeType::UNKNOWN;
}

bool isVowel(PhonemeType phoneme) {
    return phoneme >= PhonemeType::AA && phoneme <= PhonemeType::UW;
}

bool isConsonant(PhonemeType phoneme) {
    return (phoneme >= PhonemeType::B && phoneme <= PhonemeType::T) ||  // Stops
           (phoneme >= PhonemeType::CH && phoneme <= PhonemeType::ZH) || // Fricatives
           (phoneme >= PhonemeType::M && phoneme <= PhonemeType::NG) ||  // Nasals
           (phoneme >= PhonemeType::L && phoneme <= PhonemeType::Y);     // Liquids/Glides
}

} // namespace prrot
