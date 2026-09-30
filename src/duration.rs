// =============================================================================
// Rule-based duration estimation
// =============================================================================
//
// A masked diffusion model has to be told how long its output is before it
// generates any of it: every frame exists, masked, from the first step. Get it
// wrong and the failure is not graceful. Too few frames truncates mid-word;
// too many and the model has nothing to say for the remainder and collapses
// into near-silence -- measured on one Italian sentence, asking for 10 s of a
// 4 s phrase gave a peak of 0.001 and 98% silence.
//
// So OmniVoice ships an estimator, and it is not a neural one. It is a lookup
// table: every character gets a phonetic weight relative to one Latin letter,
// the weights are summed, and the sum is scaled against a reference phrase of
// known length. This is a port of `RuleDurationEstimator` from
// `omnivoice/utils/duration.py` (k2-fsa/OmniVoice, Apache 2.0, Xiaomi Corp.).
//
// ## Why a weight per character
//
// Characters are not equally expensive to say. A CJK ideograph is a whole
// syllable, a Devanagari cluster is a consonant plus its vowel, and a combining
// accent is silent -- it modifies the letter it follows and adds no time. A
// digit is the outlier: "2024" is four characters and roughly fourteen Latin
// letters' worth of speech, because it is spoken as words.
//
//     cjk 3.0   ethiopic 3.0   yi 3.0        hangul 2.5     kana 2.2
//     indic 1.8 khmer_myanmar 1.8            thai_lao 1.5   arabic 1.5
//     hebrew 1.5                             latin 1.0      cyrillic 1.0
//     greek 1.0 armenian 1.0                 georgian 1.0   default 1.0
//     punctuation 0.5             space 0.2  digit 3.5      mark 0.0
//
// ## How a character is classified
//
// Unicode general category first, script block second. The order matters:
// a Devanagari vowel sign sits inside the Devanagari block but is a combining
// mark, and classifying it by block would charge 1.8 for something silent.
// So marks, punctuation, symbols, separators and numbers are pulled out by
// category, and only what is left is looked up by block.
//
// ## The short-text boost
//
// Linear extrapolation underestimates short utterances, because the fixed
// costs -- onset, final lengthening, the breath at the end -- do not shrink
// with the text. Below 50 frames (2 s) the estimate is pulled up a power
// curve, `50 * (est/50)^(1/3)`, which leaves 50 alone and lifts 25 to 40.

#![allow(dead_code)]

// =============================================================================
// Character classes
// =============================================================================

/// The phonetic class of a character. Each carries one weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CharClass {
    // Logographic -- one character is a syllable or a word.
    Cjk,
    Ethiopic,
    Yi,
    // Syllabic blocks.
    Hangul,
    Kana,
    // Abugidas -- a consonant carrying its vowel.
    Indic,
    KhmerMyanmar,
    ThaiLao,
    // Abjads -- consonant-heavy, vowels often unwritten.
    Arabic,
    Hebrew,
    // Segmental alphabets, the 1.0 baseline.
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Georgian,
    /// Anything else with no entry of its own, also at 1.0.
    Default,
    // Category-driven classes, which win over the block lookup.
    Punctuation,
    Space,
    Digit,
    Mark,
}

// =============================================================================
// Script blocks
// =============================================================================
//
// Each entry is the *last* code point of a block and the class it belongs to,
// sorted ascending, so a code point is classified by finding the first end that
// is not below it. The list starts at U+02AF, which means every code point
// under that -- ASCII control characters included -- lands in Latin unless the
// category pass took it first.
//
// The blocks are the reference's, comments and all; several of them ("Cherokee",
// "Mongolian") deliberately map to Default rather than being given a weight
// nobody measured.

struct Block {
    end: u32,
    cls: CharClass,
}

const BLOCKS: [Block; 88] = [
    Block { end: 0x02AF, cls: CharClass::Latin },        // Latin (Basic, Supplement, Ext, IPA)
    Block { end: 0x03FF, cls: CharClass::Greek },        // Greek & Coptic
    Block { end: 0x052F, cls: CharClass::Cyrillic },     // Cyrillic
    Block { end: 0x058F, cls: CharClass::Armenian },     // Armenian
    Block { end: 0x05FF, cls: CharClass::Hebrew },       // Hebrew
    Block { end: 0x077F, cls: CharClass::Arabic },       // Arabic, Syriac, Arabic Supplement
    Block { end: 0x089F, cls: CharClass::Arabic },       // Arabic Extended-B (+ Syriac Supp)
    Block { end: 0x08FF, cls: CharClass::Arabic },       // Arabic Extended-A
    Block { end: 0x097F, cls: CharClass::Indic },        // Devanagari
    Block { end: 0x09FF, cls: CharClass::Indic },        // Bengali
    Block { end: 0x0A7F, cls: CharClass::Indic },        // Gurmukhi
    Block { end: 0x0AFF, cls: CharClass::Indic },        // Gujarati
    Block { end: 0x0B7F, cls: CharClass::Indic },        // Oriya
    Block { end: 0x0BFF, cls: CharClass::Indic },        // Tamil
    Block { end: 0x0C7F, cls: CharClass::Indic },        // Telugu
    Block { end: 0x0CFF, cls: CharClass::Indic },        // Kannada
    Block { end: 0x0D7F, cls: CharClass::Indic },        // Malayalam
    Block { end: 0x0DFF, cls: CharClass::Indic },        // Sinhala
    Block { end: 0x0EFF, cls: CharClass::ThaiLao },      // Thai & Lao
    Block { end: 0x0FFF, cls: CharClass::Indic },        // Tibetan (Abugida)
    Block { end: 0x109F, cls: CharClass::KhmerMyanmar }, // Myanmar
    Block { end: 0x10FF, cls: CharClass::Georgian },     // Georgian
    Block { end: 0x11FF, cls: CharClass::Hangul },       // Hangul Jamo
    Block { end: 0x137F, cls: CharClass::Ethiopic },     // Ethiopic
    Block { end: 0x139F, cls: CharClass::Ethiopic },     // Ethiopic Supplement
    Block { end: 0x13FF, cls: CharClass::Default },      // Cherokee
    Block { end: 0x167F, cls: CharClass::Default },      // Canadian Aboriginal Syllabics
    Block { end: 0x169F, cls: CharClass::Default },      // Ogham
    Block { end: 0x16FF, cls: CharClass::Default },      // Runic
    Block { end: 0x171F, cls: CharClass::Default },      // Tagalog (Baybayin)
    Block { end: 0x173F, cls: CharClass::Default },      // Hanunoo
    Block { end: 0x175F, cls: CharClass::Default },      // Buhid
    Block { end: 0x177F, cls: CharClass::Default },      // Tagbanwa
    Block { end: 0x17FF, cls: CharClass::KhmerMyanmar }, // Khmer
    Block { end: 0x18AF, cls: CharClass::Default },      // Mongolian
    Block { end: 0x18FF, cls: CharClass::Default },      // Canadian Aboriginal Syllabics Ext
    Block { end: 0x194F, cls: CharClass::Indic },        // Limbu
    Block { end: 0x19DF, cls: CharClass::Indic },        // Tai Le & New Tai Lue
    Block { end: 0x19FF, cls: CharClass::KhmerMyanmar }, // Khmer Symbols
    Block { end: 0x1A1F, cls: CharClass::Indic },        // Buginese
    Block { end: 0x1AAF, cls: CharClass::Indic },        // Tai Tham
    Block { end: 0x1B7F, cls: CharClass::Indic },        // Balinese
    Block { end: 0x1BBF, cls: CharClass::Indic },        // Sundanese
    Block { end: 0x1BFF, cls: CharClass::Indic },        // Batak
    Block { end: 0x1C4F, cls: CharClass::Indic },        // Lepcha
    Block { end: 0x1C7F, cls: CharClass::Indic },        // Ol Chiki (Santali)
    Block { end: 0x1C8F, cls: CharClass::Cyrillic },     // Cyrillic Extended-C
    Block { end: 0x1CBF, cls: CharClass::Georgian },     // Georgian Extended
    Block { end: 0x1CCF, cls: CharClass::Indic },        // Sundanese Supplement
    Block { end: 0x1CFF, cls: CharClass::Indic },        // Vedic Extensions
    Block { end: 0x1D7F, cls: CharClass::Latin },        // Phonetic Extensions
    Block { end: 0x1DBF, cls: CharClass::Latin },        // Phonetic Extensions Supplement
    Block { end: 0x1DFF, cls: CharClass::Default },      // Combining Diacritical Marks Supplement
    Block { end: 0x1EFF, cls: CharClass::Latin },        // Latin Extended Additional (Vietnamese)
    Block { end: 0x309F, cls: CharClass::Kana },         // Hiragana
    Block { end: 0x30FF, cls: CharClass::Kana },         // Katakana
    Block { end: 0x312F, cls: CharClass::Cjk },          // Bopomofo (Pinyin)
    Block { end: 0x318F, cls: CharClass::Hangul },       // Hangul Compatibility Jamo
    Block { end: 0x9FFF, cls: CharClass::Cjk },          // CJK Unified Ideographs (Main)
    Block { end: 0xA4CF, cls: CharClass::Yi },           // Yi Syllables
    Block { end: 0xA4FF, cls: CharClass::Default },      // Lisu
    Block { end: 0xA63F, cls: CharClass::Default },      // Vai
    Block { end: 0xA69F, cls: CharClass::Cyrillic },     // Cyrillic Extended-B
    Block { end: 0xA6FF, cls: CharClass::Default },      // Bamum
    Block { end: 0xA7FF, cls: CharClass::Latin },        // Latin Extended-D
    Block { end: 0xA82F, cls: CharClass::Indic },        // Syloti Nagri
    Block { end: 0xA87F, cls: CharClass::Default },      // Phags-pa
    Block { end: 0xA8DF, cls: CharClass::Indic },        // Saurashtra
    Block { end: 0xA8FF, cls: CharClass::Indic },        // Devanagari Extended
    Block { end: 0xA92F, cls: CharClass::Indic },        // Kayah Li
    Block { end: 0xA95F, cls: CharClass::Indic },        // Rejang
    Block { end: 0xA97F, cls: CharClass::Hangul },       // Hangul Jamo Extended-A
    Block { end: 0xA9DF, cls: CharClass::Indic },        // Javanese
    Block { end: 0xA9FF, cls: CharClass::KhmerMyanmar }, // Myanmar Extended-B
    Block { end: 0xAA5F, cls: CharClass::Indic },        // Cham
    Block { end: 0xAA7F, cls: CharClass::KhmerMyanmar }, // Myanmar Extended-A
    Block { end: 0xAADF, cls: CharClass::Indic },        // Tai Viet
    Block { end: 0xAAFF, cls: CharClass::Indic },        // Meetei Mayek Extensions
    Block { end: 0xAB2F, cls: CharClass::Ethiopic },     // Ethiopic Extended-A
    Block { end: 0xAB6F, cls: CharClass::Latin },        // Latin Extended-E
    Block { end: 0xABBF, cls: CharClass::Default },      // Cherokee Supplement
    Block { end: 0xABFF, cls: CharClass::Indic },        // Meetei Mayek
    Block { end: 0xD7AF, cls: CharClass::Hangul },       // Hangul Syllables
    Block { end: 0xFAFF, cls: CharClass::Cjk },          // CJK Compatibility
    Block { end: 0xFDFF, cls: CharClass::Arabic },       // Arabic Presentation Forms-A
    Block { end: 0xFE6F, cls: CharClass::Default },      // Variation Selectors
    Block { end: 0xFEFF, cls: CharClass::Arabic },       // Arabic Presentation Forms-B
    Block { end: 0xFFEF, cls: CharClass::Latin },        // Fullwidth Latin
];


// =============================================================================
// Unicode general categories
// =============================================================================
//
// The four category prefixes that carry their own weight, as sorted,
// non-overlapping code point ranges. `P` here is P* *and* S*, since the
// reference gives punctuation and symbols the same 0.5 -- which is what puts
// emoji at half a Latin letter.
//
// Generated from Unicode 16.0.0, the version Python 3.14's `unicodedata`
// exposes, by merging adjacent code points that share a prefix:
//
//   for cp in range(0x110000): unicodedata.category(chr(cp))[0]
//
// Categories drift between Unicode versions only at the edges of newly
// assigned blocks, and a character this table misses falls through to its
// script block rather than to nothing.

const M: CharClass = CharClass::Mark;
const P: CharClass = CharClass::Punctuation;
const Z: CharClass = CharClass::Space;
const N: CharClass = CharClass::Digit;

struct CatRange {
    lo: u32,
    hi: u32,
    cls: CharClass,
}

const fn r(lo: u32, hi: u32, cls: CharClass) -> CatRange {
    CatRange { lo, hi, cls }
}

const CATEGORIES: [CatRange; 822] = [
    r(0x0020, 0x0020, Z), r(0x0021, 0x002F, P), r(0x0030, 0x0039, N), r(0x003A, 0x0040, P),
    r(0x005B, 0x0060, P), r(0x007B, 0x007E, P), r(0x00A0, 0x00A0, Z), r(0x00A1, 0x00A9, P),
    r(0x00AB, 0x00AC, P), r(0x00AE, 0x00B1, P), r(0x00B2, 0x00B3, N), r(0x00B4, 0x00B4, P),
    r(0x00B6, 0x00B8, P), r(0x00B9, 0x00B9, N), r(0x00BB, 0x00BB, P), r(0x00BC, 0x00BE, N),
    r(0x00BF, 0x00BF, P), r(0x00D7, 0x00D7, P), r(0x00F7, 0x00F7, P), r(0x02C2, 0x02C5, P),
    r(0x02D2, 0x02DF, P), r(0x02E5, 0x02EB, P), r(0x02ED, 0x02ED, P), r(0x02EF, 0x02FF, P),
    r(0x0300, 0x036F, M), r(0x0375, 0x0375, P), r(0x037E, 0x037E, P), r(0x0384, 0x0385, P),
    r(0x0387, 0x0387, P), r(0x03F6, 0x03F6, P), r(0x0482, 0x0482, P), r(0x0483, 0x0489, M),
    r(0x055A, 0x055F, P), r(0x0589, 0x058A, P), r(0x058D, 0x058F, P), r(0x0591, 0x05BD, M),
    r(0x05BE, 0x05BE, P), r(0x05BF, 0x05BF, M), r(0x05C0, 0x05C0, P), r(0x05C1, 0x05C2, M),
    r(0x05C3, 0x05C3, P), r(0x05C4, 0x05C5, M), r(0x05C6, 0x05C6, P), r(0x05C7, 0x05C7, M),
    r(0x05F3, 0x05F4, P), r(0x0606, 0x060F, P), r(0x0610, 0x061A, M), r(0x061B, 0x061B, P),
    r(0x061D, 0x061F, P), r(0x064B, 0x065F, M), r(0x0660, 0x0669, N), r(0x066A, 0x066D, P),
    r(0x0670, 0x0670, M), r(0x06D4, 0x06D4, P), r(0x06D6, 0x06DC, M), r(0x06DE, 0x06DE, P),
    r(0x06DF, 0x06E4, M), r(0x06E7, 0x06E8, M), r(0x06E9, 0x06E9, P), r(0x06EA, 0x06ED, M),
    r(0x06F0, 0x06F9, N), r(0x06FD, 0x06FE, P), r(0x0700, 0x070D, P), r(0x0711, 0x0711, M),
    r(0x0730, 0x074A, M), r(0x07A6, 0x07B0, M), r(0x07C0, 0x07C9, N), r(0x07EB, 0x07F3, M),
    r(0x07F6, 0x07F9, P), r(0x07FD, 0x07FD, M), r(0x07FE, 0x07FF, P), r(0x0816, 0x0819, M),
    r(0x081B, 0x0823, M), r(0x0825, 0x0827, M), r(0x0829, 0x082D, M), r(0x0830, 0x083E, P),
    r(0x0859, 0x085B, M), r(0x085E, 0x085E, P), r(0x0888, 0x0888, P), r(0x0897, 0x089F, M),
    r(0x08CA, 0x08E1, M), r(0x08E3, 0x0903, M), r(0x093A, 0x093C, M), r(0x093E, 0x094F, M),
    r(0x0951, 0x0957, M), r(0x0962, 0x0963, M), r(0x0964, 0x0965, P), r(0x0966, 0x096F, N),
    r(0x0970, 0x0970, P), r(0x0981, 0x0983, M), r(0x09BC, 0x09BC, M), r(0x09BE, 0x09C4, M),
    r(0x09C7, 0x09C8, M), r(0x09CB, 0x09CD, M), r(0x09D7, 0x09D7, M), r(0x09E2, 0x09E3, M),
    r(0x09E6, 0x09EF, N), r(0x09F2, 0x09F3, P), r(0x09F4, 0x09F9, N), r(0x09FA, 0x09FB, P),
    r(0x09FD, 0x09FD, P), r(0x09FE, 0x09FE, M), r(0x0A01, 0x0A03, M), r(0x0A3C, 0x0A3C, M),
    r(0x0A3E, 0x0A42, M), r(0x0A47, 0x0A48, M), r(0x0A4B, 0x0A4D, M), r(0x0A51, 0x0A51, M),
    r(0x0A66, 0x0A6F, N), r(0x0A70, 0x0A71, M), r(0x0A75, 0x0A75, M), r(0x0A76, 0x0A76, P),
    r(0x0A81, 0x0A83, M), r(0x0ABC, 0x0ABC, M), r(0x0ABE, 0x0AC5, M), r(0x0AC7, 0x0AC9, M),
    r(0x0ACB, 0x0ACD, M), r(0x0AE2, 0x0AE3, M), r(0x0AE6, 0x0AEF, N), r(0x0AF0, 0x0AF1, P),
    r(0x0AFA, 0x0AFF, M), r(0x0B01, 0x0B03, M), r(0x0B3C, 0x0B3C, M), r(0x0B3E, 0x0B44, M),
    r(0x0B47, 0x0B48, M), r(0x0B4B, 0x0B4D, M), r(0x0B55, 0x0B57, M), r(0x0B62, 0x0B63, M),
    r(0x0B66, 0x0B6F, N), r(0x0B70, 0x0B70, P), r(0x0B72, 0x0B77, N), r(0x0B82, 0x0B82, M),
    r(0x0BBE, 0x0BC2, M), r(0x0BC6, 0x0BC8, M), r(0x0BCA, 0x0BCD, M), r(0x0BD7, 0x0BD7, M),
    r(0x0BE6, 0x0BF2, N), r(0x0BF3, 0x0BFA, P), r(0x0C00, 0x0C04, M), r(0x0C3C, 0x0C3C, M),
    r(0x0C3E, 0x0C44, M), r(0x0C46, 0x0C48, M), r(0x0C4A, 0x0C4D, M), r(0x0C55, 0x0C56, M),
    r(0x0C62, 0x0C63, M), r(0x0C66, 0x0C6F, N), r(0x0C77, 0x0C77, P), r(0x0C78, 0x0C7E, N),
    r(0x0C7F, 0x0C7F, P), r(0x0C81, 0x0C83, M), r(0x0C84, 0x0C84, P), r(0x0CBC, 0x0CBC, M),
    r(0x0CBE, 0x0CC4, M), r(0x0CC6, 0x0CC8, M), r(0x0CCA, 0x0CCD, M), r(0x0CD5, 0x0CD6, M),
    r(0x0CE2, 0x0CE3, M), r(0x0CE6, 0x0CEF, N), r(0x0CF3, 0x0CF3, M), r(0x0D00, 0x0D03, M),
    r(0x0D3B, 0x0D3C, M), r(0x0D3E, 0x0D44, M), r(0x0D46, 0x0D48, M), r(0x0D4A, 0x0D4D, M),
    r(0x0D4F, 0x0D4F, P), r(0x0D57, 0x0D57, M), r(0x0D58, 0x0D5E, N), r(0x0D62, 0x0D63, M),
    r(0x0D66, 0x0D78, N), r(0x0D79, 0x0D79, P), r(0x0D81, 0x0D83, M), r(0x0DCA, 0x0DCA, M),
    r(0x0DCF, 0x0DD4, M), r(0x0DD6, 0x0DD6, M), r(0x0DD8, 0x0DDF, M), r(0x0DE6, 0x0DEF, N),
    r(0x0DF2, 0x0DF3, M), r(0x0DF4, 0x0DF4, P), r(0x0E31, 0x0E31, M), r(0x0E34, 0x0E3A, M),
    r(0x0E3F, 0x0E3F, P), r(0x0E47, 0x0E4E, M), r(0x0E4F, 0x0E4F, P), r(0x0E50, 0x0E59, N),
    r(0x0E5A, 0x0E5B, P), r(0x0EB1, 0x0EB1, M), r(0x0EB4, 0x0EBC, M), r(0x0EC8, 0x0ECE, M),
    r(0x0ED0, 0x0ED9, N), r(0x0F01, 0x0F17, P), r(0x0F18, 0x0F19, M), r(0x0F1A, 0x0F1F, P),
    r(0x0F20, 0x0F33, N), r(0x0F34, 0x0F34, P), r(0x0F35, 0x0F35, M), r(0x0F36, 0x0F36, P),
    r(0x0F37, 0x0F37, M), r(0x0F38, 0x0F38, P), r(0x0F39, 0x0F39, M), r(0x0F3A, 0x0F3D, P),
    r(0x0F3E, 0x0F3F, M), r(0x0F71, 0x0F84, M), r(0x0F85, 0x0F85, P), r(0x0F86, 0x0F87, M),
    r(0x0F8D, 0x0F97, M), r(0x0F99, 0x0FBC, M), r(0x0FBE, 0x0FC5, P), r(0x0FC6, 0x0FC6, M),
    r(0x0FC7, 0x0FCC, P), r(0x0FCE, 0x0FDA, P), r(0x102B, 0x103E, M), r(0x1040, 0x1049, N),
    r(0x104A, 0x104F, P), r(0x1056, 0x1059, M), r(0x105E, 0x1060, M), r(0x1062, 0x1064, M),
    r(0x1067, 0x106D, M), r(0x1071, 0x1074, M), r(0x1082, 0x108D, M), r(0x108F, 0x108F, M),
    r(0x1090, 0x1099, N), r(0x109A, 0x109D, M), r(0x109E, 0x109F, P), r(0x10FB, 0x10FB, P),
    r(0x135D, 0x135F, M), r(0x1360, 0x1368, P), r(0x1369, 0x137C, N), r(0x1390, 0x1399, P),
    r(0x1400, 0x1400, P), r(0x166D, 0x166E, P), r(0x1680, 0x1680, Z), r(0x169B, 0x169C, P),
    r(0x16EB, 0x16ED, P), r(0x16EE, 0x16F0, N), r(0x1712, 0x1715, M), r(0x1732, 0x1734, M),
    r(0x1735, 0x1736, P), r(0x1752, 0x1753, M), r(0x1772, 0x1773, M), r(0x17B4, 0x17D3, M),
    r(0x17D4, 0x17D6, P), r(0x17D8, 0x17DB, P), r(0x17DD, 0x17DD, M), r(0x17E0, 0x17E9, N),
    r(0x17F0, 0x17F9, N), r(0x1800, 0x180A, P), r(0x180B, 0x180D, M), r(0x180F, 0x180F, M),
    r(0x1810, 0x1819, N), r(0x1885, 0x1886, M), r(0x18A9, 0x18A9, M), r(0x1920, 0x192B, M),
    r(0x1930, 0x193B, M), r(0x1940, 0x1940, P), r(0x1944, 0x1945, P), r(0x1946, 0x194F, N),
    r(0x19D0, 0x19DA, N), r(0x19DE, 0x19FF, P), r(0x1A17, 0x1A1B, M), r(0x1A1E, 0x1A1F, P),
    r(0x1A55, 0x1A5E, M), r(0x1A60, 0x1A7C, M), r(0x1A7F, 0x1A7F, M), r(0x1A80, 0x1A89, N),
    r(0x1A90, 0x1A99, N), r(0x1AA0, 0x1AA6, P), r(0x1AA8, 0x1AAD, P), r(0x1AB0, 0x1ACE, M),
    r(0x1B00, 0x1B04, M), r(0x1B34, 0x1B44, M), r(0x1B4E, 0x1B4F, P), r(0x1B50, 0x1B59, N),
    r(0x1B5A, 0x1B6A, P), r(0x1B6B, 0x1B73, M), r(0x1B74, 0x1B7F, P), r(0x1B80, 0x1B82, M),
    r(0x1BA1, 0x1BAD, M), r(0x1BB0, 0x1BB9, N), r(0x1BE6, 0x1BF3, M), r(0x1BFC, 0x1BFF, P),
    r(0x1C24, 0x1C37, M), r(0x1C3B, 0x1C3F, P), r(0x1C40, 0x1C49, N), r(0x1C50, 0x1C59, N),
    r(0x1C7E, 0x1C7F, P), r(0x1CC0, 0x1CC7, P), r(0x1CD0, 0x1CD2, M), r(0x1CD3, 0x1CD3, P),
    r(0x1CD4, 0x1CE8, M), r(0x1CED, 0x1CED, M), r(0x1CF4, 0x1CF4, M), r(0x1CF7, 0x1CF9, M),
    r(0x1DC0, 0x1DFF, M), r(0x1FBD, 0x1FBD, P), r(0x1FBF, 0x1FC1, P), r(0x1FCD, 0x1FCF, P),
    r(0x1FDD, 0x1FDF, P), r(0x1FED, 0x1FEF, P), r(0x1FFD, 0x1FFE, P), r(0x2000, 0x200A, Z),
    r(0x2010, 0x2027, P), r(0x2028, 0x2029, Z), r(0x202F, 0x202F, Z), r(0x2030, 0x205E, P),
    r(0x205F, 0x205F, Z), r(0x2070, 0x2070, N), r(0x2074, 0x2079, N), r(0x207A, 0x207E, P),
    r(0x2080, 0x2089, N), r(0x208A, 0x208E, P), r(0x20A0, 0x20C0, P), r(0x20D0, 0x20F0, M),
    r(0x2100, 0x2101, P), r(0x2103, 0x2106, P), r(0x2108, 0x2109, P), r(0x2114, 0x2114, P),
    r(0x2116, 0x2118, P), r(0x211E, 0x2123, P), r(0x2125, 0x2125, P), r(0x2127, 0x2127, P),
    r(0x2129, 0x2129, P), r(0x212E, 0x212E, P), r(0x213A, 0x213B, P), r(0x2140, 0x2144, P),
    r(0x214A, 0x214D, P), r(0x214F, 0x214F, P), r(0x2150, 0x2182, N), r(0x2185, 0x2189, N),
    r(0x218A, 0x218B, P), r(0x2190, 0x2429, P), r(0x2440, 0x244A, P), r(0x2460, 0x249B, N),
    r(0x249C, 0x24E9, P), r(0x24EA, 0x24FF, N), r(0x2500, 0x2775, P), r(0x2776, 0x2793, N),
    r(0x2794, 0x2B73, P), r(0x2B76, 0x2B95, P), r(0x2B97, 0x2BFF, P), r(0x2CE5, 0x2CEA, P),
    r(0x2CEF, 0x2CF1, M), r(0x2CF9, 0x2CFC, P), r(0x2CFD, 0x2CFD, N), r(0x2CFE, 0x2CFF, P),
    r(0x2D70, 0x2D70, P), r(0x2D7F, 0x2D7F, M), r(0x2DE0, 0x2DFF, M), r(0x2E00, 0x2E2E, P),
    r(0x2E30, 0x2E5D, P), r(0x2E80, 0x2E99, P), r(0x2E9B, 0x2EF3, P), r(0x2F00, 0x2FD5, P),
    r(0x2FF0, 0x2FFF, P), r(0x3000, 0x3000, Z), r(0x3001, 0x3004, P), r(0x3007, 0x3007, N),
    r(0x3008, 0x3020, P), r(0x3021, 0x3029, N), r(0x302A, 0x302F, M), r(0x3030, 0x3030, P),
    r(0x3036, 0x3037, P), r(0x3038, 0x303A, N), r(0x303D, 0x303F, P), r(0x3099, 0x309A, M),
    r(0x309B, 0x309C, P), r(0x30A0, 0x30A0, P), r(0x30FB, 0x30FB, P), r(0x3190, 0x3191, P),
    r(0x3192, 0x3195, N), r(0x3196, 0x319F, P), r(0x31C0, 0x31E5, P), r(0x31EF, 0x31EF, P),
    r(0x3200, 0x321E, P), r(0x3220, 0x3229, N), r(0x322A, 0x3247, P), r(0x3248, 0x324F, N),
    r(0x3250, 0x3250, P), r(0x3251, 0x325F, N), r(0x3260, 0x327F, P), r(0x3280, 0x3289, N),
    r(0x328A, 0x32B0, P), r(0x32B1, 0x32BF, N), r(0x32C0, 0x33FF, P), r(0x4DC0, 0x4DFF, P),
    r(0xA490, 0xA4C6, P), r(0xA4FE, 0xA4FF, P), r(0xA60D, 0xA60F, P), r(0xA620, 0xA629, N),
    r(0xA66F, 0xA672, M), r(0xA673, 0xA673, P), r(0xA674, 0xA67D, M), r(0xA67E, 0xA67E, P),
    r(0xA69E, 0xA69F, M), r(0xA6E6, 0xA6EF, N), r(0xA6F0, 0xA6F1, M), r(0xA6F2, 0xA6F7, P),
    r(0xA700, 0xA716, P), r(0xA720, 0xA721, P), r(0xA789, 0xA78A, P), r(0xA802, 0xA802, M),
    r(0xA806, 0xA806, M), r(0xA80B, 0xA80B, M), r(0xA823, 0xA827, M), r(0xA828, 0xA82B, P),
    r(0xA82C, 0xA82C, M), r(0xA830, 0xA835, N), r(0xA836, 0xA839, P), r(0xA874, 0xA877, P),
    r(0xA880, 0xA881, M), r(0xA8B4, 0xA8C5, M), r(0xA8CE, 0xA8CF, P), r(0xA8D0, 0xA8D9, N),
    r(0xA8E0, 0xA8F1, M), r(0xA8F8, 0xA8FA, P), r(0xA8FC, 0xA8FC, P), r(0xA8FF, 0xA8FF, M),
    r(0xA900, 0xA909, N), r(0xA926, 0xA92D, M), r(0xA92E, 0xA92F, P), r(0xA947, 0xA953, M),
    r(0xA95F, 0xA95F, P), r(0xA980, 0xA983, M), r(0xA9B3, 0xA9C0, M), r(0xA9C1, 0xA9CD, P),
    r(0xA9D0, 0xA9D9, N), r(0xA9DE, 0xA9DF, P), r(0xA9E5, 0xA9E5, M), r(0xA9F0, 0xA9F9, N),
    r(0xAA29, 0xAA36, M), r(0xAA43, 0xAA43, M), r(0xAA4C, 0xAA4D, M), r(0xAA50, 0xAA59, N),
    r(0xAA5C, 0xAA5F, P), r(0xAA77, 0xAA79, P), r(0xAA7B, 0xAA7D, M), r(0xAAB0, 0xAAB0, M),
    r(0xAAB2, 0xAAB4, M), r(0xAAB7, 0xAAB8, M), r(0xAABE, 0xAABF, M), r(0xAAC1, 0xAAC1, M),
    r(0xAADE, 0xAADF, P), r(0xAAEB, 0xAAEF, M), r(0xAAF0, 0xAAF1, P), r(0xAAF5, 0xAAF6, M),
    r(0xAB5B, 0xAB5B, P), r(0xAB6A, 0xAB6B, P), r(0xABE3, 0xABEA, M), r(0xABEB, 0xABEB, P),
    r(0xABEC, 0xABED, M), r(0xABF0, 0xABF9, N), r(0xFB1E, 0xFB1E, M), r(0xFB29, 0xFB29, P),
    r(0xFBB2, 0xFBC2, P), r(0xFD3E, 0xFD4F, P), r(0xFDCF, 0xFDCF, P), r(0xFDFC, 0xFDFF, P),
    r(0xFE00, 0xFE0F, M), r(0xFE10, 0xFE19, P), r(0xFE20, 0xFE2F, M), r(0xFE30, 0xFE52, P),
    r(0xFE54, 0xFE66, P), r(0xFE68, 0xFE6B, P), r(0xFF01, 0xFF0F, P), r(0xFF10, 0xFF19, N),
    r(0xFF1A, 0xFF20, P), r(0xFF3B, 0xFF40, P), r(0xFF5B, 0xFF65, P), r(0xFFE0, 0xFFE6, P),
    r(0xFFE8, 0xFFEE, P), r(0xFFFC, 0xFFFD, P), r(0x10100, 0x10102, P), r(0x10107, 0x10133, N),
    r(0x10137, 0x1013F, P), r(0x10140, 0x10178, N), r(0x10179, 0x10189, P), r(0x1018A, 0x1018B, N),
    r(0x1018C, 0x1018E, P), r(0x10190, 0x1019C, P), r(0x101A0, 0x101A0, P), r(0x101D0, 0x101FC, P),
    r(0x101FD, 0x101FD, M), r(0x102E0, 0x102E0, M), r(0x102E1, 0x102FB, N), r(0x10320, 0x10323, N),
    r(0x10341, 0x10341, N), r(0x1034A, 0x1034A, N), r(0x10376, 0x1037A, M), r(0x1039F, 0x1039F, P),
    r(0x103D0, 0x103D0, P), r(0x103D1, 0x103D5, N), r(0x104A0, 0x104A9, N), r(0x1056F, 0x1056F, P),
    r(0x10857, 0x10857, P), r(0x10858, 0x1085F, N), r(0x10877, 0x10878, P), r(0x10879, 0x1087F, N),
    r(0x108A7, 0x108AF, N), r(0x108FB, 0x108FF, N), r(0x10916, 0x1091B, N), r(0x1091F, 0x1091F, P),
    r(0x1093F, 0x1093F, P), r(0x109BC, 0x109BD, N), r(0x109C0, 0x109CF, N), r(0x109D2, 0x109FF, N),
    r(0x10A01, 0x10A03, M), r(0x10A05, 0x10A06, M), r(0x10A0C, 0x10A0F, M), r(0x10A38, 0x10A3A, M),
    r(0x10A3F, 0x10A3F, M), r(0x10A40, 0x10A48, N), r(0x10A50, 0x10A58, P), r(0x10A7D, 0x10A7E, N),
    r(0x10A7F, 0x10A7F, P), r(0x10A9D, 0x10A9F, N), r(0x10AC8, 0x10AC8, P), r(0x10AE5, 0x10AE6, M),
    r(0x10AEB, 0x10AEF, N), r(0x10AF0, 0x10AF6, P), r(0x10B39, 0x10B3F, P), r(0x10B58, 0x10B5F, N),
    r(0x10B78, 0x10B7F, N), r(0x10B99, 0x10B9C, P), r(0x10BA9, 0x10BAF, N), r(0x10CFA, 0x10CFF, N),
    r(0x10D24, 0x10D27, M), r(0x10D30, 0x10D39, N), r(0x10D40, 0x10D49, N), r(0x10D69, 0x10D6D, M),
    r(0x10D6E, 0x10D6E, P), r(0x10D8E, 0x10D8F, P), r(0x10E60, 0x10E7E, N), r(0x10EAB, 0x10EAC, M),
    r(0x10EAD, 0x10EAD, P), r(0x10EFC, 0x10EFF, M), r(0x10F1D, 0x10F26, N), r(0x10F46, 0x10F50, M),
    r(0x10F51, 0x10F54, N), r(0x10F55, 0x10F59, P), r(0x10F82, 0x10F85, M), r(0x10F86, 0x10F89, P),
    r(0x10FC5, 0x10FCB, N), r(0x11000, 0x11002, M), r(0x11038, 0x11046, M), r(0x11047, 0x1104D, P),
    r(0x11052, 0x1106F, N), r(0x11070, 0x11070, M), r(0x11073, 0x11074, M), r(0x1107F, 0x11082, M),
    r(0x110B0, 0x110BA, M), r(0x110BB, 0x110BC, P), r(0x110BE, 0x110C1, P), r(0x110C2, 0x110C2, M),
    r(0x110F0, 0x110F9, N), r(0x11100, 0x11102, M), r(0x11127, 0x11134, M), r(0x11136, 0x1113F, N),
    r(0x11140, 0x11143, P), r(0x11145, 0x11146, M), r(0x11173, 0x11173, M), r(0x11174, 0x11175, P),
    r(0x11180, 0x11182, M), r(0x111B3, 0x111C0, M), r(0x111C5, 0x111C8, P), r(0x111C9, 0x111CC, M),
    r(0x111CD, 0x111CD, P), r(0x111CE, 0x111CF, M), r(0x111D0, 0x111D9, N), r(0x111DB, 0x111DB, P),
    r(0x111DD, 0x111DF, P), r(0x111E1, 0x111F4, N), r(0x1122C, 0x11237, M), r(0x11238, 0x1123D, P),
    r(0x1123E, 0x1123E, M), r(0x11241, 0x11241, M), r(0x112A9, 0x112A9, P), r(0x112DF, 0x112EA, M),
    r(0x112F0, 0x112F9, N), r(0x11300, 0x11303, M), r(0x1133B, 0x1133C, M), r(0x1133E, 0x11344, M),
    r(0x11347, 0x11348, M), r(0x1134B, 0x1134D, M), r(0x11357, 0x11357, M), r(0x11362, 0x11363, M),
    r(0x11366, 0x1136C, M), r(0x11370, 0x11374, M), r(0x113B8, 0x113C0, M), r(0x113C2, 0x113C2, M),
    r(0x113C5, 0x113C5, M), r(0x113C7, 0x113CA, M), r(0x113CC, 0x113D0, M), r(0x113D2, 0x113D2, M),
    r(0x113D4, 0x113D5, P), r(0x113D7, 0x113D8, P), r(0x113E1, 0x113E2, M), r(0x11435, 0x11446, M),
    r(0x1144B, 0x1144F, P), r(0x11450, 0x11459, N), r(0x1145A, 0x1145B, P), r(0x1145D, 0x1145D, P),
    r(0x1145E, 0x1145E, M), r(0x114B0, 0x114C3, M), r(0x114C6, 0x114C6, P), r(0x114D0, 0x114D9, N),
    r(0x115AF, 0x115B5, M), r(0x115B8, 0x115C0, M), r(0x115C1, 0x115D7, P), r(0x115DC, 0x115DD, M),
    r(0x11630, 0x11640, M), r(0x11641, 0x11643, P), r(0x11650, 0x11659, N), r(0x11660, 0x1166C, P),
    r(0x116AB, 0x116B7, M), r(0x116B9, 0x116B9, P), r(0x116C0, 0x116C9, N), r(0x116D0, 0x116E3, N),
    r(0x1171D, 0x1172B, M), r(0x11730, 0x1173B, N), r(0x1173C, 0x1173F, P), r(0x1182C, 0x1183A, M),
    r(0x1183B, 0x1183B, P), r(0x118E0, 0x118F2, N), r(0x11930, 0x11935, M), r(0x11937, 0x11938, M),
    r(0x1193B, 0x1193E, M), r(0x11940, 0x11940, M), r(0x11942, 0x11943, M), r(0x11944, 0x11946, P),
    r(0x11950, 0x11959, N), r(0x119D1, 0x119D7, M), r(0x119DA, 0x119E0, M), r(0x119E2, 0x119E2, P),
    r(0x119E4, 0x119E4, M), r(0x11A01, 0x11A0A, M), r(0x11A33, 0x11A39, M), r(0x11A3B, 0x11A3E, M),
    r(0x11A3F, 0x11A46, P), r(0x11A47, 0x11A47, M), r(0x11A51, 0x11A5B, M), r(0x11A8A, 0x11A99, M),
    r(0x11A9A, 0x11A9C, P), r(0x11A9E, 0x11AA2, P), r(0x11B00, 0x11B09, P), r(0x11BE1, 0x11BE1, P),
    r(0x11BF0, 0x11BF9, N), r(0x11C2F, 0x11C36, M), r(0x11C38, 0x11C3F, M), r(0x11C41, 0x11C45, P),
    r(0x11C50, 0x11C6C, N), r(0x11C70, 0x11C71, P), r(0x11C92, 0x11CA7, M), r(0x11CA9, 0x11CB6, M),
    r(0x11D31, 0x11D36, M), r(0x11D3A, 0x11D3A, M), r(0x11D3C, 0x11D3D, M), r(0x11D3F, 0x11D45, M),
    r(0x11D47, 0x11D47, M), r(0x11D50, 0x11D59, N), r(0x11D8A, 0x11D8E, M), r(0x11D90, 0x11D91, M),
    r(0x11D93, 0x11D97, M), r(0x11DA0, 0x11DA9, N), r(0x11EF3, 0x11EF6, M), r(0x11EF7, 0x11EF8, P),
    r(0x11F00, 0x11F01, M), r(0x11F03, 0x11F03, M), r(0x11F34, 0x11F3A, M), r(0x11F3E, 0x11F42, M),
    r(0x11F43, 0x11F4F, P), r(0x11F50, 0x11F59, N), r(0x11F5A, 0x11F5A, M), r(0x11FC0, 0x11FD4, N),
    r(0x11FD5, 0x11FF1, P), r(0x11FFF, 0x11FFF, P), r(0x12400, 0x1246E, N), r(0x12470, 0x12474, P),
    r(0x12FF1, 0x12FF2, P), r(0x13440, 0x13440, M), r(0x13447, 0x13455, M), r(0x1611E, 0x1612F, M),
    r(0x16130, 0x16139, N), r(0x16A60, 0x16A69, N), r(0x16A6E, 0x16A6F, P), r(0x16AC0, 0x16AC9, N),
    r(0x16AF0, 0x16AF4, M), r(0x16AF5, 0x16AF5, P), r(0x16B30, 0x16B36, M), r(0x16B37, 0x16B3F, P),
    r(0x16B44, 0x16B45, P), r(0x16B50, 0x16B59, N), r(0x16B5B, 0x16B61, N), r(0x16D6D, 0x16D6F, P),
    r(0x16D70, 0x16D79, N), r(0x16E80, 0x16E96, N), r(0x16E97, 0x16E9A, P), r(0x16F4F, 0x16F4F, M),
    r(0x16F51, 0x16F87, M), r(0x16F8F, 0x16F92, M), r(0x16FE2, 0x16FE2, P), r(0x16FE4, 0x16FE4, M),
    r(0x16FF0, 0x16FF1, M), r(0x1BC9C, 0x1BC9C, P), r(0x1BC9D, 0x1BC9E, M), r(0x1BC9F, 0x1BC9F, P),
    r(0x1CC00, 0x1CCEF, P), r(0x1CCF0, 0x1CCF9, N), r(0x1CD00, 0x1CEB3, P), r(0x1CF00, 0x1CF2D, M),
    r(0x1CF30, 0x1CF46, M), r(0x1CF50, 0x1CFC3, P), r(0x1D000, 0x1D0F5, P), r(0x1D100, 0x1D126, P),
    r(0x1D129, 0x1D164, P), r(0x1D165, 0x1D169, M), r(0x1D16A, 0x1D16C, P), r(0x1D16D, 0x1D172, M),
    r(0x1D17B, 0x1D182, M), r(0x1D183, 0x1D184, P), r(0x1D185, 0x1D18B, M), r(0x1D18C, 0x1D1A9, P),
    r(0x1D1AA, 0x1D1AD, M), r(0x1D1AE, 0x1D1EA, P), r(0x1D200, 0x1D241, P), r(0x1D242, 0x1D244, M),
    r(0x1D245, 0x1D245, P), r(0x1D2C0, 0x1D2D3, N), r(0x1D2E0, 0x1D2F3, N), r(0x1D300, 0x1D356, P),
    r(0x1D360, 0x1D378, N), r(0x1D6C1, 0x1D6C1, P), r(0x1D6DB, 0x1D6DB, P), r(0x1D6FB, 0x1D6FB, P),
    r(0x1D715, 0x1D715, P), r(0x1D735, 0x1D735, P), r(0x1D74F, 0x1D74F, P), r(0x1D76F, 0x1D76F, P),
    r(0x1D789, 0x1D789, P), r(0x1D7A9, 0x1D7A9, P), r(0x1D7C3, 0x1D7C3, P), r(0x1D7CE, 0x1D7FF, N),
    r(0x1D800, 0x1D9FF, P), r(0x1DA00, 0x1DA36, M), r(0x1DA37, 0x1DA3A, P), r(0x1DA3B, 0x1DA6C, M),
    r(0x1DA6D, 0x1DA74, P), r(0x1DA75, 0x1DA75, M), r(0x1DA76, 0x1DA83, P), r(0x1DA84, 0x1DA84, M),
    r(0x1DA85, 0x1DA8B, P), r(0x1DA9B, 0x1DA9F, M), r(0x1DAA1, 0x1DAAF, M), r(0x1E000, 0x1E006, M),
    r(0x1E008, 0x1E018, M), r(0x1E01B, 0x1E021, M), r(0x1E023, 0x1E024, M), r(0x1E026, 0x1E02A, M),
    r(0x1E08F, 0x1E08F, M), r(0x1E130, 0x1E136, M), r(0x1E140, 0x1E149, N), r(0x1E14F, 0x1E14F, P),
    r(0x1E2AE, 0x1E2AE, M), r(0x1E2EC, 0x1E2EF, M), r(0x1E2F0, 0x1E2F9, N), r(0x1E2FF, 0x1E2FF, P),
    r(0x1E4EC, 0x1E4EF, M), r(0x1E4F0, 0x1E4F9, N), r(0x1E5EE, 0x1E5EF, M), r(0x1E5F1, 0x1E5FA, N),
    r(0x1E5FF, 0x1E5FF, P), r(0x1E8C7, 0x1E8CF, N), r(0x1E8D0, 0x1E8D6, M), r(0x1E944, 0x1E94A, M),
    r(0x1E950, 0x1E959, N), r(0x1E95E, 0x1E95F, P), r(0x1EC71, 0x1ECAB, N), r(0x1ECAC, 0x1ECAC, P),
    r(0x1ECAD, 0x1ECAF, N), r(0x1ECB0, 0x1ECB0, P), r(0x1ECB1, 0x1ECB4, N), r(0x1ED01, 0x1ED2D, N),
    r(0x1ED2E, 0x1ED2E, P), r(0x1ED2F, 0x1ED3D, N), r(0x1EEF0, 0x1EEF1, P), r(0x1F000, 0x1F02B, P),
    r(0x1F030, 0x1F093, P), r(0x1F0A0, 0x1F0AE, P), r(0x1F0B1, 0x1F0BF, P), r(0x1F0C1, 0x1F0CF, P),
    r(0x1F0D1, 0x1F0F5, P), r(0x1F100, 0x1F10C, N), r(0x1F10D, 0x1F1AD, P), r(0x1F1E6, 0x1F202, P),
    r(0x1F210, 0x1F23B, P), r(0x1F240, 0x1F248, P), r(0x1F250, 0x1F251, P), r(0x1F260, 0x1F265, P),
    r(0x1F300, 0x1F6D7, P), r(0x1F6DC, 0x1F6EC, P), r(0x1F6F0, 0x1F6FC, P), r(0x1F700, 0x1F776, P),
    r(0x1F77B, 0x1F7D9, P), r(0x1F7E0, 0x1F7EB, P), r(0x1F7F0, 0x1F7F0, P), r(0x1F800, 0x1F80B, P),
    r(0x1F810, 0x1F847, P), r(0x1F850, 0x1F859, P), r(0x1F860, 0x1F887, P), r(0x1F890, 0x1F8AD, P),
    r(0x1F8B0, 0x1F8BB, P), r(0x1F8C0, 0x1F8C1, P), r(0x1F900, 0x1FA53, P), r(0x1FA60, 0x1FA6D, P),
    r(0x1FA70, 0x1FA7C, P), r(0x1FA80, 0x1FA89, P), r(0x1FA8F, 0x1FAC6, P), r(0x1FACE, 0x1FADC, P),
    r(0x1FADF, 0x1FAE9, P), r(0x1FAF0, 0x1FAF8, P), r(0x1FB00, 0x1FB92, P), r(0x1FB94, 0x1FBEF, P),
    r(0x1FBF0, 0x1FBF9, N), r(0xE0100, 0xE01EF, M),
];

// =============================================================================
// Classification
// =============================================================================

/// The speaking weight of a class, relative to one Latin letter.
pub fn class_weight(c: CharClass) -> f32 {
    match c {
        CharClass::Cjk | CharClass::Ethiopic | CharClass::Yi => 3.0,
        CharClass::Hangul => 2.5,
        CharClass::Kana => 2.2,
        CharClass::Indic | CharClass::KhmerMyanmar => 1.8,
        CharClass::ThaiLao | CharClass::Arabic | CharClass::Hebrew => 1.5,
        CharClass::Latin
        | CharClass::Cyrillic
        | CharClass::Greek
        | CharClass::Armenian
        | CharClass::Georgian
        | CharClass::Default => 1.0,
        CharClass::Punctuation => 0.5,
        CharClass::Space => 0.2,
        CharClass::Digit => 3.5,
        CharClass::Mark => 0.0,
    }
}

/// Classify one code point.
pub fn char_class(ch: char) -> CharClass {
    // ASCII letters and the space are the overwhelming majority of what this
    // sees, and both would reach the same answer the long way round.
    if ch.is_ascii_alphabetic() {
        return CharClass::Latin;
    }
    if ch == ' ' {
        return CharClass::Space;
    }

    let cp = ch as u32;

    // U+0640 ARABIC TATWEEL is a letter by category but a pure typographic
    // stretch -- it lengthens the join between two letters and is not spoken.
    if cp == 0x0640 {
        return CharClass::Mark;
    }

    // Category before block, so that a combining mark inside a script block is
    // still free and a full stop inside a CJK run is still a pause.
    let cat = CATEGORIES.partition_point(|r| r.lo <= cp);
    if cat > 0 && cp <= CATEGORIES[cat - 1].hi {
        return CATEGORIES[cat - 1].cls;
    }

    let block = BLOCKS.partition_point(|b| b.end < cp);
    if block < BLOCKS.len() {
        return BLOCKS[block].cls;
    }

    // Past the last block. The supplementary planes above U+20000 are CJK
    // extensions; below that sits everything from Linear B to the emoji that
    // the category pass already claimed.
    if cp > 0x20000 { CharClass::Cjk } else { CharClass::Default }
}

/// The speaking weight of one code point.
pub fn char_weight(ch: char) -> f32 {
    class_weight(char_class(ch))
}

/// Sum the weights of every code point in a UTF-8 string.
pub fn text_weight(text: &str) -> f64 {
    let mut total = 0.0f64;
    for ch in text.chars() {
        total += char_weight(ch) as f64;
    }
    total
}

// =============================================================================
// Estimation
// =============================================================================

/// Estimate how long `target` takes to say, given a reference phrase of known
/// length.
///
/// `ref_duration` sets the unit: pass seconds and the result is seconds, pass
/// codec frames and the result is frames. Returns 0 when the reference is
/// unusable (empty, zero-weight, or non-positive duration).
///
/// `low_threshold` is where the short-text boost stops applying, in the same
/// unit; a non-positive value disables the boost. `boost_strength` is the
/// reciprocal of the curve's exponent -- 1 is linear, 3 is the default cube
/// root. The reference defaults are `low_threshold = 50.0` and
/// `boost_strength = 3.0` (`DEFAULT_LOW_THRESHOLD`, `DEFAULT_BOOST_STRENGTH`).
pub fn estimate_duration(
    target: &str,
    reference: &str,
    ref_duration: f64,
    low_threshold: f64,
    boost_strength: f64,
) -> f64 {
    if ref_duration <= 0.0 || reference.is_empty() {
        return 0.0;
    }
    let ref_weight = text_weight(reference);
    if ref_weight <= 0.0 {
        return 0.0;
    }

    // Weight per unit of time, measured on the reference phrase.
    let speed = ref_weight / ref_duration;
    let estimate = text_weight(target) / speed;

    if low_threshold > 0.0 && estimate < low_threshold {
        let alpha = 1.0 / boost_strength;
        return low_threshold * (estimate / low_threshold).powf(alpha);
    }
    estimate
}

/// The reference's default `low_threshold` for `estimate_duration`.
pub const DEFAULT_LOW_THRESHOLD: f64 = 50.0;
/// The reference's default `boost_strength` for `estimate_duration`.
pub const DEFAULT_BOOST_STRENGTH: f64 = 3.0;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// The phrase the reference calibrates against, at 25 frames -- one second.
    const REF: &str = "Nice to meet you.";
    const REF_FRAMES: f64 = 25.0;

    fn cls(cp: u32) -> CharClass {
        char_class(char::from_u32(cp).expect("scalar value"))
    }

    fn estimate(target: &str, reference: &str, ref_duration: f64) -> f64 {
        estimate_duration(target, reference, ref_duration, DEFAULT_LOW_THRESHOLD, DEFAULT_BOOST_STRENGTH)
    }

    // =========================================================================
    // Weights
    // =========================================================================

    #[test]
    fn every_class_carries_its_reference_weight() {
        assert!(approx(class_weight(CharClass::Cjk), 3.0));
        assert!(approx(class_weight(CharClass::Ethiopic), 3.0));
        assert!(approx(class_weight(CharClass::Yi), 3.0));
        assert!(approx(class_weight(CharClass::Hangul), 2.5));
        assert!(approx(class_weight(CharClass::Kana), 2.2));
        assert!(approx(class_weight(CharClass::Indic), 1.8));
        assert!(approx(class_weight(CharClass::KhmerMyanmar), 1.8));
        assert!(approx(class_weight(CharClass::ThaiLao), 1.5));
        assert!(approx(class_weight(CharClass::Arabic), 1.5));
        assert!(approx(class_weight(CharClass::Hebrew), 1.5));
        assert!(approx(class_weight(CharClass::Latin), 1.0));
        assert!(approx(class_weight(CharClass::Cyrillic), 1.0));
        assert!(approx(class_weight(CharClass::Greek), 1.0));
        assert!(approx(class_weight(CharClass::Armenian), 1.0));
        assert!(approx(class_weight(CharClass::Georgian), 1.0));
        assert!(approx(class_weight(CharClass::Default), 1.0));
        assert!(approx(class_weight(CharClass::Punctuation), 0.5));
        assert!(approx(class_weight(CharClass::Space), 0.2));
        assert!(approx(class_weight(CharClass::Digit), 3.5));
        assert!(approx(class_weight(CharClass::Mark), 0.0));
    }

    // =========================================================================
    // Classification
    // =========================================================================

    #[test]
    fn script_blocks_map_to_their_class() {
        assert_eq!(char_class('A'), CharClass::Latin);
        assert_eq!(char_class('z'), CharClass::Latin);
        assert_eq!(cls(0x00E8), CharClass::Latin); // e-grave, precomposed
        assert_eq!(cls(0x1EF9), CharClass::Latin); // Vietnamese y-tilde
        assert_eq!(cls(0x03B1), CharClass::Greek);
        assert_eq!(cls(0x0400), CharClass::Cyrillic);
        assert_eq!(cls(0x0561), CharClass::Armenian);
        assert_eq!(cls(0x10A0), CharClass::Georgian);
        assert_eq!(cls(0x05D0), CharClass::Hebrew);
        assert_eq!(cls(0x0627), CharClass::Arabic);
        assert_eq!(cls(0x0915), CharClass::Indic); // Devanagari ka
        assert_eq!(cls(0x0F00), CharClass::Indic); // Tibetan
        assert_eq!(cls(0x0E01), CharClass::ThaiLao);
        assert_eq!(cls(0x1780), CharClass::KhmerMyanmar);
        assert_eq!(cls(0x1000), CharClass::KhmerMyanmar); // Myanmar
        assert_eq!(cls(0x1200), CharClass::Ethiopic);
        assert_eq!(cls(0xA000), CharClass::Yi);
        assert_eq!(cls(0x3042), CharClass::Kana); // Hiragana
        assert_eq!(cls(0x30A2), CharClass::Kana); // Katakana
        assert_eq!(cls(0xD55C), CharClass::Hangul); // Hangul syllable
        assert_eq!(cls(0x4E2D), CharClass::Cjk);
    }

    #[test]
    fn block_lookup_takes_the_first_end_at_or_above_the_code_point() {
        // The boundaries are what a bisect over end points gets wrong when it is
        // off by one, and every block below is somebody's alphabet.
        assert_eq!(cls(0x02AF), CharClass::Latin); // last of the Latin run
        assert_eq!(cls(0x02B0), CharClass::Greek); // first past it
        assert_eq!(cls(0x9FFF), CharClass::Cjk); // last CJK ideograph
        assert_eq!(cls(0xA000), CharClass::Yi); // first Yi syllable
        assert_eq!(cls(0xD7AF), CharClass::Hangul); // last Hangul syllable
    }

    #[test]
    fn the_category_pass_wins_over_the_block() {
        // A Devanagari vowel sign lives inside the Devanagari block but is a
        // combining mark: silent, and charging it 1.8 would inflate every Hindi
        // estimate.
        assert_eq!(cls(0x093F), CharClass::Mark); // Devanagari vowel sign I
        assert_eq!(cls(0x0301), CharClass::Mark); // combining acute
        assert_eq!(cls(0x17D2), CharClass::Mark); // Khmer sign coeng
        assert_eq!(cls(0x064E), CharClass::Mark); // Arabic fatha

        // Punctuation inside a CJK run is still a pause, not an ideograph.
        assert_eq!(cls(0xFF0C), CharClass::Punctuation); // fullwidth comma
        assert_eq!(cls(0x3001), CharClass::Punctuation); // ideographic comma
        assert_eq!(char_class('.'), CharClass::Punctuation);
        assert_eq!(cls(0x20AC), CharClass::Punctuation); // euro sign (Sc)
        assert_eq!(cls(0x1F600), CharClass::Punctuation); // emoji (So)
    }

    #[test]
    fn digits_weigh_3_5_in_every_script() {
        assert_eq!(char_class('0'), CharClass::Digit);
        assert_eq!(cls(0x0BE6), CharClass::Digit); // Tamil zero
        assert_eq!(cls(0x00B2), CharClass::Digit); // superscript two
        assert_eq!(cls(0xFF10), CharClass::Digit); // fullwidth zero
        assert_eq!(cls(0x1FBF5), CharClass::Digit); // segmented digit five
    }

    #[test]
    fn separators_are_spaces_control_characters_are_not() {
        assert_eq!(char_class(' '), CharClass::Space);
        assert_eq!(cls(0x00A0), CharClass::Space); // no-break space
        assert_eq!(cls(0x3000), CharClass::Space); // ideographic space
        // A newline is category Cc, which no branch claims, so it falls through to
        // the first block and counts as a Latin letter. The reference does this
        // too, and text arriving here is a single line anyway.
        assert_eq!(char_class('\n'), CharClass::Latin);
    }

    #[test]
    fn arabic_tatweel_is_silent() {
        // U+0640 is a letter by category -- it stretches the join between two
        // letters and is not pronounced.
        assert_eq!(cls(0x0640), CharClass::Mark);
    }

    #[test]
    fn the_supplementary_planes_split_at_u_20000() {
        assert_eq!(cls(0xFFEF), CharClass::Latin); // last block entry
        assert_eq!(cls(0xFFF0), CharClass::Default); // past every block
        assert_eq!(cls(0x20000), CharClass::Default); // the bound is strict
        assert_eq!(cls(0x21000), CharClass::Cjk); // CJK Extension B
    }

    // =========================================================================
    // Text weight
    // =========================================================================

    #[test]
    fn text_weight_sums_the_reference_phrase_to_14_1() {
        // 13 Latin letters, 3 spaces at 0.2, one full stop at 0.5.
        assert!(approx(text_weight(REF) as f32, 14.1));
        assert!(approx(text_weight("") as f32, 0.0));
    }

    #[test]
    fn digits_weigh_what_they_take_to_say() {
        // "2024" is four characters and about fourteen letters of speech, because
        // it is spoken as words. Counting characters would ask for a third of the
        // audio it needs.
        assert!(approx(text_weight("2024") as f32, 14.0));
        assert!(approx(text_weight("abcd") as f32, 4.0));
    }

    #[test]
    fn text_weight_is_script_aware() {
        assert!(approx(text_weight("Ciao, mi chiamo Giulia.") as f32, 19.6));
        // Four ideographs at 3.0 and two fullwidth punctuation marks at 0.5.
        assert!(approx(text_weight("你好，世界！") as f32, 13.0));
        // Hindi: the vowel signs and the virama cost nothing.
        assert!(approx(text_weight("नमस्ते दुनिया") as f32, 12.8));
    }

    // =========================================================================
    // Estimation
    // =========================================================================

    #[test]
    fn estimate_duration_scales_against_the_reference() {
        // Well above the boost threshold, so the relation is linear: 100 Latin
        // letters against 14.1 weight in 25 frames.
        let est = estimate(&"A".repeat(100), REF, REF_FRAMES);
        assert!(approx(est as f32, 177.305));
        assert!(approx(est as f32, (100.0 / (14.1 / 25.0)) as f32));
    }

    #[test]
    fn short_estimates_are_boosted_up_a_cube_root() {
        // The reference phrase measures 25 frames by construction, and the boost
        // lifts it to 50 * (25/50)^(1/3).
        let est = estimate(REF, REF, REF_FRAMES);
        assert!(approx(est as f32, (50.0 * 0.5f64.cbrt()) as f32));
        assert!(approx(est as f32, 39.685));

        // Without the boost it is exactly the reference length again.
        assert!(approx(
            estimate_duration(REF, REF, REF_FRAMES, 0.0, DEFAULT_BOOST_STRENGTH) as f32,
            25.0
        ));
    }

    #[test]
    fn the_boost_applies_only_below_the_threshold() {
        let linear = |n: f64| n / (14.1 / 25.0);

        // 29 Latin letters land just above 50 frames, and pass through untouched.
        let above = estimate(&"A".repeat(29), REF, REF_FRAMES);
        assert!(above > 50.0);
        assert!(approx(above as f32, linear(29.0) as f32));

        // 28 land just below, and get pulled up -- a little here, more further
        // down: 4 letters go from 7.1 frames to 26.
        let below = estimate(&"A".repeat(28), REF, REF_FRAMES);
        assert!(below < 50.0);
        assert!(below > linear(28.0));
        assert!(approx(estimate("abcd", REF, REF_FRAMES) as f32, 26.076));
    }

    #[test]
    fn the_boost_is_monotone() {
        // Compression, not saturation: longer text still asks for more frames.
        let mut prev = 0.0;
        for n in 1..=400 {
            let est = estimate(&"A".repeat(n), REF, REF_FRAMES);
            assert!(est > prev);
            prev = est;
        }
    }

    #[test]
    fn an_unusable_reference_estimates_nothing() {
        assert_eq!(estimate("hello", "", REF_FRAMES), 0.0);
        assert_eq!(estimate("hello", REF, 0.0), 0.0);
        assert_eq!(estimate("hello", REF, -1.0), 0.0);
        // A reference of nothing but combining marks weighs zero.
        assert_eq!(estimate("hello", "\u{0301}\u{0301}", REF_FRAMES), 0.0);
    }
}
