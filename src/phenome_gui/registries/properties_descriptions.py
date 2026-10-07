PRESET_INFO = {
    "basic": {
        "label": "Basic",
        "description": "<b>Basic:</b></br> Essential intensity stats (mean, std) and mask shape properties (area, perimeter, eccentricity, solidity). Recommended for most users.",
    },
    "standard": {
        "label": "Standard",
        "description": "<b>Standard:</b></br> Comprehensive set including all shape, intensity stats, and concentric ring features.",
    },
    "complete": {
        "label": "Complete",
        "description": "<b>Complete:</b></br> Maximum features including texture (GLCM), quality metrics (blur, entropy), and advanced shape descriptors.",
    },
    "shape": {
        "label": "Shape",
        "description": "<b>Shape:</b></br> Mask shape properties only (area, eccentricity, axis lengths, etc.). Requires masks.",
    },
    "intensity": {
        "label": "Intensity",
        "description": "<b>Intensity:</b></br> Image intensity statistics only (mean, std, min, max). Works without masks.",
    }
}
