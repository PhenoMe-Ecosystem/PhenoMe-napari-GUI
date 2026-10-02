"""
Style registry for the QtWidgets.
"""

# COLORS
COLOR_OK = "#539055"
COLOR_WAIT = "#b4a86e"
COLOR_ERROR = "#9e4f49"

STYLESHEETS = {
    "STYLE_OK": f"""
        QLineEdit {{
            border: 2px solid {COLOR_OK};
            border-radius: 4px;
            padding: 2px;
        }}
        QPushButton {{
            border: 2px solid {COLOR_OK};
            border-radius: 4px;
            padding: 2px;
        }}
    """,

    "STYLE_WAIT": f"""
        QLineEdit {{
            border: 2px solid {COLOR_WAIT};
            border-radius: 4px;
            padding: 2px;
        }}
        QPushButton {{
            border: 2px solid {COLOR_WAIT};
            border-radius: 4px;
            padding: 2px;
        }}
    """,

    "STYLE_ERROR": f"""
        QLineEdit {{
            border: 2px solid {COLOR_ERROR};
            border-radius: 4px;
            padding: 2px;
        }}
        QPushButton {{
            border: 2px solid {COLOR_ERROR};
            border-radius: 4px;
            padding: 2px;
        }}
    """,
}
