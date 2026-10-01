"""
Style registry for the QtWidgets.
"""

STYLESHEETS = {
    "STYLE_OK": """
        QLineEdit {
            border: 2px solid green;
            border-radius: 4px;
            padding: 2px;
        }
    """,

    "STYLE_WAIT":"""
        QLineEdit {
            border: 2px solid yellow;
            border-radius: 4px;
            padding: 2px;
        }
    """,

    "STYLE_BAD":"""
        QLineEdit {
            border: 2px solid red;
            border-radius: 4px;
            padding: 2px;
        }
    """
}
