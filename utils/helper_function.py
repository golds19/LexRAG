"""
helper functions
"""
def clean_text(text:str):
    """
    This function applies basic text
    preprocessing
    """
    # remove excessive whitespace
    text = " ".join(text.split())

    # fix ligatures
    # Fix ligatures (e.g., "ﬁ" to "fi")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    return text