"""
Formatting Utilities for Urban Road Analytics and Maintenance Assessment System.
"""

def format_number(val, decimals=1):
    """Format float or int to string with commas and specified decimals."""
    if val is None or str(val) == "nan":
        return "N/A"
    try:
        if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
            return f"{int(val):,}"
        return f"{float(val):,.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)

def format_percentage(val, decimals=1):
    """Format float as percentage."""
    if val is None or str(val) == "nan":
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}%"
    except (ValueError, TypeError):
        return str(val)

def priority_badge_html(priority: str) -> str:
    """Generate HTML badge for maintenance priority."""
    colors = {
        "High": ("#721c24", "#f8d7da", "#f5c6cb"),
        "Medium": ("#856404", "#fff3cd", "#ffeeba"),
        "Low": ("#155724", "#d4edda", "#c3e6cb")
    }
    text_color, bg_color, border_color = colors.get(priority, ("#383d41", "#e2e3e5", "#d6d8db"))
    return f"""<span style="
        background-color: {bg_color};
        color: {text_color};
        border: 1px solid {border_color};
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;">{priority}</span>"""

def status_badge_html(status: str) -> str:
    """Generate HTML badge for validation status."""
    colors = {
        "PASS": ("#155724", "#d4edda", "#c3e6cb"),
        "WARNING": ("#856404", "#fff3cd", "#ffeeba"),
        "FAIL": ("#721c24", "#f8d7da", "#f5c6cb")
    }
    text_color, bg_color, border_color = colors.get(status, ("#383d41", "#e2e3e5", "#d6d8db"))
    return f"""<span style="
        background-color: {bg_color};
        color: {text_color};
        border: 1px solid {border_color};
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;">{status}</span>"""

def metric_card_html(title: str, value: str, subtitle: str = "", border_color: str = "#3b82f6") -> str:
    """Render a clean metric card container."""
    sub_html = f'<div style="font-size: 0.78rem; color: #6b7280; margin-top: 4px;">{subtitle}</div>' if subtitle else ""
    return f"""<div style="
        background-color: #ffffff;
        border-left: 4px solid {border_color};
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        padding: 14px 16px;
        margin-bottom: 12px;">
        <div style="font-size: 0.82rem; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: #111827; margin-top: 2px;">{value}</div>
        {sub_html}
    </div>"""
