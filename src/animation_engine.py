"""
src/animation_engine.py
Animated brand visuals — GIF generation using Matplotlib FuncAnimation.

Styles: Fade In, Slide In Left, Slide In Up, Typewriter, Zoom In, Pulse.
Falls back to base64-encoded static PNG if animation export fails.
"""

from __future__ import annotations
import io
import logging
import base64
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _hex_to_rgb_float(h: str) -> tuple[float,float,float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))


def _create_animation_gif(
    company:    str,
    slogan:     str,
    primary:    str,
    accent:     str,
    style:      str,
    font_name:  str = "serif",
    fps:        int = 15,
    duration_s: float = 2.5,
) -> bytes | None:
    """
    Generate a branded animated GIF.
    Returns raw GIF bytes or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import FancyBboxPatch

        bg   = _hex_to_rgb_float(primary)
        fg   = _hex_to_rgb_float(accent)
        n_frames = int(fps * duration_s)
        fig, ax  = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis("off")

        company_txt = ax.text(0.5, 0.6, company.upper(),
            ha="center", va="center", color=fg,
            fontsize=28, fontweight="light",
            fontfamily="serif", letter_spacing=4 if hasattr(ax.texts[0],'set_letter_spacing') else None,
        )
        slogan_txt  = ax.text(0.5, 0.38, slogan,
            ha="center", va="center", color=fg,
            fontsize=11, fontweight="light", alpha=0,
        )
        # Thin divider line
        line, = ax.plot([0.25, 0.75],[0.51,0.51], color=fg, linewidth=0.6, alpha=0)

        def _ease(t): return t*t*(3-2*t)  # smoothstep

        def _fade_in(frame):
            t = _ease(frame / n_frames)
            company_txt.set_alpha(t)
            slogan_txt.set_alpha(max(0, t*2-1))
            line.set_alpha(max(0, t*2-0.8))

        def _slide_left(frame):
            t = _ease(frame / n_frames)
            company_txt.set_alpha(t)
            company_txt.set_x(max(0.5, 1.5 - t))
            slogan_txt.set_alpha(max(0, t*1.5-0.5))
            line.set_alpha(max(0, t*2-1))

        def _slide_up(frame):
            t = _ease(frame / n_frames)
            y_company = 0.4 + 0.2*t
            company_txt.set_alpha(t)
            company_txt.set_y(y_company)
            slogan_txt.set_alpha(max(0, t*1.5-0.5))
            line.set_alpha(max(0, t*2-1))

        def _typewriter(frame):
            chars = max(0, int(frame / n_frames * len(company)))
            company_txt.set_text(company.upper()[:chars])
            company_txt.set_alpha(1.0)
            slogan_txt.set_alpha(1.0 if frame > n_frames*0.7 else 0)
            line.set_alpha(1.0 if frame > n_frames*0.7 else 0)

        def _zoom_in(frame):
            t = _ease(frame / n_frames)
            size = 8 + 20*t
            company_txt.set_fontsize(size)
            company_txt.set_alpha(t)
            slogan_txt.set_alpha(max(0, t*1.5-0.5))
            line.set_alpha(max(0, t*2-1))

        def _pulse(frame):
            t = frame / n_frames
            size = 26 + 4*np.sin(t*2*np.pi*2)
            company_txt.set_fontsize(size)
            company_txt.set_alpha(1.0)
            slogan_txt.set_alpha(1.0)
            line.set_alpha(1.0)

        anim_funcs = {
            "Fade In":         _fade_in,
            "Slide In Left":   _slide_left,
            "Slide In Up":     _slide_up,
            "Typewriter":      _typewriter,
            "Zoom In":         _zoom_in,
            "Pulse":           _pulse,
        }
        update_fn = anim_funcs.get(style, _fade_in)
        # Init state
        company_txt.set_alpha(0); slogan_txt.set_alpha(0); line.set_alpha(0)

        anim = animation.FuncAnimation(
            fig, update_fn, frames=n_frames, interval=1000/fps, blit=False)

        buf = io.BytesIO()
        anim.save(buf, writer="pillow", fps=fps)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        log.warning("Animation GIF generation failed: %s", e)
        plt.close("all")
        return None


def get_animation_gif(
    company:    str,
    slogan:     str,
    palette:    list[dict],
    style:      str = "Fade In",
) -> tuple[Optional[bytes], str]:
    """
    Returns (gif_bytes_or_None, base64_str_for_fallback_png).
    Callers should use gif if available, else render the PNG.
    """
    primary = palette[0]["hex"] if palette else "#0A0A08"
    accent  = palette[2]["hex"] if len(palette) > 2 else "#C8A94A"

    gif = _create_animation_gif(company, slogan, primary, accent, style)

    # Fallback static PNG
    fallback_b64 = ""
    if gif is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            bg  = _hex_to_rgb_float(primary)
            fg  = _hex_to_rgb_float(accent)
            fig, ax = plt.subplots(figsize=(8,4))
            fig.patch.set_facecolor(bg); ax.set_facecolor(bg); ax.axis("off")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.text(0.5,0.60,company.upper(),ha="center",va="center",
                    color=fg,fontsize=26,fontweight="light",fontfamily="serif")
            ax.plot([0.28,0.72],[0.51,0.51],color=fg,linewidth=0.6)
            ax.text(0.5,0.38,slogan,ha="center",va="center",
                    color=fg,fontsize=11,fontweight="light")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            plt.close(fig)
            fallback_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as e2:
            log.warning("Static fallback PNG also failed: %s", e2)

    return gif, fallback_b64
