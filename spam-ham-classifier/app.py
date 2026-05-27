"""
Spam/Ham Classifier - Clean UI
Run AFTER train.py has been executed.
"""

import os
import re
import string
import tkinter as tk
from tkinter import font as tkfont
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# ── Load saved model & vectorizer ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
NAME_PATH  = os.path.join(BASE_DIR, "models", "model_name.pkl")

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
model_name = joblib.load(NAME_PATH)

stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ── NLP pipeline ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r'\d+', '', text)
    text   = text.translate(str.maketrans('', '', string.punctuation))
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    return ' '.join(stemmer.stem(w) for w in tokens if w not in stop_words)

def predict(raw: str):
    """Returns (label_str, confidence_0_to_1_or_None)."""
    vec   = vectorizer.transform([preprocess(raw)])
    label = "SPAM" if model.predict(vec)[0] == 1 else "HAM"
    try:
        conf = float(max(model.predict_proba(vec)[0]))
    except AttributeError:
        conf = None
    return label, conf


# ═════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═════════════════════════════════════════════════════════════════════════════
C = {
    "bg"        : "#F7F8FC",
    "surface"   : "#FFFFFF",
    "border"    : "#E4E7EF",
    "border_focus": "#6366F1",
    "text_dark" : "#1E1E2E",
    "text_mid"  : "#6B7280",
    "text_light": "#9CA3AF",
    # accent
    "indigo"    : "#6366F1",
    "indigo_dk" : "#4F46E5",
    "indigo_bg" : "#EEF2FF",
    # verdict
    "spam_fg"   : "#DC2626",
    "spam_bg"   : "#FEF2F2",
    "spam_bd"   : "#FCA5A5",
    "ham_fg"    : "#16A34A",
    "ham_bg"    : "#F0FDF4",
    "ham_bd"    : "#86EFAC",
    # progress bar track
    "bar_track" : "#E5E7EB",
}


# ═════════════════════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ═════════════════════════════════════════════════════════════════════════════
class FlatButton(tk.Label):
    """A clean flat button using tk.Label for reliable cross-platform rendering."""

    def __init__(self, parent, text, command,
                 bg=None, fg="#FFFFFF",
                 hover_bg=None, font=None,
                 padx=22, pady=10, **kw):
        _bg       = bg       or C["indigo"]
        _hover_bg = hover_bg or C["indigo_dk"]
        _font     = font or tkfont.Font(family="Segoe UI", size=10, weight="bold")
        super().__init__(parent, text=text, bg=_bg, fg=fg,
                         font=_font, padx=padx, pady=pady,
                         cursor="hand2", relief="flat", **kw)
        self._bg       = _bg
        self._hover_bg = _hover_bg
        self._cmd      = command
        self.bind("<Enter>",    lambda _: self.config(bg=self._hover_bg))
        self.bind("<Leave>",    lambda _: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda _: self._cmd() if self._cmd else None)


class Divider(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, height=1, bg=C["border"], **kw)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class SpamApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Spam Detector")
        self.geometry("640x720")
        self.minsize(580, 640)
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        self._define_fonts()
        self._build_ui()

    # ── Font definitions ──────────────────────────────────────────────────────
    def _define_fonts(self):
        self.F = {
            "heading"   : tkfont.Font(family="Segoe UI", size=18, weight="bold"),
            "subheading": tkfont.Font(family="Segoe UI", size=10),
            "label"     : tkfont.Font(family="Segoe UI", size=10, weight="bold"),
            "body"      : tkfont.Font(family="Segoe UI", size=11),
            "small"     : tkfont.Font(family="Segoe UI", size=9),
            "verdict"   : tkfont.Font(family="Segoe UI", size=22, weight="bold"),
            "conf"      : tkfont.Font(family="Segoe UI", size=10),
            "badge"     : tkfont.Font(family="Segoe UI", size=8, weight="bold"),
            "history"   : tkfont.Font(family="Consolas",  size=9),
        }

    # ── Master layout ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top nav bar ───────────────────────────────────────────────────────
        nav = tk.Frame(self, bg=C["surface"], height=60)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        nav_inner = tk.Frame(nav, bg=C["surface"])
        nav_inner.pack(fill="both", expand=True, padx=24)

        # dot logo
        dot_canvas = tk.Canvas(nav_inner, width=28, height=28,
                               bg=C["surface"], highlightthickness=0)
        dot_canvas.pack(side="left", pady=16)
        dot_canvas.create_oval(0, 0, 28, 28, fill=C["indigo"], outline="")
        dot_canvas.create_oval(8, 8, 20, 20, fill="#FFFFFF", outline="")

        tk.Label(nav_inner, text="  SpamGuard", font=self.F["heading"],
                 bg=C["surface"], fg=C["text_dark"]).pack(side="left", pady=16)

        # model badge
        badge = tk.Label(nav_inner,
                         text=f"  {model_name}  ",
                         font=self.F["badge"],
                         bg=C["indigo_bg"], fg=C["indigo"],
                         relief="flat", padx=4, pady=3)
        badge.pack(side="right", pady=20)

        Divider(self).pack(fill="x")

        # ── Scrollable body ───────────────────────────────────────────────────
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=28, pady=20)

        # ── Input section ─────────────────────────────────────────────────────
        self._build_input_section(body)

        # ── Result section ────────────────────────────────────────────────────
        self._build_result_section(body)

        # ── History section ───────────────────────────────────────────────────
        self._build_history_section(body)

        # ── Footer ────────────────────────────────────────────────────────────
        Divider(self).pack(fill="x")
        tk.Label(self,
                 text="TF-IDF + NLP Pipeline  •  Ctrl+Enter to classify",
                 font=self.F["small"], bg=C["surface"], fg=C["text_light"],
                 pady=8).pack(fill="x")

    # ── Input section ─────────────────────────────────────────────────────────
    def _build_input_section(self, parent):
        card = tk.Frame(parent, bg=C["surface"],
                        highlightthickness=1,
                        highlightbackground=C["border"])
        card.pack(fill="x", pady=(0, 16))

        inner = tk.Frame(card, bg=C["surface"])
        inner.pack(fill="x", padx=20, pady=18)

        # label row
        lbl_row = tk.Frame(inner, bg=C["surface"])
        lbl_row.pack(fill="x", pady=(0, 10))
        tk.Label(lbl_row, text="Message", font=self.F["label"],
                 bg=C["surface"], fg=C["text_dark"]).pack(side="left")
        tk.Label(lbl_row, text="Paste or type any SMS / email snippet",
                 font=self.F["small"], bg=C["surface"],
                 fg=C["text_light"]).pack(side="left", padx=(8, 0))

        # text area with focus ring
        self._focus_ring = tk.Frame(inner, bg=C["border"],
                                    highlightthickness=0)
        self._focus_ring.pack(fill="x")

        text_wrap = tk.Frame(self._focus_ring, bg=C["surface"], pady=1, padx=1)
        text_wrap.pack(fill="x", padx=1, pady=1)

        self.text_input = tk.Text(
            text_wrap, height=5,
            font=self.F["body"],
            bg=C["surface"], fg=C["text_dark"],
            insertbackground=C["indigo"],
            relief="flat", bd=0,
            padx=12, pady=10,
            wrap="word",
            highlightthickness=0,
        )
        self.text_input.pack(fill="x")
        self.text_input.bind("<FocusIn>",  self._focus_in)
        self.text_input.bind("<FocusOut>", self._focus_out)
        self.text_input.bind("<Control-Return>", lambda _: self._classify())

        # char counter
        self._char_var = tk.StringVar(value="0 chars")
        tk.Label(inner, textvariable=self._char_var,
                 font=self.F["small"], bg=C["surface"],
                 fg=C["text_light"]).pack(anchor="e", pady=(4, 0))
        self.text_input.bind("<KeyRelease>", self._update_char_count)

        # button row
        btn_row = tk.Frame(inner, bg=C["surface"])
        btn_row.pack(fill="x", pady=(14, 0))

        FlatButton(btn_row, "Classify", command=self._classify).pack(side="left")

        tk.Button(btn_row, text="Clear",
                  font=self.F["label"],
                  bg=C["bg"], fg=C["text_mid"],
                  activebackground=C["border"],
                  activeforeground=C["text_dark"],
                  relief="flat", cursor="hand2",
                  padx=18, pady=8,
                  command=self._clear).pack(side="left", padx=(10, 0))

    def _focus_in(self, _):
        self._focus_ring.config(bg=C["border_focus"])

    def _focus_out(self, _):
        self._focus_ring.config(bg=C["border"])

    def _update_char_count(self, _=None):
        n = len(self.text_input.get("1.0", "end").strip())
        self._char_var.set(f"{n} chars")

    # ── Result section ────────────────────────────────────────────────────────
    def _build_result_section(self, parent):
        self._result_frame = tk.Frame(parent, bg=C["surface"],
                                      highlightthickness=1,
                                      highlightbackground=C["border"])
        self._result_frame.pack(fill="x", pady=(0, 16))

        inner = tk.Frame(self._result_frame, bg=C["surface"])
        inner.pack(fill="x", padx=20, pady=18)

        tk.Label(inner, text="Result", font=self.F["label"],
                 bg=C["surface"], fg=C["text_dark"]).pack(anchor="w")

        # placeholder row
        self._placeholder = tk.Label(
            inner,
            text="Enter a message and click Classify to see the verdict.",
            font=self.F["body"], bg=C["surface"],
            fg=C["text_light"], wraplength=520, justify="left"
        )
        self._placeholder.pack(anchor="w", pady=(12, 0))

        # verdict row (hidden until first classification)
        self._verdict_row = tk.Frame(inner, bg=C["surface"])

        self._icon_lbl = tk.Label(self._verdict_row, text="",
                                  font=tkfont.Font(family="Segoe UI", size=28),
                                  bg=C["surface"])
        self._icon_lbl.pack(side="left")

        verdict_text = tk.Frame(self._verdict_row, bg=C["surface"])
        verdict_text.pack(side="left", padx=(14, 0))

        self._verdict_lbl = tk.Label(verdict_text, text="",
                                     font=self.F["verdict"],
                                     bg=C["surface"], anchor="w")
        self._verdict_lbl.pack(anchor="w")

        self._conf_lbl = tk.Label(verdict_text, text="",
                                  font=self.F["conf"],
                                  bg=C["surface"], anchor="w")
        self._conf_lbl.pack(anchor="w", pady=(2, 0))

        # confidence bar (lives below verdict_row)
        self._bar_outer = tk.Frame(inner, bg=C["bar_track"], height=6)
        self._bar_inner = tk.Frame(self._bar_outer, bg=C["indigo"], height=6)
        self._bar_inner.place(x=0, y=0, relheight=1, relwidth=0)

    def _show_verdict(self, label: str, conf):
        self._placeholder.pack_forget()
        self._verdict_row.pack(anchor="w", pady=(12, 0))
        self._bar_outer.pack(fill="x", pady=(14, 0))

        is_spam  = (label == "SPAM")
        fg_clr   = C["spam_fg"]   if is_spam else C["ham_fg"]
        bg_clr   = C["spam_bg"]   if is_spam else C["ham_bg"]
        bd_clr   = C["spam_bd"]   if is_spam else C["ham_bd"]
        icon     = "X"            if is_spam else "OK"
        bar_clr  = C["spam_fg"]   if is_spam else C["ham_fg"]

        # update card colours
        self._result_frame.config(highlightbackground=bd_clr)
        for w in (self._result_frame,
                  self._verdict_row, self._conf_lbl,
                  self._verdict_lbl, self._icon_lbl):
            w.config(bg=bg_clr)
        # the inner wrapper too
        self._result_frame.winfo_children()[0].config(bg=bg_clr)

        self._icon_lbl.config(text=f"[{icon}]", fg=fg_clr)
        self._verdict_lbl.config(text=label, fg=fg_clr)

        if conf is not None:
            pct = conf * 100
            self._conf_lbl.config(
                text=f"Confidence: {pct:.1f}%", fg=fg_clr)
            self._bar_outer.config(bg=C["bar_track"])
            self._bar_inner.config(bg=bar_clr)
            self._bar_outer.update_idletasks()
            self._bar_inner.place(x=0, y=0, relheight=1, relwidth=conf)
        else:
            self._conf_lbl.config(text="", fg=fg_clr)
            self._bar_inner.place(x=0, y=0, relheight=1, relwidth=0.95 if is_spam else 0.05)
            self._bar_inner.config(bg=bar_clr)
            self._bar_outer.config(bg=C["bar_track"])

    def _reset_verdict(self):
        self._verdict_row.pack_forget()
        self._bar_outer.pack_forget()
        self._placeholder.pack(anchor="w", pady=(12, 0))
        self._result_frame.config(highlightbackground=C["border"],
                                  bg=C["surface"])
        for w in (self._result_frame,
                  self._verdict_row, self._conf_lbl,
                  self._verdict_lbl, self._icon_lbl):
            try:
                w.config(bg=C["surface"])
            except Exception:
                pass

    # ── History section ───────────────────────────────────────────────────────
    def _build_history_section(self, parent):
        hdr = tk.Frame(parent, bg=C["bg"])
        hdr.pack(fill="x", pady=(0, 8))

        tk.Label(hdr, text="Recent", font=self.F["label"],
                 bg=C["bg"], fg=C["text_dark"]).pack(side="left")
        self._clear_hist_btn = tk.Label(
            hdr, text="Clear history",
            font=self.F["small"], bg=C["bg"],
            fg=C["indigo"], cursor="hand2"
        )
        self._clear_hist_btn.pack(side="right")
        self._clear_hist_btn.bind("<Button-1>", self._clear_history)

        # history list (scrollable)
        hist_wrap = tk.Frame(parent, bg=C["surface"],
                             highlightthickness=1,
                             highlightbackground=C["border"])
        hist_wrap.pack(fill="both", expand=True)

        self._hist_canvas = tk.Canvas(
            hist_wrap, bg=C["surface"],
            highlightthickness=0, bd=0
        )
        scrollbar = tk.Scrollbar(hist_wrap, orient="vertical",
                                 command=self._hist_canvas.yview)
        self._hist_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._hist_canvas.pack(side="left", fill="both", expand=True)

        self._hist_inner = tk.Frame(self._hist_canvas, bg=C["surface"])
        self._hist_window = self._hist_canvas.create_window(
            (0, 0), window=self._hist_inner, anchor="nw"
        )
        self._hist_inner.bind("<Configure>", self._on_hist_configure)
        self._hist_canvas.bind("<Configure>", self._on_canvas_configure)

        self._history_rows = []

        # empty state label
        self._empty_lbl = tk.Label(
            self._hist_inner,
            text="No predictions yet.",
            font=self.F["small"], bg=C["surface"],
            fg=C["text_light"], pady=20
        )
        self._empty_lbl.pack()

    def _on_hist_configure(self, _):
        self._hist_canvas.configure(
            scrollregion=self._hist_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._hist_canvas.itemconfig(self._hist_window, width=event.width)

    def _add_history_row(self, label: str, snippet: str, conf):
        if self._empty_lbl.winfo_ismapped():
            self._empty_lbl.pack_forget()

        is_spam = (label == "SPAM")
        fg = C["spam_fg"] if is_spam else C["ham_fg"]
        bg = C["spam_bg"] if is_spam else C["ham_bg"]
        tag_text = label

        row = tk.Frame(self._hist_inner, bg=C["surface"])
        row.pack(fill="x", padx=0, pady=0)

        # coloured tag pill
        tag_lbl = tk.Label(row, text=f" {tag_text} ",
                           font=self.F["badge"],
                           bg=bg, fg=fg, padx=4, pady=3)
        tag_lbl.pack(side="left", padx=(12, 10), pady=8)

        # snippet
        tk.Label(row, text=snippet,
                 font=self.F["history"],
                 bg=C["surface"], fg=C["text_mid"],
                 anchor="w").pack(side="left", fill="x", expand=True)

        # confidence
        if conf is not None:
            tk.Label(row, text=f"{conf*100:.0f}%",
                     font=self.F["small"],
                     bg=C["surface"], fg=C["text_light"],
                     padx=12).pack(side="right")

        Divider(self._hist_inner).pack(fill="x", padx=12)

        self._history_rows.append(row)
        # auto-scroll
        self._hist_canvas.update_idletasks()
        self._hist_canvas.yview_moveto(1.0)

    def _clear_history(self, _=None):
        for row in self._history_rows:
            row.destroy()
        for w in self._hist_inner.winfo_children():
            if isinstance(w, tk.Frame) and w != self._empty_lbl:
                w.destroy()
        self._history_rows.clear()
        # remove extra dividers
        for child in self._hist_inner.winfo_children():
            child.destroy()
        self._empty_lbl = tk.Label(
            self._hist_inner,
            text="No predictions yet.",
            font=self.F["small"], bg=C["surface"],
            fg=C["text_light"], pady=20
        )
        self._empty_lbl.pack()

    # ── Actions ───────────────────────────────────────────────────────────────
    def _classify(self):
        raw = self.text_input.get("1.0", "end").strip()
        if not raw:
            self._shake_input()
            return

        label, conf = predict(raw)
        self._show_verdict(label, conf)

        snippet = raw[:60] + ("..." if len(raw) > 60 else "")
        self._add_history_row(label, snippet, conf)

    def _clear(self):
        self.text_input.delete("1.0", "end")
        self._char_var.set("0 chars")
        self._reset_verdict()

    def _shake_input(self):
        frame = self._focus_ring
        orig_x = frame.winfo_x()
        for dx in [6, -6, 4, -4, 2, -2, 0]:
            frame.place(x=orig_x + dx)
            self.update()
            self.after(25)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train.py first!")
    else:
        app = SpamApp()
        app.mainloop()
