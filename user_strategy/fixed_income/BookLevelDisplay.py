"""
BookLevelDisplay — visualizzazione animata su console della posizione
del market maker sul book rispetto al best di mercato.

Uso tipico (da update_HF / update_LF della strategia):

    display = BookLevelDisplay()
    ...
    def update_HF(self):
        ...
        display.render(self.performance, self.compliance)

Funziona su Windows 10+ (ANSI via VT) e Linux/macOS.
"""

import os
import sys
from datetime import datetime
from typing import Optional

from user_strategy.fixed_income._mm_models import MMComplianceTracker, QuotePerformance


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

class _Ansi:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    @staticmethod
    def enable_windows() -> bool:
        """Abilita ANSI su Windows 10+ tramite SetConsoleMode."""
        if os.name != "nt":
            return True
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 0x0001 | 0x0004)
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Caratteri: unicode o ASCII a seconda dell'encoding del terminale
# ─────────────────────────────────────────────────────────────────────────────

class _Ch:
    """Seleziona a runtime caratteri unicode o fallback ASCII."""

    _UNICODE = {
        "thick": "━",
        "thin":  "─",
        "bar_full": "█",
        "bar_hi":   "▓",
        "bar_lo":   "░",
        "dot":   "●",
        "ok":    "✓",
        "err":   "✗",
        "arrow": "►",
        "pipe":  "│",
        "spinner": ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    }
    _ASCII = {
        "thick": "=",
        "thin":  "-",
        "bar_full": "|",
        "bar_hi":   "#",
        "bar_lo":   ".",
        "dot":   "*",
        "ok":    "OK",
        "err":   "!!",
        "arrow": ">",
        "pipe":  "|",
        "spinner": ("|", "/", "-", "\\", "|", "/", "-", "\\", "|", "/"),
    }

    def __init__(self):
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            "━─█▓░●✓✗►│⠋".encode(enc)
            self._ch = self._UNICODE
        except (UnicodeEncodeError, LookupError):
            self._ch = self._ASCII

    def __getattr__(self, name: str):
        return self._ch[name]


# ─────────────────────────────────────────────────────────────────────────────
# BookLevelDisplay
# ─────────────────────────────────────────────────────────────────────────────

class BookLevelDisplay:
    """
    Renderizza su console la posizione del market maker sul book (bid/ask)
    rispetto al best di mercato, con refresh animato in-place.

    Ogni chiamata a ``render()`` sovrascrive le righe precedenti usando
    ANSI escape codes, creando un effetto di aggiornamento live.

    Parameters
    ----------
    at_best_tol_bps:
        Tolleranza in bps per considerare un ordine "at best" a livello
        visivo. Default 1 bps.
    width:
        Larghezza totale del pannello in caratteri. Default 62.
    bar_width:
        Larghezza delle barre quantità. Default 18.
    """

    def __init__(
        self,
        at_best_tol_bps: float = 1.0,
        width: int = 62,
        bar_width: int = 18,
    ):
        self._tol = at_best_tol_bps
        self._width = width
        self._bar_width = bar_width
        self._tick = 0
        self._first_render_done = False
        self._ch = _Ch()
        self._ansi = _Ansi.enable_windows()
        self._color = self._ansi and (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )

    # ── public ───────────────────────────────────────────────────────────────

    def render(
        self,
        performances: dict[str, QuotePerformance],
        compliance: dict[str, MMComplianceTracker],
    ) -> None:
        """
        Aggiorna la visualizzazione in-place.

        Parameters
        ----------
        performances:
            Dict ``"isin:market"`` → :class:`QuotePerformance`.
        compliance:
            Dict ``"isin:market"`` → :class:`MMComplianceTracker`.
        """
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        spinner = self._ch.spinner[self._tick % len(self._ch.spinner)]
        self._tick += 1

        lines: list[str] = []
        for key, perf in performances.items():
            tracker = compliance.get(key)
            lines.extend(self._render_instrument(perf, tracker, ts, spinner))
            lines.append("")

        if lines and lines[-1] == "":
            lines.pop()

        output = "\n".join(lines)

        # \033[H = cursore a home (riga 1, col 1)
        # \033[J = cancella da cursore a fine schermo
        # Combinati in un unico write atomico con il contenuto → no flickering.
        # Più robusto di \033[NA\033[J: non dipende dal conteggio esatto delle righe.
        buf = "\033[H\033[J" if self._ansi and self._first_render_done else ""
        self._first_render_done = True
        buf += output + "\n"
        self._safe_write(buf)

    # ── private: layout ──────────────────────────────────────────────────────

    def _render_instrument(
        self,
        perf: QuotePerformance,
        tracker: Optional[MMComplianceTracker],
        ts: str,
        spinner: str,
    ) -> list[str]:
        w = self._width
        thick = self._c(self._ch.thick * w, _Ansi.BOLD)
        thin  = self._c(self._ch.thin  * w, _Ansi.DIM)

        dot   = self._status_dot(perf)
        isin  = self._c(perf.isin, _Ansi.BOLD, _Ansi.WHITE)
        mkt   = self._c(perf.market, _Ansi.CYAN, _Ansi.BOLD)
        time_ = self._c(ts, _Ansi.DIM)
        spn   = self._c(spinner, _Ansi.DIM)

        return [
            thick,
            f"  {dot} {isin}  {mkt}  {spn}  {time_}",
            thick,
            *self._render_ask(perf),
            *self._render_spread(perf),
            *self._render_bid(perf),
            thin,
            self._render_qty_row(perf),
            *([self._render_compliance_row(tracker)] if tracker else []),
            thick,
        ]

    # ── ask side ─────────────────────────────────────────────────────────────

    def _render_ask(self, perf: QuotePerformance) -> list[str]:
        lines: list[str] = []
        best = perf.best_ask
        ours = perf.ask_order_price

        # best ask di mercato (in alto — prezzo più alto)
        if best is not None:
            lines.append(self._price_row(
                side_label="ASK",
                price=best,
                label=self._c("BEST ASK", _Ansi.DIM),
                bar_color=_Ansi.DIM,
                qty=None,
                indicator="",
                arrow=False,
            ))

        # nostro ask
        if ours is not None:
            at, dist_bps = self._at_best_ask(ours, best)
            if at:
                ind = self._c(f"{self._ch.ok} AT BEST", _Ansi.GREEN, _Ansi.BOLD)
                col = _Ansi.GREEN
            elif dist_bps is not None:
                ind = self._c(f"{self._ch.err} +{dist_bps:.1f}bp", _Ansi.YELLOW)
                col = _Ansi.YELLOW
            else:
                ind = self._c("?", _Ansi.DIM)
                col = _Ansi.WHITE
            lines.append(self._price_row(
                side_label="   ",
                price=ours,
                label=self._c("OUR ASK ", col),
                bar_color=col,
                qty=perf.ask_order_quantity,
                indicator=ind,
                arrow=True,
                price_color=col,
            ))
        else:
            lines.append(self._missing_row("OUR ASK"))

        return lines

    # ── bid side ─────────────────────────────────────────────────────────────

    def _render_bid(self, perf: QuotePerformance) -> list[str]:
        lines: list[str] = []
        best = perf.best_bid
        ours = perf.bid_order_price

        # nostro bid
        if ours is not None:
            at, dist_bps = self._at_best_bid(ours, best)
            if at:
                ind = self._c(f"{self._ch.ok} AT BEST", _Ansi.GREEN, _Ansi.BOLD)
                col = _Ansi.GREEN
            elif dist_bps is not None:
                ind = self._c(f"{self._ch.err} -{dist_bps:.1f}bp", _Ansi.YELLOW)
                col = _Ansi.YELLOW
            else:
                ind = self._c("?", _Ansi.DIM)
                col = _Ansi.WHITE
            lines.append(self._price_row(
                side_label="   ",
                price=ours,
                label=self._c("OUR BID ", col),
                bar_color=col,
                qty=perf.bid_order_quantity,
                indicator=ind,
                arrow=True,
                price_color=col,
            ))
        else:
            lines.append(self._missing_row("OUR BID"))

        # best bid di mercato (in basso — prezzo più basso)
        if best is not None:
            lines.append(self._price_row(
                side_label="BID",
                price=best,
                label=self._c("BEST BID", _Ansi.DIM),
                bar_color=_Ansi.DIM,
                qty=None,
                indicator="",
                arrow=False,
            ))

        return lines

    # ── spread row ───────────────────────────────────────────────────────────

    def _render_spread(self, perf: QuotePerformance) -> list[str]:
        parts: list[str] = []

        if perf.market_spread is not None and perf.best_bid and perf.best_ask:
            mid = (perf.best_bid + perf.best_ask) / 2
            mkt_bps = perf.market_spread / mid * 10_000
            parts.append(f"mkt {self._c(f'{mkt_bps:.1f}bp', _Ansi.CYAN)}")

        if perf.our_spread_pct is not None:
            our_bps = perf.our_spread_pct * 10_000
            col = _Ansi.GREEN if perf.meets_spread_req else _Ansi.RED
            parts.append(f"ours {self._c(f'{our_bps:.1f}bp', col)}")

        inner = f"  {self._ch.pipe}  ".join(parts) if parts else self._c("no spread data", _Ansi.DIM)
        line = f"  {self._ch.thin * 6}  spread: {inner}"
        return [self._c(line, _Ansi.DIM)]

    # ── bottom rows ──────────────────────────────────────────────────────────

    def _render_qty_row(self, perf: QuotePerformance) -> str:
        def fmt(qty: Optional[float], ok: bool) -> str:
            s = f"{qty:>6.0f}" if qty is not None else "   ---"
            col = _Ansi.GREEN if ok else _Ansi.RED
            return self._c(s, col)

        ok = perf.meets_quantity_req
        bid_s = fmt(perf.bid_order_quantity, ok)
        ask_s = fmt(perf.ask_order_quantity, ok)
        return f"  Qty  BID: {bid_s}  ASK: {ask_s}"

    def _render_compliance_row(self, tracker: MMComplianceTracker) -> str:
        ratio   = tracker.compliance_ratio
        filled  = round(ratio * self._bar_width)
        empty   = self._bar_width - filled
        col     = _Ansi.GREEN if ratio >= 0.8 else (_Ansi.YELLOW if ratio >= 0.6 else _Ansi.RED)
        bar     = self._c(self._ch.bar_hi * filled, col) + self._c(self._ch.bar_lo * empty, _Ansi.DIM)
        pct     = self._c(f"{ratio * 100:5.1f}%", col, _Ansi.BOLD)
        ticks   = self._c(f"({tracker.compliant_ticks}/{tracker.total_ticks})", _Ansi.DIM)
        return f"  Compliance  {bar} {pct} {ticks}"

    # ── row builders ─────────────────────────────────────────────────────────

    def _price_row(
        self,
        side_label: str,
        price: float,
        label: str,
        bar_color: str,
        qty: Optional[float],
        indicator: str,
        arrow: bool,
        price_color: str = _Ansi.WHITE,
    ) -> str:
        arrow_ch = self._c(self._ch.arrow, price_color, _Ansi.BOLD) if arrow else " "
        price_s  = self._c(f"{price:>10.4f}", price_color, _Ansi.BOLD)
        qty_s    = f"{qty:>6.0f}" if qty is not None else "      "
        bar_len  = min(self._bar_width, int((qty or 0) / 100)) if qty else 4
        bar_s    = self._c(self._ch.bar_full * max(1, bar_len), bar_color)
        return f"  {side_label:3}  {arrow_ch} {price_s}  {bar_s:<{self._bar_width}}  {qty_s}  {label}  {indicator}"

    def _missing_row(self, label: str) -> str:
        return (
            f"        {self._c('       ---', _Ansi.RED)}  "
            f"{self._c(label, _Ansi.DIM)}  "
            f"{self._c(self._ch.err + ' MISSING', _Ansi.RED, _Ansi.BOLD)}"
        )

    # ── status dot ───────────────────────────────────────────────────────────

    def _status_dot(self, perf: QuotePerformance) -> str:
        if perf.is_compliant:
            return self._c(self._ch.dot, _Ansi.GREEN)
        if perf.is_two_sided:
            return self._c(self._ch.dot, _Ansi.YELLOW)
        return self._c(self._ch.dot, _Ansi.RED)

    # ── at-best helpers ──────────────────────────────────────────────────────

    def _at_best_ask(
        self, our: float, best: Optional[float]
    ) -> tuple[bool, Optional[float]]:
        """Ritorna (at_best, dist_bps). Ask è at best se our <= best."""
        if best is None:
            return False, None
        dist_bps = (our - best) / best * 10_000
        return our <= best or abs(dist_bps) <= self._tol, max(dist_bps, 0.0)

    def _at_best_bid(
        self, our: float, best: Optional[float]
    ) -> tuple[bool, Optional[float]]:
        """Ritorna (at_best, dist_bps). Bid è at best se our >= best."""
        if best is None:
            return False, None
        dist_bps = (best - our) / best * 10_000
        return our >= best or abs(dist_bps) <= self._tol, max(dist_bps, 0.0)

    # ── ANSI helpers ─────────────────────────────────────────────────────────

    def _c(self, text: str, *codes: str) -> str:
        if not self._color:
            return text
        return "".join(codes) + text + _Ansi.RESET

    def _safe_write(self, text: str) -> None:
        """Scrive su stdout gestendo UnicodeEncodeError (es. cp1252 su Windows)."""
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", "ascii") or "ascii"
            sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
            sys.stdout.flush()

