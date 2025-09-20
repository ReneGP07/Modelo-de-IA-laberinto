
class Control:
    """Maneja atajos de teclado para pausar, avanzar nivel y salir sin guardar."""
    def __init__(self):
        self.paused = False
        self.advance_level = False
        self.quit_without_save = False

    def bind_to(self, fig):
        def on_key(evt):
            k = (evt.key or "").lower()
            if k in ("p", " "):
                self.paused = not self.paused
            elif k in ("n", "right"):
                self.advance_level = True
            elif k in ("q", "escape"):
                self.quit_without_save = True
        fig.canvas.mpl_connect("key_press_event", on_key)
