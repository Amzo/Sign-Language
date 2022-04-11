import tkinter as tk


class PiWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Raspberry Pi Camera")
        self.geometry("640x480")
        self.piFrame = None
        self.piVideoLabel = tk.Label(self, borderwidth=3)
        self.piVideoLabel.grid(row=0, column=0)

    def updatePiWindow(self):
        self.piVideoLabel.configure(image=self.piFrame)
