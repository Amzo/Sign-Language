import tkinter as tk
from enum import Enum
from tkinter import scrolledtext


class LogInfo(Enum):
    info = 1
    warning = 2
    error = 3
    debug = 4


class DebugWindow(tk.Toplevel):
    def __init__(self, master=None, main=None):
        super().__init__(master=master)
        self.master = main
        self.title("Debug Menu")
        self.geometry("640x480")
        self.outputText = scrolledtext.ScrolledText(self, height=29, width=76, state=tk.DISABLED)
        self.outputText.place(x=5, y=5)
        self.protocol("WM_DELETE_WINDOW", self.onClosing)
        self.master.debug.set(True)

        self.outputText.tag_configure("info", foreground="blue")
        self.outputText.tag_configure("error", foreground="red")
        self.outputText.tag_configure("warn", foreground="yellow")
        self.outputText.tag_configure("debug", foreground="purple")

    def onClosing(self):
        self.master.debug.set(False)
        self.destroy()

    def logText(self, log_type, msg):
        self.outputText.config(state=tk.NORMAL)
        if log_type == 1:
            self.outputText.insert(tk.END, "INFO: ", "info")
            self.outputText.insert(tk.END, msg + "\n")
        elif log_type == 2:
            self.outputText.insert(tk.END, "WARN: ", "warn")
            self.outputText.insert(tk.END, msg + "\n")
        elif log_type == 3:
            self.outputText.insert(tk.END, "ERROR: ", "error")
            self.outputText.insert(tk.END, msg + "\n")
        elif log_type == 4:
            self.outputText.insert(tk.END, "DEBUG: ", "debug")
            self.outputText.insert(tk.END, msg + "\n")
