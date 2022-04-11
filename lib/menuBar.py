import tkinter as tk


class AboutWindow:
    def __init__(self, master=None, main=None):
        self.master = main
        self.debugWindow = tk.Toplevel(master)
        self.debugWindow.title("Debug Menu")
        self.debugWindow.geometry("640x480")
        self.debugWindow.protocol("WM_DELETE_WINDOW", self.onClosing)


class MenuBar:
    def __init__(self, master=None):
        self.debugWindow = None
        self.rootWindow = master
        self.menuBar = tk.Menu(self.rootWindow.rootWindow)

        self.fileMenu = tk.Menu(self.menuBar, tearoff=0)
        self.fileMenu.add_command(label="Exit", command=master.onClosing)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)

        self.debugMenu = tk.Menu(self.menuBar, tearoff=0)
        self.debugMenu.add_command(label="Enable Debug", command=self.rootWindow.enableDebug)
        self.menuBar.add_cascade(label="Debug", menu=self.debugMenu)

        self.helpMenu = tk.Menu(self.menuBar, tearoff=0)
        self.helpMenu.add_command(label="About...", command=self.about)
        self.menuBar.add_cascade(label="Help", menu=self.helpMenu)

        master.rootWindow.configure(menu=self.menuBar)

    def about(self):
        pass
