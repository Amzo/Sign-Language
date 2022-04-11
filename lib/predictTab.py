import os.path
import pickle
import threading
from functools import partial

import tkinter as tk
from tkinter import messagebox, filedialog

from lib import piWindow
from lib import server as ourServer
from lib.debug import LogInfo
from lib import models as ourModels


class PredictTab:
    def __init__(self, master=None, tabs=None):
        self.model = None
        self.modelLoaded = False
        self.loaded_model = None
        self.up = None
        self.ts = None
        self.piWindow = None
        self.mainWindow = None
        self.videoStream = None
        self.rootClass = master
        self.up = tk.StringVar(master.rootWindow)
        self.down = tk.StringVar(master.rootWindow)
        self.left = tk.StringVar(master.rootWindow)
        self.right = tk.StringVar(master.rootWindow)
        self.forward = tk.StringVar(master.rootWindow)
        self.reverse = tk.StringVar(master.rootWindow)
        self.turnLeft = tk.StringVar(master.rootWindow)
        self.turnRight = tk.StringVar(master.rootWindow)
        self.connectText = tk.StringVar(master.rootWindow)
        self.keyMap = {}

        self.selectedModel = tk.StringVar(master.rootWindow)
        self.modelList = ['CNN', 'KNN']
        self.selectedModel.set(self.modelList[0])

        self.modelLocation = tk.StringVar(master.rootWindow)

        self.labelList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                          "U", "V", "W", "X", "Y", "Z"]

        self.connectFrame = tk.LabelFrame(tabs.tab1, text="Car Connection", height=130, width=220)
        self.cameraFrame = tk.LabelFrame(tabs.tab1, text="Camera Control", height=120, width=220)
        self.carFrame = tk.LabelFrame(tabs.tab1, text="Car Control", height=110, width=220)
        self.modelFrame = tk.LabelFrame(tabs.tab1, text="Model Select", height=70, width=220)

        self.videoPredLabel = tk.Label(tabs.tab1, borderwidth=3)
        self.hostLbl = tk.Label(tabs.tab1, text="Host:", font="Helvetica 9 bold")
        self.hostIp = tk.Entry(tabs.tab1, width=18)
        self.portLbl = tk.Label(tabs.tab1, text="Port:", font="Helvetica 9 bold")
        self.port = tk.Entry(tabs.tab1, width=18)
        self.labelUp = tk.OptionMenu(tabs.tab1, self.up, *self.labelList,
                                     command=partial(self.updateList, "up"))
        self.upLabel = tk.Label(tabs.tab1, text="Up", font="Helvetica 9 bold")
        self.labelDown = tk.OptionMenu(tabs.tab1, self.down, *self.labelList,
                                       command=partial(self.updateList, "down"))
        self.downLabel = tk.Label(tabs.tab1, text="Down", font="Helvetica 9 bold")
        self.labelLeft = tk.OptionMenu(tabs.tab1, self.left, *self.labelList,
                                       command=partial(self.updateList, "left"))
        self.lookLeftLabel = tk.Label(tabs.tab1, text="Left", font="Helvetica 9 bold")
        self.labelRight = tk.OptionMenu(tabs.tab1, self.right, *self.labelList,
                                        command=partial(self.updateList, "right"))
        self.lookRightLabel = tk.Label(tabs.tab1, text="Right", font="Helvetica 9 bold")
        self.moveForwardLabel = tk.Label(tabs.tab1, text="Forward", font="Helvetica 9 bold")
        self.labelForward = tk.OptionMenu(tabs.tab1, self.forward, *self.labelList,
                                          command=partial(self.updateList, "forward"))
        self.moveBackWardsLabel = tk.Label(tabs.tab1, text="Reverse", font="Helvetica 9 bold")
        self.labelBackwards = tk.OptionMenu(tabs.tab1, self.reverse, *self.labelList,
                                            command=partial(self.updateList, "reverse"))
        self.turnLeftLabel = tk.Label(tabs.tab1, text="Left", font="Helvetica 9 bold")
        self.labelTurnLeft = tk.OptionMenu(tabs.tab1, self.turnLeft, *self.labelList,
                                           command=partial(self.updateList, "turnLeft"))
        self.turnRightLabel = tk.Label(tabs.tab1, text="Right", font="Helvetica 9 bold")
        self.labelTurnRight = tk.OptionMenu(tabs.tab1, self.turnRight, *self.labelList,
                                            command=partial(self.updateList, "turnRight"))

        self.modelSelect = tk.Label(tabs.tab1, text="Model:", font="Helvetica 9 bold")
        self.modelDropDown = tk.OptionMenu(tabs.tab1, self.selectedModel, *self.modelList)

        self.modelBrowseButton = tk.Button(tabs.tab1, text="Browse", command=self.modelBrowse)
        self.moveButton = tk.Button(tabs.tab1, text="Move", command=self.move)
        self.connectText.set("Connect")
        self.connectButton = tk.Button(tabs.tab1, textvariable=self.connectText,
                                       command=partial(self.connect))

    def placeWidgets(self, placement):
        self.videoPredLabel.grid(row=0, column=0)
        self.connectFrame.place(x=650, y=5)

        self.hostLbl.place(x=placement - 50, y=40, anchor=tk.E)
        self.hostIp.place(x=placement - 30, y=40, anchor=tk.W)
        self.portLbl.place(x=placement - 50, y=70, anchor=tk.E)
        self.port.place(x=placement - 30, y=70, anchor=tk.W)
        self.connectButton.place(x=placement, y=110, anchor=tk.CENTER)

        self.cameraFrame.place(x=650, y=135)
        self.labelUp.place(x=placement - 31, y=180, anchor=tk.CENTER)
        self.upLabel.place(x=placement - 70, y=180, anchor=tk.E)
        self.labelDown.place(x=placement + 80, y=180, anchor=tk.CENTER)
        self.downLabel.place(x=placement, y=180, anchor=tk.W)
        self.labelLeft.place(x=placement - 31, y=220, anchor=tk.CENTER)
        self.lookLeftLabel.place(x=placement - 70, y=220, anchor=tk.E)
        self.labelRight.place(x=placement + 80, y=220, anchor=tk.CENTER)
        self.lookRightLabel.place(x=placement, y=220, anchor=tk.W)

        self.carFrame.place(x=650, y=255)

        self.moveForwardLabel.place(x=placement - 50, y=295, anchor=tk.E)
        self.labelForward.place(x=placement - 31, y=295, anchor=tk.CENTER)
        self.moveBackWardsLabel.place(x=placement, y=295, anchor=tk.W)
        self.labelBackwards.place(x=placement + 80, y=295, anchor=tk.CENTER)
        self.turnLeftLabel.place(x=placement - 70, y=335, anchor=tk.E)
        self.labelTurnLeft.place(x=placement - 31, y=335, anchor=tk.CENTER)
        self.turnRightLabel.place(x=placement, y=335, anchor=tk.W)
        self.labelTurnRight.place(x=placement + 80, y=335, anchor=tk.CENTER)

        self.modelFrame.place(x=650, y=365)
        self.modelSelect.place(x=placement - 60, y=405, anchor=tk.E)
        self.modelDropDown.place(x=placement + 20, y=405, anchor=tk.E)
        self.modelBrowseButton.place(x=placement + 40, y=405, anchor=tk.W)
        self.moveButton.place(x=placement, y=460, anchor=tk.CENTER)

    def connect(self):
        self.rootClass.piWindow = piWindow.PiWindow(self.rootClass.rootWindow)
        self.mainWindow = self.rootClass
        h = str(self.hostIp.get())
        p1 = int(self.port.get())
        self.videoStream = ourServer.VideoStreamHandler
        self.ts = ourServer.Server(h, p1, self, self.mainWindow, self.rootClass.piWindow)

        T = threading.Thread(target=self.ts.run, args=(self.videoStream,), daemon=True)
        T.start()

    def updateList(self, who, *args):
        value = str(*args[0])
        print(who, value)

        if value in self.keyMap.values():
            messagebox.showerror('Error', 'this character is already bound')
            print(self.keyMap)
        else:
            self.keyMap[who] = value

    def modelBrowse(self):
        filetypes = (
            ('Pickle files', '*.pkl'),
        )

        if self.selectedModel.get() == "CNN":
            self.modelLocation.set(filedialog.askdirectory())

            if os.path.isdir(self.modelLocation.get() + "/assets") and os.path.isdir(
                    self.modelLocation.get() + "/variables"):
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "Found CNN Model")

        elif self.selectedModel.get() == "KNN":
            self.modelLocation.set(filedialog.askopenfilename(filetypes=filetypes))

            if os.path.isfile(self.modelLocation.get()):
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "Found KNN Model")

    def move(self):
        try:
            self.ts.connectFlag
        except AttributeError:
            messagebox.showerror('Error', 'Connect to the remote car server')

        if len(self.keyMap) != 8:
            messagebox.showerror('Error', 'Map the keys to a character')
        else:
            if self.selectedModel.get() == "KNN":
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "Loading KNN Model from {}".format(self.modelLocation.get()))

                self.loaded_model = pickle.load(
                    open(self.modelLocation.get(), 'rb'))

                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "Switching modelLoaded to True")

                self.modelLoaded = True
            elif self.selectedModel.get() == "CNN":
                self.loaded_model = ourModels.loadModel()
                self.modelLoaded = True
