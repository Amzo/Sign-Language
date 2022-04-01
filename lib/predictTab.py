import pickle
import threading
from functools import partial

import tkinter as tk
from tkinter import messagebox


from lib import piWindow
from lib import server as ourServer


class PredictTab:
    def __init__(self, master=None, tabs=None):
        self.knnLoaded = False
        self.loaded_model = None
        self.up = None
        self.ts = None
        self.piWindow = None
        self.mainWindow = None
        self.videoStream = None
        self.rootClass = master
        self.knn = tk.BooleanVar(master.rootWindow)
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
        self.labelList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                          "U", "V", "W", "X", "Y", "Z"]

        self.videoPredLabel = tk.Label(tabs.tab1, borderwidth=3)

        self.hostLbl = tk.Label(tabs.tab1, text="Host:", font="Helvetica 9 bold")

        self.hostIp = tk.Entry(tabs.tab1, width=13)

        self.portLbl = tk.Label(tabs.tab1, text="Port:", font="Helvetica 9 bold")

        self.port = tk.Entry(tabs.tab1, width=13)

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

        self.knnCheck = tk.Checkbutton(tabs.tab1, text="Use KNN", variable=self.knn)

        self.moveButton = tk.Button(tabs.tab1, text="Move", command=self.move)

        self.connectText.set("Connect")
        self.connectButton = tk.Button(tabs.tab1, textvariable=self.connectText,
                                       command=partial(self.connect))

    def placeWidgets(self, placement):
        self.videoPredLabel.grid(row=0, column=0)
        self.hostLbl.place(x=placement, y=20, anchor=tk.E)
        self.hostIp.place(x=placement + 20, y=20, anchor=tk.W)
        self.portLbl.place(x=placement, y=50, anchor=tk.E)
        self.port.place(x=placement + 20, y=50, anchor=tk.W)
        self.connectButton.place(x=placement + 30, y=80, anchor=tk.CENTER)

        self.labelUp.place(x=placement - 21, y=120, anchor=tk.CENTER)
        self.upLabel.place(x=placement - 60, y=120, anchor=tk.E)
        self.labelDown.place(x=placement + 90, y=120, anchor=tk.CENTER)
        self.downLabel.place(x=placement + 10, y=120, anchor=tk.W)
        self.labelLeft.place(x=placement - 21, y=150, anchor=tk.CENTER)
        self.lookLeftLabel.place(x=placement - 60, y=150, anchor=tk.E)
        self.labelRight.place(x=placement + 90, y=150, anchor=tk.CENTER)
        self.lookRightLabel.place(x=placement + 10, y=150, anchor=tk.W)
        self.moveForwardLabel.place(x=placement - 60, y=200, anchor=tk.E)
        self.labelForward.place(x=placement - 21, y=200, anchor=tk.CENTER)
        self.moveBackWardsLabel.place(x=placement + 10, y=200, anchor=tk.W)
        self.labelBackwards.place(x=placement + 90, y=200, anchor=tk.CENTER)
        self.turnLeftLabel.place(x=placement - 60, y=230, anchor=tk.E)
        self.labelTurnLeft.place(x=placement - 21, y=230, anchor=tk.CENTER)
        self.turnRightLabel.place(x=placement + 10, y=230, anchor=tk.W)
        self.labelTurnRight.place(x=placement + 90, y=230, anchor=tk.CENTER)
        self.knnCheck.place(x=placement + 25, y=280, anchor=tk.W)
        self.moveButton.place(x=placement + 30, y=320, anchor=tk.CENTER)

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

    def move(self):
        try:
            self.ts.connectFlag
        except AttributeError:
            messagebox.showerror('Error', 'Connect to the remote car server')

        if len(self.keyMap) != 8:
            messagebox.showerror('Error', 'Map the keys to a character')
        else:
            self.loaded_model = pickle.load(
                open('C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\Model\\knn.pkl', 'rb'))
            self.knnLoaded = True

