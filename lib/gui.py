#!/usr/bin/env python3

import threading
import tkinter as tk
from tkinter import ttk, StringVar

from lib import predictTab, dataCaptureTab, menuBar, debug, trainTab


class TabWidget:
    def __init__(self, master=None):
        self.tabControl = ttk.Notebook(master)
        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tab3 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text='Predict')
        self.tabControl.add(self.tab2, text='Train')
        self.tabControl.add(self.tab3, text='Data Capture')
        self.tabControl.pack(expand=1, fill="both")


class Gui(threading.Thread):
    dataCount: StringVar

    def __init__(self):
        threading.Thread.__init__(self)
        self.ts = None
        self.piWindow = None
        self.trainTab = None
        self.tab = None
        self.dataTab = None
        self.predictTab = None
        self.tabsObjects = None
        self.menuBar = None
        self.debugWindow = None
        self.debug = None
        self.model = None
        self.modelLoaded = None

        # Frame data for each widget
        self.imgFrame = None
        self.signImg = None
        self.predFrame = None

        self.fingerPoints = None

        # calculate the remaining width of the window based on camera size to center widgets
        self.rootWindow = None
        self.remainingWidth = 880 - 640
        self.center = 880 - (self.remainingWidth / 2)
        self.centerTrain = 880 - ((880 - 570) / 2)
        self.start()

    def run(self):
        self.rootWindow = tk.Tk()
        self.rootWindow.title("KF6007: Artificial Intelligence and Robotics")
        self.rootWindow.geometry("880x520")
        self.rootWindow.resizable(False, False)

        self.debug = tk.BooleanVar(self.rootWindow)
        self.debug.set(False)

        self.modelLoaded = tk.BooleanVar(self.rootWindow)
        self.modelLoaded.set(False)

        self.menuBar = menuBar.MenuBar(master=self)

        self.tabsObjects = TabWidget(master=self.rootWindow)

        #########################################################################################
        #                                                                                       #
        #                                   Prediction tab widgets                              #
        #                                                                                       #
        #########################################################################################

        self.predictTab = predictTab.PredictTab(master=self, tabs=self.tabsObjects)
        self.predictTab.placeWidgets(self.center)

        #########################################################################################
        #                                                                                       #
        #                                   Data Capture Widgets                                #
        #                                                                                       #
        #########################################################################################

        self.dataTab = dataCaptureTab.DataCaptureTab(master=self, tabs=self.tabsObjects,
                                                     sign_list=self.predictTab.labelList)
        self.dataTab.placeWidgets(self.center)

        #########################################################################################
        #                                                                                       #
        #                                    Training Widgets                                   #
        #                                                                                       #
        #########################################################################################

        self.trainTab = trainTab.TrainTab(master=self, tabs=self.tabsObjects)
        self.trainTab.placeWidgets()

        # event to handle tab change
        self.tabsObjects.tabControl.bind('<<NotebookTabChanged>>', self.tabChanged)
        self.rootWindow.protocol("WM_DELETE_WINDOW", self.onClosing)

        self.rootWindow.mainloop()

    def onClosing(self):
        self.rootWindow.quit()

    def enableDebug(self):
        self.debugWindow = debug.DebugWindow(master=self.rootWindow, main=self)

    def tabChanged(self, event):
        self.tab = event.widget.tab('current')['text']

    def updateWindow(self):
        assert isinstance(self.imgFrame, object)
        assert isinstance(self.signImg, object)

        if self.tab == "Data Capture":
            self.dataTab.videoLabel.configure(image=self.imgFrame)
        elif self.tab == "Predict":
            self.predictTab.videoPredLabel.configure(image=self.predFrame)
            try:
                self.piWindow.updatePiWindow()
                # Window is now closed, add cleanup code later
                pass
            except AttributeError:
                # The video streaming window hasn't been created yet
                pass
        try:
            if self.trainTab.ts.connectFlag:
                self.trainTab.connectText.set("Connected")
                self.trainTab.connectButton.configure(state=tk.DISABLED)
            else:
                self.trainTab.connectText.set("Connect")
                self.trainTab.connectButton.configure(state=tk.NORMAL)
        except AttributeError:
            # Not started yet
            pass

        self.rootWindow.after(1, self.updateWindow)

