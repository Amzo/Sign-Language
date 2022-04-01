import tkinter as tk
from functools import partial
from threading import Thread
from tkinter import scrolledtext, ttk, filedialog

from lib.debug import LogInfo
from lib import models as ourModels


class TrainTab:
    def __init__(self, master=None, tabs=None):
        self.model = None
        self.csvFile = None
        self.saveLocation = None
        self.epochs = None
        self.rootWindow = master

        self.precision = tk.StringVar(master.rootWindow)
        self.recall = tk.StringVar(master.rootWindow)
        self.f1 = tk.StringVar(master.rootWindow)
        self.progressValue = tk.DoubleVar(master.rootWindow)
        self.saveLocation = tk.StringVar(master.rootWindow)
        self.csvFile = tk.StringVar(master.rootWindow)
        self.modelSaveLocation = tk.StringVar(master.rootWindow)

        # placement for precision under text box
        self.placement = (570 / 2.3)

        self.outputWindowLabel = tk.Label(tabs.tab2, text="Learning Output:", font="Helvetica 10 bold")
        self.outputText = scrolledtext.ScrolledText(tabs.tab2, height=20, width=70, state=tk.DISABLED)
        self.accuracyLabel = tk.Label(tabs.tab2, text="Learning Accuracy:", font="Helvetica 10 bold")
        self.precisionLabel = tk.Label(tabs.tab2, text="Precision:", font="Helvetica 10 bold")
        self.precisionTxt = tk.Label(tabs.tab2, textvariable=self.precision, font="Helvetica 10 bold")
        self.recallLabel = tk.Label(tabs.tab2, text="recall:", font="Helvetica 10 bold")
        self.recallTxt = tk.Label(tabs.tab2, textvariable=self.recall, font="Helvetica 10 bold")
        self.F1Label = tk.Label(tabs.tab2, text="F1 Score:", font="Helvetica 10 bold")
        self.F1Txt = tk.Label(tabs.tab2, textvariable=self.f1, font="Helvetica 10 bold")

        # data input for training
        self.imageInputLabel = tk.Label(tabs.tab2, text="Image Folder", font="Helvetica 10 bold")
        self.imageInputText = tk.Entry(tabs.tab2, width=30)
        self.imageFolderButton = ttk.Button(tabs.tab2, text="browse",
                                            command=partial(self.browseFolder, "directory"))

        self.csvInputLabel = tk.Label(tabs.tab2, text="CSV data file", font="Helvetica 10 bold")
        self.csvInputText = tk.Entry(tabs.tab2, width=30)
        self.csvBrowseButton = ttk.Button(tabs.tab2, text="browse",
                                          command=partial(self.browseFolder, "file"))

        self.modelLocationLbl = tk.Label(tabs.tab2, text="Model save location", font="Helvetica 10 bold")
        self.modelLocation = tk.Entry(tabs.tab2, width=30)
        self.modelButton = ttk.Button(tabs.tab2, text="Browse",
                                      command=partial(self.browseFolder, "saveDirectory"))

        self.epochLabel = tk.Label(tabs.tab2, text="Epochs: ", font="Helvetica 10 bold")
        self.epochInput = tk.Entry(tabs.tab2, width=3)
        self.progressbar = ttk.Progressbar(tabs.tab2, variable=self.progressValue, maximum=100, length=180)
        self.trainButton = ttk.Button(tabs.tab2, text="Train", command=self.runThread)

        ################################################################################################################
        # Text tags for colourful formatting
        ################################################################################################################
        self.outputText.tag_configure("info", foreground="blue")
        self.outputText.tag_configure("error", foreground="red")
        self.outputText.tag_configure("warn", foreground="yellow")
        self.outputText.tag_configure("debug", foreground="purple")

    def placeWidgets(self):
        self.outputWindowLabel.place(x=65, y=10, anchor=tk.CENTER)
        self.outputText.place(x=10, y=30)
        self.accuracyLabel.place(x=10, y=360)
        self.precisionLabel.place(x=10, y=390)
        self.precisionTxt.place(x=10, y=420)
        self.recallLabel.place(x=10 + self.placement, y=390)
        self.recallTxt.place(x=10 + self.placement, y=420)
        self.F1Label.place(x=10 + self.placement * 2, y=390)
        self.F1Txt.place(x=10 + self.placement * 2, y=420)

        # data input for training
        self.imageInputLabel.place(x=self.rootWindow.centerTrain, y=10, anchor=tk.CENTER)
        self.imageInputText.place(x=self.rootWindow.centerTrain, y=40, anchor=tk.CENTER)
        self.imageFolderButton.place(x=self.rootWindow.centerTrain, y=70, anchor=tk.CENTER)
        self.csvInputLabel.place(x=self.rootWindow.centerTrain, y=100, anchor=tk.CENTER)
        self.csvInputText.place(x=self.rootWindow.centerTrain, y=130, anchor=tk.CENTER)
        self.csvBrowseButton.place(x=self.rootWindow.centerTrain, y=170, anchor=tk.CENTER)
        self.modelLocationLbl.place(x=self.rootWindow.centerTrain, y=200, anchor=tk.CENTER)
        self.modelLocation.place(x=self.rootWindow.centerTrain, y=230, anchor=tk.CENTER)
        self.modelButton.place(x=self.rootWindow.centerTrain, y=260, anchor=tk.CENTER)
        self.epochLabel.place(x=self.rootWindow.centerTrain, y=290, anchor=tk.E)
        self.epochInput.place(x=self.rootWindow.centerTrain, y=290, anchor=tk.W)
        self.progressbar.place(x=self.rootWindow.centerTrain, y=320, anchor=tk.CENTER)
        self.trainButton.place(x=self.rootWindow.centerTrain - 15, y=350, anchor=tk.E)

    def browseFolder(self, browse_type):
        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )
        if browse_type == "directory":
            self.saveLocation.set(filedialog.askdirectory())
            self.imageInputText.insert(tk.END, self.saveLocation.get())
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                    "Updating image directory to {}".format(self.saveLocation.get()))
        if browse_type == "save":
            self.saveLocation.set(tk.filedialog.askdirectory())
        elif browse_type == "file":
            self.csvFile.set(tk.filedialog.askopenfilename(filetypes=filetypes))
            self.csvInputText.insert(tk.END, self.csvFile.get())
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                    "Selected the following csv file {}".format(self.csvFile.get()))
        elif browse_type == "saveDirectory":
            self.modelSaveLocation.set(tk.filedialog.askdirectory())
            self.modelLocation.insert(tk.END, self.modelSaveLocation.get())

    def runThread(self):
        # Don't block main thread with heavy I/O intensive tasks
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.info.value, "Running  training on new thread")

        self.epochs = int(self.epochInput.get())
        t1 = Thread(target=self.runTraining)
        t1.start()

    def runTraining(self):
        self.model = ourModels.customModel()
        self.model.createModel()
        self.model.train(log_window=self.rootWindow, model_save_dir=self.modelSaveLocation.get(), data_dir=self.saveLocation.get(),
                         csv_file=self.csvFile.get())
