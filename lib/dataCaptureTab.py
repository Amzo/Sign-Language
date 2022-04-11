import csv
import os
import tkinter as tk
from tkinter import filedialog

from functools import partial
from threading import Thread

import PIL
from PIL import ImageTk
from PIL import Image

from lib.debug import LogInfo


class DataCaptureTab:
    def __init__(self, master=None, tabs=None, sign_list=None):
        # output widgets
        self.rootClass = master
        self.signImg = object
        self.signLabelDigit = tk.StringVar(master.rootWindow)
        self.dataCount = tk.StringVar(master.rootWindow)
        self.dataCount.set("0")

        self.saveLocation = tk.StringVar(master.rootWindow)
        self.sign = tk.StringVar(master.rootWindow)
        self.sign.set("A")

        self.typeList = ["Train", "Test", "Validate"]

        self.type = tk.StringVar(master.rootWindow)
        self.type.set(self.typeList[0])

        self.videoLabel = tk.Label(tabs.tab3, borderwidth=3)
        self.signLabel = tk.Label(tabs.tab3, borderwidth=1)
        self.outputLabel = tk.Label(tabs.tab3, text="Output Label:", font='Helvetica 10 bold')
        self.saveLocationLabel = tk.Label(tabs.tab3, text="Image save location:", font='Helvetica 10 bold')

        self.labelOutput = tk.OptionMenu(tabs.tab3, self.sign, *sign_list,
                                         command=self.updateSignImage)

        self.selectedFolder = tk.Entry(tabs.tab3, textvariable=self.saveLocation)

        self.dataCountLabel = tk.Label(tabs.tab3, text="Total images collected:", font='Helvetica 10 bold')
        self.countLabel = tk.Label(tabs.tab3, textvariable=self.dataCount)

        self.captureTypeLabel = tk.Label(tabs.tab3, text="Capture Type", font="Helvetica 10 bold")

        self.captureType = tk.OptionMenu(tabs.tab3, self.type, *self.typeList)

        self.outputFolderButton = tk.Button(tabs.tab3, text="Browse",
                                            command=partial(self.browseFolder, "save", True))

        self.captureDataButton = tk.Button(tabs.tab3, text="Capture", command=self.captureDataThread)

        # Add a default image to label
        img = (Image.open("images/A.png"))
        resized_image = img.resize((100, 100), Image.ANTIALIAS)

        self.signImg = ImageTk.PhotoImage(resized_image)
        self.signLabel.configure(image=self.signImg)

    def placeWidgets(self, placement):
        self.videoLabel.grid(row=0, column=0)
        self.signLabel.place(x=placement, y=110, anchor=tk.CENTER)
        self.outputLabel.place(x=placement, y=10, anchor=tk.CENTER)
        self.saveLocationLabel.place(x=placement, y=180, anchor=tk.CENTER)
        self.labelOutput.place(x=placement, y=40, anchor=tk.CENTER)
        self.selectedFolder.place(x=placement, y=210, anchor=tk.CENTER)
        self.dataCountLabel.place(x=placement, y=270, anchor=tk.CENTER)
        self.countLabel.place(x=placement, y=300, anchor=tk.CENTER)
        self.outputFolderButton.place(x=placement, y=240, anchor=tk.CENTER)
        self.captureTypeLabel.place(x=placement, y=330, anchor=tk.CENTER)
        self.captureType.place(x=placement, y=360, anchor=tk.CENTER)
        self.captureDataButton.place(x=placement, y=420, anchor=tk.CENTER)

    def captureDataThread(self):
        t1 = Thread(target=self.captureData)
        t1.start()

    def captureData(self):
        if not os.path.isdir(self.saveLocation.get() + "/{}/{}".format(self.type.get(), self.sign.get())):
            if self.rootClass.debug.get():
                self.rootClass.debugWindow.logText(LogInfo.debug.value, "Directory {} doesn't exist.".format(
                    self.saveLocation.get()))

            try:
                os.makedirs(self.saveLocation.get() + "/{}/{}".format(self.type.get(), self.sign.get()))
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.info.value, "Created directory {}.".format(
                        self.saveLocation.get()))
            except FileNotFoundError:
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.error.value,
                                                       "Can't find the path to create the directory: {}".format(
                                                           self.saveLocation.get()))

        if os.path.isdir(self.saveLocation.get()):
            imageName = "image{}.jpg".format(self.dataCount.get())

            row = [imageName]
            try:
                for i in self.rootClass.fingerPoints:
                    row.append(i)
            except TypeError:
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.error.value,
                                                       "No Finger points found, can't write it to csv file")
            else:
                row.append(self.sign.get())

                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "writing csv file")

                with open('{}/data.csv'.format(self.saveLocation.get()), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                resizedImage = self.rootClass.saveFrame.resize((100, 100), PIL.Image.ANTIALIAS)

                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "Saving image {}/{}/{}/image{}.png".format(
                        self.saveLocation.get(), self.type.get(), self.sign.get(), self.dataCount.get()))

                resizedImage.save('{}/{}/{}/image{}.jpg'.format(self.saveLocation.get(), self.type.get(), self.signLabelDigit.get(), self.dataCount.get()))
                # update the data count
                try:
                    self.dataCount.set(str(int(self.dataCount.get()) + 1))
                except ValueError:
                    if self.rootClass.debug.get():
                        self.rootClass.debugWindow.logText(LogInfo.error.value,
                                                           "Invalid literal for data count: defaulting to 0")
                    self.dataCount.set(str(0))

    def getDataCount(self, search_dir=None):
        if search_dir is None:
            search_dir = self.saveLocation.get()

        self.dataCount.set(str(0))
        for r, d, files in os.walk(search_dir):
            for file in files:
                if file.endswith(".jpg"):
                    self.dataCount.set(str(int(self.dataCount.get()) + 1))

        if self.rootClass.debug.get():
            self.rootClass.debugWindow.logText(LogInfo.debug.value, "found {} files".format(self.dataCount.get()))

    def updateSignImage(self, *args):
        img = (Image.open("images/{}.png".format(*args)))
        self.signLabelDigit.set("{}".format(*args))

        resized_image = img.resize((100, 100), Image.ANTIALIAS)
        self.signImg = ImageTk.PhotoImage(resized_image)

        if self.rootClass.debug.get():
            self.rootClass.debugWindow.logText(LogInfo.debug.value, "Changed sign image to images/{}.png".format(*args))

        self.signLabel.configure(image=self.signImg)

    def browseFolder(self, browse_type, data_count):
        if browse_type == "save":
            self.saveLocation.set(filedialog.askdirectory())

        if data_count:
            if self.rootClass.debug.get():
                self.rootClass.debugWindow.logText(LogInfo.debug.value, "Calculating the number of files")
            self.getDataCount()
