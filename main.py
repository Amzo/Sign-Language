#!/usr/bin/env python3
from typing import Any

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

from lib import gui


ourGUI = gui.Gui()
webCamera = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

ourGUI.updateWindow()
threadRunning = True

while threadRunning:
    success, frame = webCamera.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    tmpList: list[list[int | Any]] = []

    if results.multi_hand_landmarks:
        for handMarks in results.multi_hand_landmarks:
            for index, landMark in enumerate(handMarks.landmark):
                tmpList.append(landMark.x)
                tmpList.append(landMark.y)
            mpDraw.draw_landmarks(frame, handMarks, mpHands.HAND_CONNECTIONS)
        ourGUI.fingerPoints = tmpList

    assert isinstance(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), object)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:
        ourGUI.imgFrame = ImageTk.PhotoImage(image=img)
        ourGUI.predFrame = ImageTk.PhotoImage(image=Image.fromarray(imgRGB))
        ourGUI.saveFrame = Image.fromarray(imgRGB)
        ourGUI.checkFrame = cv2.resize(imgRGB, (100, 100))
    except (AttributeError, RuntimeError):
        # tkinter is on a separate thread if these attribute fails, or runtime error
        # then the thread is dead
        threadRunning = not threadRunning
