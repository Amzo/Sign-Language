import io
import socket
import socketserver
import struct
import cv2
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
from lib import models as ourModel
from lib.debug import LogInfo


class VideoStreamHandler(socketserver.StreamRequestHandler):
    def handle(self, connection, commands, tabGui, rootGui, gui):
        try:
            connection = self.clientSocket.makefile('rb')
        except:
            print("Connection Failed")

        finally:
            # use pred to save the prediction on every frame and only send the max predicted character over when the
            # list reaches a size of 24. Since we're running 24 frames per second, this is 1 prediction per second
            # and minimizes the odd incorrect prediction in that time. Trying to throw in multiple signs within a second
            # is also hard to accomplish
            pred = []

            while self.connectFlag:
                try:

                    imageLength = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

                    if not imageLength:
                        break

                    imageStream = io.BytesIO()
                    imageStream.write(connection.read(imageLength))
                    imageStream.seek(0)

                    fileBytes = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)

                    image = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)
                    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # image = cv2.resize(image, (320, 180))
                    gui.piFrame = ImageTk.PhotoImage(image=Image.fromarray(imageRGB))

                    if (tabGui.selectedModel.get() == "KNN" or tabGui.selectedModel.get() == "CNN") \
                            and tabGui.modelLoaded and rootGui.fingerPoints is not None:

                        if rootGui.debug.get():
                            rootGui.debugWindow.logText(LogInfo.debug.value, "Gathering Datapoints for prediction")

                        points = pd.DataFrame(rootGui.fingerPoints)
                        points = points.transpose()

                        # reset the datapoints as hand may have moved from out of camera
                        # this will prevent it prediction the same character based on previously
                        # saved ata points
                        rootGui.fingerPoints = None

                        tabGui.ourModel.makePrediction(check_frame=rootGui.checkFrame, points=points)

                        pred.append(tabGui.ourModel.results[0])

                        if rootGui.debug.get():
                            rootGui.debugWindow.logText(LogInfo.debug.value, "Got prediction {}".format(pred))

                        if len(pred) >= 12:
                            highest = max(pred, key=pred.count)
                            try:
                                results = [k for k, v in tabGui.keyMap.items() if v == highest]
                                print("Got Prediction {} which is mapped to {}".format(highest, results[0]))
                            except IndexError:
                                highest == "Stop"

                            if highest in tabGui.keyMap.values():
                                # Translate the keys to car commands, as these keys are already mapped on the car
                                if results[0] == "turnRight":
                                    command = "a"
                                elif results[0] == "turnLeft":
                                    command = "d"
                                elif results[0] == "forward":
                                    command = "w"
                                elif results[0] == "reverse":
                                    command = "s"
                                elif results[0] == "up":
                                    command = "i"
                                elif results[0] == "down":
                                    command = "k"
                                elif results[0] == "left":
                                    command = "j"
                                elif results[0] == "right":
                                    command = "l"
                                else:
                                    command = "t"
                                print("Sending command {}".format(command))
                                commands.send(('{0}\n'.format(command)).encode('utf-8'))

                            pred = []

                except Exception as e:
                    print(e)
                    self.connectFlag = False


class Server:
    def __init__(self, host, port, tabGui, rootGui, gui):
        self.connectFlag = None
        self.gui = gui
        self.tabs = tabGui
        self.rootGui = rootGui
        self.host = host
        self.port = port
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.commandSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def video_stream(self, videostream):
        self.clientSocket.connect((self.host, self.port))
        self.commandSocket.connect((self.host, 8080))

        self.connectFlag = True
        videostream.handle(self, self.clientSocket, self.commandSocket, self.tabs, self.rootGui, self.gui)

    def is_valid(self, buf):
        byteValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                byteValid = False
            else:
                try:
                    Image.open(io.BytesIO(buf)).verify()
                except:
                    byteValid = False
        return byteValid

    def run(self, videostream):
        self.video_stream(videostream)
