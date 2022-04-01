import io
import socket
import socketserver
import struct
import cv2
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
from lib import models as ourModel


class VideoStreamHandler(socketserver.StreamRequestHandler):
    def handle(self, connection, commands, tabGui, rootGui, gui):
        try:
            connection = self.clientSocket.makefile('rb')
        except:
            print("Connection Failed")

        finally:
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

                    if tabGui.knn.get() and tabGui.knnLoaded:
                        points = pd.DataFrame(rootGui.fingerPoints)
                        points = points.transpose()

                        pred = tabGui.loaded_model.predict(points)

                        try:
                            pred[0]
                            results = [k for k, v in tabGui.keyMap.items() if v == pred[0]]
                            print("Got Prediction {} which is mapped to {}".format(pred[0], results[0]))
                        except IndexError:
                            pred[0] == "Stop"

                        if pred[0] in tabGui.keyMap.values():
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
