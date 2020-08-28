import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtWidgets, QtCore
import sys
import numpy as np
import ConvertingFunctions, MainFunctions


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        top = 400
        left = 400
        width = 300
        height = 200

        icon = "icons\paint.png"

        self.setWindowTitle("Paint Application")
        self.setGeometry(top, left, width, height)
        self.setWindowIcon(QIcon(icon))

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 20
        self.brushColor = Qt.green
        self.lastPoint = QPoint()
        self.mode = "Scribble"
        self.node = None
        self.square = []
        self.img = None

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        brushMenu = mainMenu.addMenu("Brush ")
        brushColor = mainMenu.addMenu("Brush Color")
        modeMenu = mainMenu.addMenu("Paint Mode")

        saveAction = QAction(QIcon("icons/save.png"), "Save", self)
        saveAction.setShortcut("Ctrl+S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        BrushSizeAction = QAction(QIcon("icons/size.png"), "Size", self)
        BrushSizeAction.setShortcut("Ctrl+B")
        brushMenu.addAction(BrushSizeAction)
        BrushSizeAction.triggered.connect(self.setBrushSize)

        BrushColorAction = QAction(QIcon("icons/black.png"), "Color", self)
        BrushColorAction.setShortcut("Ctrl+P")
        brushMenu.addAction(BrushColorAction)
        BrushColorAction.triggered.connect(self.setBrushColor)

        modeAction = QAction(QIcon("icons/brush.png"), "Mode", self)
        modeAction.setShortcut("Ctrl+M")
        modeMenu.addAction(modeAction)
        modeAction.triggered.connect(self.setMode)

        clearAction = QAction(QIcon("icons/clear.png"), "Clear", self)
        clearAction.setShortcut("Ctrl+C")
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        getMatrixAction = QAction(QIcon("icons/mat.png"), "getMatrix", self)
        getMatrixAction.setShortcut("Ctrl+X")
        fileMenu.addAction(getMatrixAction)
        getMatrixAction.triggered.connect(self.getMatrix)

        redAction = QAction(QIcon("icons/red.png"), "G inputs", self)
        redAction.setShortcut("Ctrl+R")
        brushColor.addAction(redAction)
        redAction.triggered.connect(self.redColor)

        greenAction = QAction(QIcon("icons/green.png"), "S inputs", self)
        greenAction.setShortcut("Ctrl+G")
        brushColor.addAction(greenAction)
        greenAction.triggered.connect(self.greenColor)

    def redColor(self):
        self.brushColor = Qt.red

    def greenColor(self):
        self.brushColor = Qt.green

    def reSetSquare(self):
        self.square = []

    def addSquare(self, point):
        if len(self.square) < 2:
            self.square.append(point)
        else:
            self.reSetSquare()
            self.append(point)

    def setMode(self):
        items = ("Scribble", "Line", "Square", "Circle")
        item, okPressed = QtWidgets.QInputDialog.getItem(self, "Get item", "Mode:", items, 0, False)
        if okPressed and item:
            self.mode = item
            self.node = None

    def setBrushColor(self):
        newColor = QtWidgets.QColorDialog.getColor(self.brushColor)
        if newColor.isValid():
            self.brushColor = newColor

    def setBrushSize(self):
        newSize, success = QtWidgets.QInputDialog.getInt(self, "Scribble", "Select pen width:",
                                                         self.brushSize, 1, 50, 1)
        if success:
            self.brushSize = newSize

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

            position = event.pos()
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if self.mode == "Line":
                if self.node is not None:
                    painter.drawLine(self.node, position)
                self.node = position
                self.drawing = False
            elif self.mode == "Square":
                if len(self.square) < 2:
                    self.addSquare(position)
                else:
                    painter.drawRect(QtCore.QRect(self.square[0], self.square[1]))
                    self.reSetSquare()

                self.drawing = False
        self.update()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftToolBarArea) & self.drawing:
            position = event.pos()
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if self.mode == "Scribble":
                painter.drawLine(self.lastPoint, position)
                # print(QPoint(position))
                self.lastPoint = position
            self.update()

        elif (event.buttons() & Qt.RightToolBarArea):
            self.reSetSquare()
            self.node = None
            self.drawing = False

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def save(self):
        filePath, filter_ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                        "PNG(*.png);; JPEG(*.jpg *.jpeg);; ALL Files(*.*)")
        if filePath == "":
            return
        self.image.save(filePath)

    def default_save(self):
        filePath = './new_trajectory.png'
        if filePath == "":
            return
        self.image.save(filePath)

    def clear(self):
        self.image.fill(Qt.white)
        self.node = None
        self.update()

    def getMatrix(self, show_=False):
        img = self._QImage2numpy(self.image)
        if show_:
            plt.figure()
            plt.imshow(img)
            plt.show()
        self.img = img

    # Convert QImage to numpy array
    def _QImage2numpy(self, src):
        b = src.bits()
        b.setsize(self.height() * self.width() * 4)
        arr = np.frombuffer(b, np.uint8).reshape((self.height(), self.width(), 4))
        arr = arr[:, :, [2, 1, 0, 3]]  # No clue why, but QImage seems to be not RGB but BGR
        return (arr)


def create_trajectory(show_=False):
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()

    window.getMatrix()
    frame_img = window.img

    window.default_save()
    # Green
    Sresult = ConvertingFunctions.reduce_matrix(frame_img[:, :, 0], 10)

    plt.imshow(frame_img[:, :, 0])
    plt.title("Trajectory in the pyramidal cells")
    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.show()

    if show_:
        plt.imshow(Sresult)
        plt.title("S legend")

    np.save('S_inputs.npy', Sresult)

    result_S = MainFunctions.create_trajectory_matrix('S_inputs.npy')

    if np.count_nonzero(frame_img[:, :, 1] != 255) > 0:
        # Red
        Gresult = ConvertingFunctions.reduce_matrix(frame_img[:, :, 1], 10)
        np.save('G_inputs.npy', Gresult)
        result_G = MainFunctions.create_trajectory_matrix('G_inputs.npy')

        plt.imshow(frame_img[:, :, 1])
        plt.title("Special input in the pyramidal cells")
        plt.xlabel("meters")
        plt.ylabel("meters")
        plt.show()

        if show_:
            plt.imshow(Gresult)
            plt.title("G legend")

        return result_S, result_G

    else:
        return result_S


def create_trajectory_and_inputs(show_=False):
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()

    window.getMatrix()
    frame_img = window.img

    window.default_save()
    # Green
    Sresult = ConvertingFunctions.reduce_matrix(frame_img[:, :, 0], 10)

    plt.imshow(frame_img[:, :, 0])
    plt.title("Trajectory in the pyramidal cells")
    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.show()

    if show_:
        plt.imshow(Sresult)
        plt.title("S legend")

    np.save('S_inputs.npy', Sresult)

    result_S = MainFunctions.create_trajectory_matrix('S_inputs.npy')

    if np.count_nonzero(frame_img[:, :, 1] != 255) == 0:
        raise Warning("With this method, must add external inputs!")
    else:
        # Red
        Gresult = ConvertingFunctions.reduce_matrix(frame_img[:, :, 1], 10)
        np.save('G_inputs.npy', Gresult)
        result_G = MainFunctions.create_trajectory_matrix('G_inputs.npy')

        plt.imshow(frame_img[:, :, 1])
        plt.title("Special input in the pyramidal cells")
        plt.xlabel("meters")
        plt.ylabel("meters")
        plt.show()

        if show_:
            plt.imshow(Gresult)
            plt.title("G legend")

        result_S = result_S + result_G

        return result_S, result_G


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()
    frame_img = window.img
    window.default_save()
    # Red
    Gresult = ConvertingFunctions.reduce_matrix(frame_img[:, :, 1], 10)
    # Green
    Sresult = ConvertingFunctions.reduce_matrix(frame_img[:, :, 0], 10)

    plt.imshow(Gresult)
    plt.title("G results")

    plt.imshow(Sresult)
    plt.title("S legend")

    np.save('G_inputs.npy', Gresult)
    np.save('S_inputs.npy', Sresult)
