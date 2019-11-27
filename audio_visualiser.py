import sys
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from multiprocessing import Queue


class App(QtGui.QMainWindow):
    def __init__(self, parent=None, audio_queue: Queue = None):
        super(App, self).__init__(parent)
        self.audio_queue = audio_queue

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        # self.view.setAspectLocked(False)
        # self.view.setRange(QtCore.QRectF(0, 0, 100, 100))

        #  image plot
        x = np.arange(200)
        y1 = np.sin(x)
        self.bg1 = pg.BarGraphItem(x=x, height=y1, width=0.3, brush='r')
        self.view.addItem(self.bg1)

        self.canvas.nextRow()
        #  line plot
        self.otherplot = self.canvas.addPlot()
        self.h2 = self.otherplot.plot(pen='y')
        # self.otherplot.setLogMode(x=True)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):

        try:
            self.ydata = np.roll(self.audio_queue.get(True), 1)[0:200]
        except Exception as ex:
            print(ex)
            self.ydata = np.sin(self.x / 3. + self.counter / 9.)

        # self.img.setImage(self.data)
        self.h2.setData(self.ydata)
        self.bg1.setOpts(y=self.ydata)

        now = time.time()
        dt = (now - self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1


def start_app(audio_queue=None):
    app = QtGui.QApplication(sys.argv)
    thisapp = App(audio_queue=audio_queue)
    thisapp.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    start_app()
