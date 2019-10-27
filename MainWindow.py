from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import Utils

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow,self).__init__(*args, **kwargs)
        self.title = str(Utils.MAINWINDOWTITLE)
        self.layout = QFormLayout()
        self.window = QWidget()
        self.window.setWindowTitle(self.title)
        self.dataTOWindow()
        self.window.show()
        self.txtParticulas = None
        self.slider2 = None
        self.txtInercia = None
        self.txtC1 = None
        self.txtC2 = None
        self.txtAlpha = None
        self.txtVizinhanca = None
        self.txtIteracoes = None
        self.btnRunAlgoritm = None

    def dataTOWindow(self):

        #PARTICULAS
        self.txtParticulas = self.createNumericTextIndex(8, Qt.AlignLeft, 20)
        self.layout.addRow('N Particulas: : ', self.txtParticulas)

        #INERCIA
        self.txtInercia = self.createNumericTextIndex(3, Qt.AlignLeft, 20)
        self.layout.addRow('Inercia: ', self.txtInercia)

        #C1
        self.txtC1 = self.createNumericTextIndex(4, Qt.AlignLeft, 20)
        self.layout.addRow('c1: ', self.txtC1)

        #C2
        self.txtC2 = self.createNumericTextIndex(4, Qt.AlignLeft, 20)
        self.layout.addRow('c2: ', self.txtC2)

        #ALPHA
        self.txtAlpha = self.createNumericTextIndex(3, Qt.AlignLeft, 20)
        self.layout.addRow('alpha: ', self.txtAlpha)

        #VIZINHANCA
        self.txtVizinhanca = self.createNumericTextIndex(6,Qt.AlignLeft,20)
        self.layout.addRow('Vizinhanca: ', self.txtVizinhanca)

        #SLIDER RELEVANT FEATURES
        self.layout.addWidget(QLabel('Numero de Features relevantes - init_pos'))
        slider2 = self.createNumericSlider(0, 100) #Relevant Features
        self.layout.addWidget(slider2)
        labelsSlider = self.labelsFromSlider(slider2)
        for i in labelsSlider:
            self.layout.addWidget(i)

        #ITERACOES
        self.txtIteracoes = self.createNumericTextIndex(8, Qt.AlignLeft, 20)
        self.layout.addRow('Iteracoes: ', self.txtIteracoes)

        #BOTAO CORRER
        self.btnRunAlgoritm = QPushButton('Center')
        self.btnRunAlgoritm.setText("Run")
        self.layout.addWidget(self.btnRunAlgoritm)

        self.window.setLayout(self.layout)

    def createNumericTextIndex(self, maxLength, alignment, fontSize):

        e1 = QLineEdit()
        e1.setValidator(QIntValidator())
        e1.setMaxLength(maxLength)
        e1.setAlignment(alignment)
        e1.setFont(QFont("Arial", fontSize))

        return e1

    def createNumericSlider(self, minValue, maxValue):

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minValue)
        slider.setMaximum(maxValue)
        slider.setValue((maxValue-minValue)/2)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)

        return slider

    def labelsFromSlider(self, slider : QSlider):

        label1 = QLabel()
        label1.setAlignment(Qt.AlignLeft)
        label1.setText(str(slider.minimum()))
        label2 = QLabel()
        label2.setAlignment(Qt.AlignCenter)
        label2.setText(str(slider.value()))
        label3 = QLabel()
        label3.setAlignment(Qt.AlignRight)
        label3.setText(str(slider.maximum()))

        list =[label1,label2, label3]

        return list