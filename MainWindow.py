from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import Utils
import UtilsPSO
import UtilsFactory
import numpy
import pyswarms as ps
import Orange

class MainWindow(QMainWindow):

    def __init__(self, dataset : Orange.data.Table):
        super(MainWindow,self).__init__()
        self.dataset = dataset
        self.title = str(Utils.MAINWINDOWTITLE)
        self.layout = QFormLayout()
        self.window = QWidget()
        self.window.setWindowTitle(self.title)
        self.window.show()
        self.txtParticulas = QLineEdit
        self.slider2 = QSlider
        self.txtInercia = QLineEdit
        self.txtC1 = QLineEdit
        self.txtC2 = QLineEdit
        self.txtAlpha = QLineEdit
        self.txtVizinhanca = QLineEdit
        self.txtIteracoes = QLineEdit
        self.btnRunAlgoritm = QPushButton
        self.dataTOWindow()

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
        self.slider2 = self.createNumericSlider(0, 100) #Relevant Features
        self.layout.addWidget(self.slider2)
        labelsSlider = self.labelsFromSlider(self.slider2)
        for i in labelsSlider:
            self.layout.addWidget(i)

        #ITERACOES
        self.txtIteracoes = self.createNumericTextIndex(8, Qt.AlignLeft, 20)
        self.layout.addRow('Iteracoes: ', self.txtIteracoes)

        #BOTAO CORRER
        self.btnRunAlgoritm = QPushButton('Center')
        self.btnRunAlgoritm.setText("Run")
        self.btnRunAlgoritm.clicked.connect(self.btnClick) #CLICK DO BOTAO
        self.layout.addWidget(self.btnRunAlgoritm)

        self.window.setLayout(self.layout)

    def getValueParticle(self, part):
        return int(part.text())

    def btnClick(self):
        # DEFINICAO DO ALGORITMO PSO
        n_particles = self.getValueParticle(self.txtParticulas)
        psoArgs = {UtilsPSO.UtilsPSO.INERCIA: int(self.txtInercia.text()), UtilsPSO.UtilsPSO.C1: int(self.txtC1.text()), UtilsPSO.UtilsPSO.C2: int(self.txtC2.text()),
                   UtilsPSO.UtilsPSO.ALPHA: int(self.txtAlpha.text()), UtilsPSO.UtilsPSO.NEIGHBORS: int(self.txtVizinhanca.text()),
                   'p': 2}  # p não é relevante, visto que todas as particulas se veem umas as outras, o p representa a distancia entre cada uma das particulas

        factory = UtilsFactory.UtilsFactory()
        classificador = factory.getUtil(Utils.CLASSIFIER).getClassifier(gamma=0.01, vizinhos=5)  # CRIACAO DO CLASSIFICADOR
        psoAlgorithm = factory.getUtil(Utils.PSO).getPso(**psoArgs)
        optionsPySwarms = {'c1': psoAlgorithm.getC1(), 'c2': psoAlgorithm.getC2(), 'w': psoAlgorithm.getInercia(),
                           'k': psoArgs.get(UtilsPSO.UtilsPSO.NEIGHBORS), 'p': psoArgs.get('p')}

        dimensionsOfProblem = self.dataset.getDataset().X.shape[1]  # FEATURES DO DATASET
        initPos = Utils.createArrayInitialPos(n_particles, dimensionsOfProblem, self.slider2.value() )
        optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensionsOfProblem,
                                          options=optionsPySwarms, init_pos=initPos)
        bestCost, bestPos = optimizer.optimize(psoAlgorithm.aplicarFuncaoObjetivoTodasParticulas, 2, dataset=self.dataset,
                                               classifier=classificador, alpha=psoAlgorithm.getAlpha())

        # CONTAGEM DE QUANTAS FEATURES SAO RELEVANTES
        bestPos = Utils.listToNumpy(bestPos)
        newFeatures = numpy.count_nonzero(bestPos)
        #print(newFeatures)

        # CRIACAO DA COPIA
        deepCopy = Utils.createCloneOfReducedDataset(self.dataset, bestPos)
        print(deepCopy.getDataset().X.shape)

    def createNumericTextIndex(self, maxLength, alignment, fontSize):

        e1 = QLineEdit()
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