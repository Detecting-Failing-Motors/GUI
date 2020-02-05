from PyQt5.QtGui import *  
from PyQt5.QtGui import QValidator
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
from functions import *
import os
import sip
import csv


from PyQt5.uic import loadUiType
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
message = ""
"""
These RegExp are used to check for user input correctness

"""
regexp_checkCSV = QRegExp('^\/([A-z0-9-_+\s]+\/)*([A-z0-9]+\.(csv))$')
validator = QRegExpValidator(regexp_checkCSV)

regexp_checkint = QIntValidator()
intvalidator = QRegExpValidator(regexp_checkint)

regexp_checkdouble = QDoubleValidator()
doublevalidator = QRegExpValidator(regexp_checkdouble)
	
Ui_MainWindow, QMainWindow = loadUiType('MQAAdraft4_3.ui')

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ): #provide values for attributes at runtime
        super(Main, self).__init__()
        self.setupUi(self)
        self.pushBrowse.clicked.connect(self.selectFile)
        self.pushBrowse_2.clicked.connect(self.selectmlFile)
        self.pushApply.clicked.connect(self.apply)
        self.pushRun.clicked.connect(self.run)
        self.HomeDirectory = os.getcwd() #saves the primary working directory
        self.directory = os.listdir(self.HomeDirectory)
        self.saveBttn.clicked.connect(self.file_save)
        self.actionOpen.triggered.connect(self.file_open)
        self.actionReset.triggered.connect(self.plots_close)
        self.message = 0
        self.UI = []
        self.reset = 0
        self.plotinfo = []
        self.plot2info = []
        self.plot3info = []
        self.plot4info = []
        self.plot5info = []
        self.plot6info = []
        
        
        #check User input for correct .csv file
        self.inputFile.setValidator(validator)
        self.inputFile.textChanged.connect(self.check_state)
        self.inputFile.textChanged.emit(self.inputFile.text())
        #check User input for correct .csv file for ml
        self.mlData.setValidator(validator)
        self.mlData.textChanged.connect(self.check_state)
        self.mlData.textChanged.emit(self.mlData.text())
        #shaft speed
        self.shaftSpeed.setValidator(regexp_checkdouble)
        self.shaftSpeed.textChanged.connect(self.check_state)
        self.shaftSpeed.textChanged.emit(self.shaftSpeed.text())
        #Num of rolling elements
        self.numberofElements.setValidator(regexp_checkint)
        self.numberofElements.textChanged.connect(self.check_state)
        self.numberofElements.textChanged.emit(self.numberofElements.text())
        #diameter of rolling elements
        self.diameterofElements.setValidator(regexp_checkdouble)
        self.diameterofElements.textChanged.connect(self.check_state)
        self.diameterofElements.textChanged.emit(self.diameterofElements.text())
        #pitch diameter
        self.pitchDiameter.setValidator(regexp_checkdouble)
        self.pitchDiameter.textChanged.connect(self.check_state)
        self.pitchDiameter.textChanged.emit(self.pitchDiameter.text())
        #Contact angle
        self.contactAngle.setValidator(regexp_checkdouble)
        self.contactAngle.textChanged.connect(self.check_state)
        self.contactAngle.textChanged.emit(self.contactAngle.text())
        #Frequency
        self.samFreq.setValidator(regexp_checkdouble)
        self.samFreq.textChanged.connect(self.check_state)
        self.samFreq.textChanged.emit(self.samFreq.text())
        
        
        
   
    def check_state(self, *args, **kwargs): #this function is changes the color of the lineedit fields depending on state
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        if state == QValidator.Acceptable:
            color = '#c4df9b' # green
        elif state == QtGui.QValidator.Intermediate:
            color = '#fff79a' # yellow
        else:
            color = '#f6989d' # red
        sender.setStyleSheet('QLineEdit { background-color: %s }' % color)
        
    #creates a dictionary from the saved CSV 
    def file_open(self): #function called when the open file action in triggered. Creates a dictionary from a CSV file.
        filename = QFileDialog.getOpenFileName()[0]
        reader = csv.reader(open(filename, 'r'))
        d = {}
        for row in reader:
            k, v = row
            d[k] = v
                   
        print(d)
        
        self.setTextInfile(d)
        return True
    #used the dicitonary created above to assign saved variables to input parameters
    def setTextInfile(self, d):
        self.inputName.setText(d['inputName'])
        self.inputApplication.setText(d['inputApplication'])
        self.inputModelnum.setText(d['inputModelnum'])
        self.inputSavingalias.setText(d['inputSavingalias'])
        self.inputFile.setText(d['inputFile'])
        self.mlData.setText(d['mlData'])
        self.horsepower.setText(d['horsepower'])
        self.voltage.setText(d['voltage'])
        self.phase.setText(d['phase'])
        self.shaftnum.setText(d['shaftnum'])
        self.shaftSpeed.setText(d['shaftSpeed'])
        self.numberofElements.setText(d['numberofElements'])
        self.diameterofElements.setText(d['diameterofElements'])
        self.pitchDiameter.setText(d['pitchDiameter'])
        self.contactAngle.setText(d['contactAngle'])
        self.samFreq.setText(d['samFreq'])
        

    
    """
    Hmm i wonder if this is used to save the file?
    """
    def file_save(self,): #called when the save btn is clicked. converts user input to dictionary then to dataframe then to csv file. 
        dict = CreateSaveDictionary(self.inputName.text(),self.inputApplication.text(),self.inputModelnum.text(),self.inputSavingalias.text(),self.inputFile.text(),self.mlData.text(), self.horsepower.text(), self.voltage.text(), self.phase.text(), self.shaftnum.text(),self.shaftSpeed.text(),self.numberofElements.text(), self.diameterofElements.text(), self.pitchDiameter.text(), self.contactAngle.text(), self.samFreq.text())
        CreateCSVfromDict(dict)
    """
    These functions create the figure widgets and toolbars
    """    
    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        #self.canvas.setParent(None)
        self.spectrumUI.addWidget(self.canvas)
        #self.canvas.draw()
        #self.toolbar = NavigationToolbar(self.canvas,self.mainspectrum, coordinates=True)
        #self.toolbar.setParent(None)
        #self.spectrumUI.addWidget(self.toolbar)
        
        
    def addgraph11(self, fig):
        self.canvas1 = FigureCanvas(fig)
        self.graph11UI.addWidget(self.canvas1)
        #self.canvas1.draw()
        self.toolbar1 = NavigationToolbar(self.canvas1,self.graph11, coordinates=True)
        self.graph11UI.addWidget(self.toolbar1)
        
    def addgraph12(self, fig):
        self.canvas2 = FigureCanvas(fig)
        self.graph12UI.addWidget(self.canvas2)
        #self.canvas2.draw()
        self.toolbar2 = NavigationToolbar(self.canvas2, 
                self.graph12, coordinates=True)
        self.graph12UI.addWidget(self.toolbar2)
        
    def addgraph13(self, fig):
        self.canvas5 = FigureCanvas(fig)
        self.graph13UI.addWidget(self.canvas5)
        #self.canvas2.draw()
        self.toolbar5 = NavigationToolbar(self.canvas5, 
                self.graph13, coordinates=True)
        self.graph13UI.addWidget(self.toolbar5)
        
    def addgraph21(self, fig):
        self.canvas3 = FigureCanvas(fig)
        self.graph21UI.addWidget(self.canvas3)
        #self.canvas3.draw()
        self.toolbar3 = NavigationToolbar(self.canvas3, 
                self.graph21, coordinates=True)
        self.graph21UI.addWidget(self.toolbar3)
        
    def addgraph22(self, fig):
        self.canvas4 = FigureCanvas(fig)
        self.graph22UI.addWidget(self.canvas4)
        #self.canvas4.draw()
        self.toolbar4 = NavigationToolbar(self.canvas4, 
                self.graph22, coordinates=True)
        self.graph22UI.addWidget(self.toolbar4)

       
    #clearly selects file
    def selectFile(self,):
        self.inputFile.setText(QFileDialog.getOpenFileName()[0])
        self.inputfile = self.inputFile.text()
    #selects file but for a ml data   
    def selectmlFile(self,):
        self.mlData.setText(QFileDialog.getOpenFileName()[0])
        
    """
    Apply checks user inputs and then assigns them to function parameter variables
    In case the user doesn't supply input for a specific field, default inputs will
    be inserted.
    """    
    def apply(self,):
        if self.inputSavingalias.text() != "":
           self.savingalias = self.inputSavingalias.text()+".csv"
           self.inputSavingalias.setText(self.savingalias)
            
        if self.inputFile.text() != "":
            try:
                temp = self.inputFile.text()
                self.FileOfInterest = self.inputFile.text()
            except: 
                print("pu")
        else:
            print("no input file selected, using demo file.")
            self.FileOfInterest = "AccelerometerActualData.csv"
            self.inputFile.setText("/AccelerometerActualData.csv")
            
        if self.mlData.text() != "":
            self.TrainingDataFile = self.mlData.text() 
            print(self.TrainingDataFile)
        else:
            print("no ML data selected")
            self.TrainingDataFile = "NoNegatives.csv" #default file location
            self.mlData.setText("/NoNegatives.csv")
            
        if self.shaftSpeed.text() != "":
            self.n = float(self.shaftSpeed.text())
            print("type n =",type(self.n))
        else:
            print("no shaft speed selected")
            self.n = 2000/60
            self.shaftSpeed.setText(str(self.n))
            
            
        if self.numberofElements.text() != "":
            self.N = int(self.numberofElements.text())
        else:
            print("Number of elements not specified")
            self.N = 16
            self.numberofElements.setText(str(self.N))
            
        if self.diameterofElements.text() != "":
            self.Bd = float(self.diameterofElements.text())   
        else:
            print("Diameter of elements not specified")
            self.Bd = 0.331*254
            self.diameterofElements.setText(str(self.Bd))
            
        if self.pitchDiameter.text() != "":
            self.Pd = float(self.pitchDiameter.text())
        else:
            print("no pitch diameter specified")
            self.Pd = Pd = 2.815*254
            self.pitchDiameter.setText(str(self.Pd))
            
        if self.contactAngle.text() != "":
            self.phi = float(self.contactAngle.text())
        else:
            print("Contact angle not specified")
            self.phi = (15.17*3.14159)/180
            self.contactAngle.setText(str(self.phi))
        
        if self.samFreq.text() != "":
            self.SampleFrequency = float(self.samFreq.text())
           
        else:
            print("no sample frequency specified")
            self.SampleFrequency = 20000
            self.samFreq.setText(str(self.SampleFrequency))
            
        self.popup = MyPopup("Applied")
        self.popup.setGeometry(QtCore.QRect(100, 100, 400, 200))
        self.popup.show()
        

 ##############################################       
    def getPlot(self,X,Y,xlabel,ylabel,Title):
        if self.reset != 0:
            self.sub0.cla()
        self.sub0.plot(X,Y,c = np.random.rand(3,))
        self.sub0.set_xlabel(xlabel, fontsize=12)
        self.sub0.set_ylabel(ylabel, fontsize=12)
        self.sub0.set_title(Title)
        self.sub0.grid(True)
        if self.reset !=0:    
            self.canvas.draw()
        
    
        return True 
    def getPlot1(self,X,Y,xlabel,ylabel,Title):
        if self.reset != 0:
            self.sub1.cla()
        self.sub1.plot(X,Y,c = np.random.rand(3,))
        self.sub1.set_xlabel(xlabel, fontsize=12)
        self.sub1.set_ylabel(ylabel, fontsize=12)
        self.sub1.set_title(Title)
        self.sub1.grid(True)
        if self.reset !=0:    
            self.canvas1.draw()
            
        
    
        return True
    def getPlot2(self,X,Y,xlabel,ylabel,Title):
        if self.reset != 0:
            self.sub2.cla()
        self.sub2.plot(X,Y,c = np.random.rand(3,))
        self.sub2.set_xlabel(xlabel, fontsize=12)
        self.sub2.set_ylabel(ylabel, fontsize=12)
        self.sub2.set_title(Title)
        self.sub2.grid(True)
        if self.reset !=0:    
            self.canvas2.draw()
    
        return True
    def getPlot3(self,X,Y,xlabel,ylabel,Title):
        if self.reset != 0:
            self.sub3.cla()
        self.sub3.plot(X,Y,c = np.random.rand(3,))
        self.sub3.set_xlabel(xlabel, fontsize=12)
        self.sub3.set_ylabel(ylabel, fontsize=12)
        self.sub3.set_title(Title)
        self.sub3.grid(True)
        if self.reset !=0:    
            self.canvas3.draw()
    
        return True
    def getPlot4(self,X,Y,xlabel,ylabel,Title):
        if self.reset != 0:
            self.sub4.cla()
        self.sub4.plot(X,Y,c = np.random.rand(3,))
        self.sub4.set_xlabel(xlabel, fontsize=12)
        self.sub4.set_ylabel(ylabel, fontsize=12)
        self.sub4.set_title(Title)
        self.sub4.grid(True)
        if self.reset !=0:    
            self.canvas4.draw()
    
        return True
    
    def getPlot5(self,X,Y,xlabel,ylabel,Title):
        if self.reset != 0:
            self.sub4.cla()
        self.sub5.plot(X,Y,c = np.random.rand(3,))
        self.sub5.set_xlabel(xlabel, fontsize=12)
        self.sub5.set_ylabel(ylabel, fontsize=12)
        self.sub5.set_title(Title)
        self.sub5.grid(True)
        if self.reset !=0:    
            self.canvas5.draw()
    
        return True
        
    def run(self,): #called when run is clicked
            
        if self.reset == 0:
            #instantiate the figures
            self.fig0 = plt.figure()
            self.sub0 = self.fig0.add_subplot()
            
            self.fig1 = plt.figure() 
            self.sub1 = self.fig1.add_subplot()
            
            self.fig2 = plt.figure()
            self.sub2 = self.fig2.add_subplot()
            
            self.fig3 = plt.figure()
            self.sub3 = self.fig3.add_subplot()
            
            self.fig4 = plt.figure()
            self.sub4 = self.fig4.add_subplot()
            
            self.fig5 = plt.figure()
            self.sub5 = self.fig5.add_subplot()
            
            
        #begin calling ml functions for processing
        Data = getValuesFromRawData(self.FileOfInterest)
        UserInput = UserInputs2WorkingForm(self.n,self.N,self.Bd,self.Pd,self.phi,self.SampleFrequency,Data,self.HomeDirectory,self.directory,self.TrainingDataFile,self.inputSavingalias,self.inputName,self.inputModelnum)
        self.UI = UserInput
        BearingInfo = BearingInfomation(UserInput)
        X_train, X_test, Y_train, Y_test = GetSplitTrainingData(UserInput)
        classifier = TrainModel(X_train, Y_train)
        Y_test_pred = PredictModel(classifier,X_test)
        ##
        #Create time series array
        t = np.arange(0,UserInput['Time of Sampling'],1/UserInput['Sampling Frequency'])
        
        #Perform FFT, PSD, Correlation, DC Offset
        UserInput1 = RemoveDCOffset(UserInput)
        x1 = FourierTransform(UserInput1)
        x2 = get_psd_values(UserInput1)
        x3 = get_autocorr_values(UserInput1)
        
        
        TEST = getTESTDataFrame(UserInput)
        TEST1 = TEST.values[:,0:(TEST.shape[1]-1)]
        OUTCOME = PredictModel(classifier,TEST1)
        plt.close('all')
        figs = []
        #figs = getGraphs(UserInput)
        self.plotinfo,self.plot2info,self.plot3info,self.plot4info,self.plot5info,self.plot6info = getGraphs(UserInput)
        
        
        self.getPlot(self.plotinfo[0],self.plotinfo[1],self.plotinfo[2],self.plotinfo[3],self.plotinfo[4])
        self.getPlot1(self.plot2info[0],self.plot2info[1],self.plot2info[2],self.plot2info[3],self.plot2info[4])
        self.getPlot2(self.plot3info[0],self.plot3info[1],self.plot3info[2],self.plot3info[3],self.plot3info[4])
        self.getPlot3(self.plot4info[0],self.plot4info[1],self.plot4info[2],self.plot4info[3],self.plot4info[4])
        self.getPlot4(self.plot5info[0],self.plot5info[1],self.plot5info[2],self.plot5info[3],self.plot5info[4])
        self.getPlot5(self.plot6info[0],self.plot6info[1],self.plot6info[2],self.plot6info[3],self.plot6info[4])
    
        Prediction = PredictProbModel(classifier,X_test)
        print("outcome:")
        print(OUTCOME)
        print("Accuracy on training set is : {}".format(classifier.score(X_train, Y_train)))
        print("Accuracy on test set is : {}".format(classifier.score(X_test, Y_test)))
        
    
        if self.reset == 0:
            self.addmpl(self.fig0)
            self.addgraph11(self.fig1)
            self.addgraph12(self.fig2)
            self.addgraph21(self.fig3)
            self.addgraph22(self.fig4)
            
        
            
        self.BSF.setText(str(truncate(BearingInfo['BSF'],3)))
        self.BPFI.setText(str(truncate(BearingInfo['BPFI'],3)))
        self.BPFO.setText(str(truncate(BearingInfo['BPFO'],3)))
        self.FTF.setText(str(truncate(BearingInfo['FTF'],3)))
        self.earlyEdit.setText(str(Prediction[0,0]))
        self.suspectEdit.setText(str(Prediction[0,1]))
        self.normalEdit.setText(str(Prediction[0,2]))
        self.immEdit.setText(str(Prediction[0,3]))
        self.innerEdit.setText(str(Prediction[0,4]))
        self.rollingEdit.setText(str(Prediction[0,5]))
        self.stageEdit.setText(str(Prediction[0,6]))
        self.reset = 1 
 
        
        

        
    def close_application(self,): #self explanitory
        sys.exit()

###############################################################################
    def updategraphs(self,fig):
        print("made it to update graphs")
        sip.delete(self.canvas)
        sip.delete(self.canvas1)
        sip.delete(self.canvas2)
        sip.delete(self.canvas3)
        sip.delete(self.canvas4)
        #self.spectrumUI.removeWidget(self.canvas)
        self.canvas = FigureCanvas(fig[0])
        self.canvas1 = FigureCanvas(fig[1])
        self.canvas2 = FigureCanvas(fig[2])
        self.canvas3 = FigureCanvas(fig[3])
        self.canvas4 = FigureCanvas(fig[4])
        
    def rmmpl(self,):
        print("in rmmpl")
        self.spectrumUI.removeWidget(self.canvas)
        self.canvas.close()
        self.spectrumUI.removeWidget(self.toolbar)
        self.toolbar.close()
        
        self.graph11UI.removeWidget(self.canvas)
        self.canvas.close()
        self.graph11UI.removeWidget(self.toolbar)
        self.toolbar.close()
        
        self.graph12UI.removeWidget(self.canvas)
        self.canvas.close()
        self.graph12UI.removeWidget(self.toolbar)
        self.toolbar.close()
        
        self.graph21UI.removeWidget(self.canvas)
        self.canvas.close()
        self.graph21UI.removeWidget(self.toolbar)
        self.toolbar.close()
        
        self.graph22UI.removeWidget(self.canvas)
        self.canvas.close()
        self.graph22UI.removeWidget(self.toolbar)
        self.toolbar.close()       
            
    def plots_close(self,):
        print("made it to plots_close")
        self.spectrumUI.removeWidget(self.canvas)
        sip.delete(self.canvas)
        self.canvas = None
        self.spectrumUI.removeWidget(self.toolbar)
        sip.delete(self.toolbar)
        self.toolbar = None
    
        self.graph11UI.removeWidget(self.canvas1)
        self.canvas1.close()
        self.graph11UI.removeWidget(self.toolbar1)
        self.toolbar1.close()
        
        self.graph12UI.removeWidget(self.canvas2)
        self.canvas2.close()
        self.graph12UI.removeWidget(self.toolbar2)
        self.toolbar2.close()
        
        self.graph21UI.removeWidget(self.canvas3)
        self.canvas3.close()
        self.graph21UI.removeWidget(self.toolbar3)
        self.toolbar3.close()
        
        self.graph22UI.removeWidget(self.canvas4)
        self.canvas4.close()
        self.graph22UI.removeWidget(self.toolbar4)
        self.toolbar4.close()       

class MyPopup(QWidget): #creates popup windows
    def __init__(self, message):
        QWidget.__init__(self)
        alertholder = QLabel(self)
        alertholder.setText(message)
        alertholder.setAlignment(Qt.AlignCenter)
        
        vbox = QVBoxLayout()
        vbox.addWidget(alertholder)
        self.setLayout(vbox)
 





        
        
if __name__ == "__main__": #instantiates GUI and opens it
    from PyQt5 import *
    import sys
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    app.exec_()

    
    
    
    
    
    
