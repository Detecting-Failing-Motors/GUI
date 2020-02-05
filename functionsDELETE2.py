
# coding: utf-8

# In[1]:

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import welch
from detect_peaks import detect_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.backends.backend_pdf
from sklearn import model_selection
from sklearn.metrics import classification_report
from datetime import datetime
import GUI6

# In[2]:


#Import Classifiers
#https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
#Linear Models
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Hinge
#from sklearn.linear_model import Huber #Gave an error
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Log
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ModifiedHuber
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RANSACRegressor
#from sklearn.linear_model import RandomizedLasso #deprecated in 0.19 and will be removed in 0.21.
#from sklearn.linear_model import RandomizedLogisticRegression #deprecated in 0.19 and will be removed in 0.21.
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SquaredLoss
from sklearn.linear_model import TheilSenRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Linear and Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import BaseEstimator
from sklearn.discriminant_analysis import ClassifierMixin
from sklearn.discriminant_analysis import LinearClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import StandardScaler
from sklearn.discriminant_analysis import TransformerMixin

#Kernel Ridge Regression
from sklearn.kernel_ridge import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_ridge import RegressorMixin

#Support Vector Machines
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.svm import SVR

#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor

#Nearest Neighbors
from sklearn.neighbors import BallTree
#from sklearn.neighbors import DistanceMetric #DistanceMetric is an abstract class
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KernelDensity
#from sklearn.neighbors import LSHForest #deprecated in 0.19. It will be removed in version 0.21.
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor

#Gaussian Processes
#from sklearn.gaussian_process import GaussianProcess #deprecated in version 0.18 and will be removed in 0.20.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier

#Cross Decomposition
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSSVD


#Naive Bayes
#from sklearn.naive_bayes import ABCMeta #Gave an error
#from sklearn.naive_bayes import BaseDiscreteNB #Can't instantiate abstract class BaseNB with abstract methods _joint_log_likelihood
from sklearn.naive_bayes import BaseEstimator
#from sklearn.naive_bayes import BaseNB #Can't instantiate abstract class BaseNB with abstract methods _joint_log_likelihood
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier

#Ensemble Methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import BaseEnsemble #Can't instantiate abstract class BaseEnsemble with abstract methods __init__
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import VotingClassifier


#Multiclass and multilabel algorithms
from sklearn.multiclass import BaseEstimator
from sklearn.multiclass import ClassifierMixin
from sklearn.multiclass import LabelBinarizer
from sklearn.multiclass import MetaEstimatorMixin
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import Parallel
#from sklearn.multioutput import ABCMeta #Gave an error
from sklearn.multioutput import BaseEstimator
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import ClassifierMixin
from sklearn.multioutput import MetaEstimatorMixin
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.multioutput import MultiOutputEstimator #Can't instantiate abstract class MultiOutputEstimator with abstract methods __init__
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import Parallel
from sklearn.multioutput import RegressorMixin

#Semi-Suportvised
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation

#Isotonic Regression
from sklearn.isotonic import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.isotonic import RegressorMixin
from sklearn.isotonic import TransformerMixin

# Neural network models (supervised)
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# In[3]:

def getValuesFromRawData(FileOfInterest):
    
    dataset = pd.read_csv(FileOfInterest, header = None, index_col = False)
    dataset.rename(columns={0: "Time", 1: "Value"}, inplace = True)
    print(dataset.head(20))

    #Prepare Raw Data
    i = 0
    time = []
    amp = []
    while i < len(dataset['Time'].values):
        i += 1

        if dataset['Time'].values[i] == 0 and dataset['Value'].values[i] == 0:
            break
        else:
            time.append(dataset['Time'].values[i])
            amp.append(dataset['Value'].values[i])
            
    return amp


def GetData(FileOfInterest):
    """
    Returns an 1-D array of IMS data
    
    Subfunction for UserInputs2WorkingForm
    GetData(
    FileOfInterest - Title of CSV containing the raw data
    )
    This function reads the IMS bearing dataset for set 1 that was taken from:
    http://data-acoustics.com/measurements/bearing-faults/bearing-4/
    """
    """
    #Get Data what tyler had before I messed with it.
    data = pd.read_table(FileOfInterest,header = None)
    data.columns = ['b1x','b1y','b2x','b2y','b3x','b3y','b4x','b4y']
    return np.transpose(data.values[:,0])
    """

    #Brendan stuff
    dataset = pd.read_csv(FileOfInterest, header = None, index_col = False)
    dataset.rename(columns={0: "Time", 1: "Values"},inplace = True)
    return np.transpose(dataset.values[:,0])

def UserInputs2WorkingForm(n,N,Bd,Pd,phi,SampleFrequency,Data,HomeDirectory,directory,TrainingDataFile,save,name_id,modelnum):
    """
    Returns a dictionary of relevant file information
    
    UserInputs2WorkingForm(
        n - Shaft rotational speed [Hz], n
        N - No. of rolling elements [-], N
        Bd - Diameter of a rolling element [mm], Bd
        Pd - Pitch diameter [mm], Pd
        phi - Contact angle [rad], Phi
        SampleFrequecy - SampleFrequency,
        FileOfInterest - Title of CSV containing the raw data
        HomeDirectory - Location of the Current Directory
        directory - Directory of the FileOfInterest
        TrainingDataFile - Title of CSV containg the training data for the machine learning
        )
        
    This functions serves to take all relevant motor characteristics and puts them in a 
    dictionary.
    This dictionary will serve as the building blocks for the rest of the functions.
    """
    #Get Extra Info
    #sig = GetData(FileOfInterest)
    NumberOfSamples = len(Data)
    dt = 1/SampleFrequency
    Tmax = dt*NumberOfSamples
    
    #Arrange
    x = {
        'n': n, #Shaft rotational speed [Hz], n
        'N': N, #No. of rolling elements [-], N
        'Bd': Bd, #Diameter of a rolling element [mm], Bd
        'Pd': Pd, #Pitch diameter [mm], Pd
        'Phi': phi, #Contact angle [rad], Phi
        'Sampling Frequency': SampleFrequency,
        'Time of Sampling': Tmax,
        'Number of Samples': NumberOfSamples,
        'File of Interest': Data,
        'HomeDirectory': HomeDirectory,
        'Working Directory': directory,
        'TrainingFileName': TrainingDataFile,
        'Signal Data of Interest': Data,  
        'Saving Alias': save,
        'Name/ID': name_id,
        'Model#:': modelnum
    }
    return x


# In[4]:


def ReplaceSignalDataofInterest(Data,UserInput,filename):
    """
    Returns a dictionary of relevant file information
    
    UserInputs2WorkingForm(
        Data - This is the actual data
        UserInput - UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
        
    This functions replaces the signal data of interest value of a dictionary.
    This function was created to "cheat" the system and allow the generation of training data 
    by manually inputting the actual data. This helps becuase right now the GetData() function
    only gets B1X column for simplicity.
    """
    
    x = UserInput.copy()
    x['Signal Data of Interest'] = Data 
    x['File of Interest'] = filename

    return x


# In[5]:


def BearingInfomation(UserInput):
    """
    Returns a dictionary with Bearing Characteristic Frequencies
    
    BearingInfomation(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
        
    This functions calculates the bearing characteristic frequencies
    """
    #Get Needed Info
    n = UserInput['n']
    N = UserInput['N']
    Bd = UserInput['Bd']
    Pd = UserInput['Pd']
    phi = UserInput['Phi']
    
    #Calculate Bearing Frequncies
    xx = Bd/Pd*np.cos(phi)
    BPFI = (N/2)*(1 + xx)*n
    BPFO = (N/2)*(1 - xx)*n
    BSF = (Pd/(2*Bd))*(1-(xx)**2)*n
    FTF= (1/2)*(1 - xx)*n
    
    #Arrange
    x = {
        "BPFI": BPFI,
        "BPFO": BPFO,
        "BSF":  BSF,
        "FTF":  FTF
    }
    return x


# In[6]:
    
def CreateSaveDictionary(inputName,inputApplication,inputModelnum,inputSavingalias,inputFile,mlData, horsepower, voltage, phase, shaftnum,shaftSpeed,numberofElements, diameterofElements, pitchDiameter, contactAngle, samFreq):

    x = {
         "inputName": inputName,
         "inputApplication": inputApplication,
         "inputModelnum": inputModelnum,
         "inputSavingalias": inputSavingalias,
         "inputFile": inputFile,
         "mlData": mlData,
         "horsepower": horsepower,
         "voltage": voltage,
         "phase": phase,
         "shaftnum":shaftnum,
         "shaftSpeed": shaftSpeed,
         "numberofElements": numberofElements,
         "diameterofElements": diameterofElements,
         "pitchDiameter": pitchDiameter,
         "contactAngle": contactAngle, 
         "samFreq": samFreq   
    }
    return x
    
def CreateCSVfromDict(x):
    values = []
    names = []
    
    for i in x:
        values.append(x[i])
        names.append(i)
    
    df = pd.DataFrame(values, index = names)  
    df.to_csv(x["inputSavingalias"])
    
    return True


def RemoveDCOffset(UserInput):
    """
    Returns a modified dictionary
    
    RemoveDCOffset(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
        
    This functions removes the dc bias from the signal in the UserInput dictionary
    """
    temp = UserInput.copy()
    temp["Signal Data of Interest"] = temp["Signal Data of Interest"] - np.mean(temp["Signal Data of Interest"])
    return temp


# In[7]:


def FourierTransform(UserInput):
    """
    Returns a dictionary
    
    FourierTransform(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
    )
    
    The functions perfroms fast fourier transform on the UserInput Signal 
    Data of Interest
    """

    #Get Needed Info
    sig = UserInput['Signal Data of Interest']
    NumberOfSamples = UserInput['Number of Samples']
    Tmax = UserInput['Time of Sampling']
    
    #Fourier Transform
    frq = np.arange(NumberOfSamples)/(Tmax)# two sides frequency range
    frq = frq[range(int(NumberOfSamples/(2)))] # one side frequency range
    Y = abs(np.fft.fft(sig))/NumberOfSamples # fft computing and normalization
    Y = Y[range(int(NumberOfSamples/2))]
    
    #Arrange
    x = {
        "Frequency":frq,
        "Freq. Amp.": Y
        }
    return x


# In[8]:


def get_psd_values(UserInput):
    """
    Returns a dictionary
    
    get_psd_values(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
    )
    
    The functions perfroms power spectrum density on the UserInput Signal 
    Data of Interest
    """
    #Get Needed Info
    sig = UserInput['Signal Data of Interest']
    SamplingFrequency = UserInput['Sampling Frequency']
    
    #Perfrom psd
    frq, psd_values = welch(sig, fs=SamplingFrequency)
    
    #Arrange
    x = {
        "Frequency":frq,
        "PSD": psd_values
        }
    return x


# In[9]:


def autocorr(x):
    print("about to print x:")
    print(x)
    
    """
    Taken from:
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    
    Returns the autocorrelation of the signal x
    
    autocorr(
        x - signal of interest
        )
    
    This functions performs correlation
    """
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(UserInput):
    """
    Modified from: 
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    
    Returns a dictionary
    
    get_autocorr_values(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
    )
    
    The functions perfroms autocorrelation on the UserInput Signal 
    Data of Interest
    """
    #Get needed info
    sig = UserInput['Signal Data of Interest']
    print("sig:")
    print(sig)
    Tmax = UserInput['Time of Sampling']
    N = UserInput['Number of Samples']
    
    #Call correlation function
    autocorr_values = autocorr(sig)
    
    #Arrange
    x_values = np.array([Tmax * jj for jj in range(0, N)])
    x = {
        "X Values":x_values,
        "Autocorr Values": autocorr_values
        }
    return x


# In[10]:


def TimeDomainInformation(UserInput):
    """
    Returns a dictionary with Time Domain Characteristics
    
    TimeDomainInformation(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
        
    This functions calculates the Time Domain Characteristics
    """
    #Get Needed Info
    sig = UserInput['Signal Data of Interest']
    
    #Arrange
    x = {
        "RMS": np.mean(sig**2),
        "STD": np.std(sig),
        "Mean": np.mean(sig),
        "Max": np.max(sig),
        "Min": np.min(sig),
        "Peak-to-Peak": (np.max(sig) - np.min(sig)),
        "Max ABS": np.max(abs(sig)),
        "Kurtosis": kurtosis(sig),
        "Skew": skew(sig),
    }

    return x


# In[11]:


def PosMagnitude(Y):
    len(Y)
    mag = 0
    for i in range(0,len(Y)):
        if Y[i] > 0:
            mag = mag + Y[i]**2
    mag = mag ** 0.5
    return mag


def plotPeaks(X,Y,xlabel,ylabel,Title):
    #Set Parameters
    Ymag = PosMagnitude(Y)
    Ynew = Y/Ymag
    min_peak_height = .04
    threshold = 0.15*np.std(Ynew)
    
    #Get indices of peak
    peak = detect_peaks(Ynew,edge = 'rising',mph = min_peak_height, mpd = 5, threshold = threshold )
    fig = plt.figure()
    plt.plot(X,Y)

    for i in peak:
       plt.scatter(X[i],Y[i], c= 'r', marker='*',s = 80)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(Title)
    plt.grid(True)
    plt.show()
    
    return fig

def GetSortedPeak(X,Y):
    """
    SubFunction for FrequencyDomainInformation
    
    Returns Amplitude of Y, Loctation
    
    GetSortedPeak(
        X - Independent Variable
        Y - Dependent Variable
        )
        
    Uses detect_peaks function taken from Github:
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    
    Get the indices of relevant peaks
    Then Returns the amplitude,location of the relevant peaks
    """
    #Original
    """
    #Set Parameters
    min_peak_height = 0.1 * np.nanmax(Y) #Original
    threshold = 0.05 * np.nanmax(Y) #Original
    
    #Get indices of peak
    peak = detect_peaks(Y,edge = 'rising',mph = min_peak_height, mpd = 2, threshold = threshold ) #Original
    """
    #NEW
    #Set Parameters
    Ymag = PosMagnitude(Y)
    Ynew = Y/Ymag
    min_peak_height = .04
    threshold = 0.15*np.std(Ynew)
    
    #Get indices of peak
    peak = detect_peaks(Ynew,edge = 'rising',mph = min_peak_height, mpd = 5, threshold = threshold )
    
    #Get values corresponding to indices 
    m = []
    mm = []
    for i in peak:
        m.append(Y[i]) 
        mm.append(X[i])

    #Sort arcording to the amplitude
    mmm = np.argsort(m)
    n = []
    nn = []
    for i in mmm:
        n.append(m[i])
        nn.append(mm[i])
    
    #Sort in Descending Amplitdue while keeping locations matched
    n  = n[::-1] #amplitude
    nn = nn[::-1] #location
    
    #Arrange
    return n, nn

def FrequencyDomainInformation(UserInput):
    """
    Returns a dictionary with Frequency Domain Characteristics
    Top 5 frequncy and amplitudes for:
    fft
    psd
    correlation
    
    FrequencyDomainInformation(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    """
    #Call FFT, PSD, and Correlation Values
    x1 = FourierTransform(UserInput)
    x2 = get_psd_values(UserInput)
    x3 = get_autocorr_values(UserInput)
    FTamp,FTfreq = GetSortedPeak(x1['Frequency'],x1['Freq. Amp.'])
    PSDamp,PSDfreq = GetSortedPeak(x2['Frequency'],x2['PSD'])
    Cor,CorTime = GetSortedPeak(x3['X Values'],x3['Autocorr Values'])

    #Originally -999 
    #Now 77777
    #Take Care of Empty Values
    while len(FTamp) <= 5:
        FTamp.append(['77777'])
    while len(FTfreq) <= 5:
        FTfreq.append(['77777'])
    while len(PSDamp) <= 5:
        PSDamp.append(['77777'])
    while len(PSDfreq) <= 5:
        PSDfreq.append(['77777'])
    while len(Cor) <= 5:
        Cor.append(['77777'])
    while len(CorTime) <= 5:
        CorTime.append(['77777'])
    
    #Arrange
    x = {
        "FFT Frq @ Peak 1": FTfreq[0],
        "FFT Frq @ Peak 2": FTfreq[1],
        "FFT Frq @ Peak 3": FTfreq[2],
        "FFT Frq @ Peak 4": FTfreq[3],
        "FFT Frq @ Peak 5": FTfreq[4],
        "FFT Amp @ Peak 1": FTamp[0],
        "FFT Amp @ Peak 2": FTamp[1],
        "FFT Amp @ Peak 3": FTamp[2],
        "FFT Amp @ Peak 4": FTamp[3],
        "FFT Amp @ Peak 5": FTamp[4],
        "PSD Frq @ Peak 1": PSDfreq[0],
        "PSD Frq @ Peak 2": PSDfreq[1],
        "PSD Frq @ Peak 3": PSDfreq[2],
        "PSD Frq @ Peak 4": PSDfreq[3],
        "PSD Frq @ Peak 5": PSDfreq[4],
        "PSD Amp @ Peak 1": PSDamp[0],
        "PSD Amp @ Peak 2": PSDamp[1],
        "PSD Amp @ Peak 3": PSDamp[2],
        "PSD Amp @ Peak 4": PSDamp[3],
        "PSD Amp @ Peak 5": PSDamp[4],
        "Autocorrelate Time @ Peak 1": CorTime[0],
        "Autocorrelate Time @ Peak 2": CorTime[1],
        "Autocorrelate Time @ Peak 3": CorTime[2],
        "Autocorrelate Time @ Peak 4": CorTime[3],
        "Autocorrelate Time @ Peak 5": CorTime[4],
        "Autocorrelate @ Peak 1": Cor[0],
        "Autocorrelate @ Peak 2": Cor[1],
        "Autocorrelate @ Peak 3": Cor[2],
        "Autocorrelate @ Peak 4": Cor[3],
        "Autocorrelate @ Peak 5": Cor[4]
    }
    return x


# In[12]:


"""
http://mkalikatzarakis.eu/wp-content/uploads/2018/12/IMS_dset.html
Previous work done on this dataset states that seven different states of health were observed:

Early (initial run-in of the bearings)
Normal
Suspect (the health seems to be deteriorating)
Imminent failure (for bearings 1 and 2, which didn’t actually fail, but were severely worn out)
Inner race failure (bearing 3)
Rolling element failure (bearing 4)
Stage 2 failure (bearing 4)
For the first test (the one we are working on), the following labels have been proposed per file:

Bearing 1
early: 2003.10.22.12.06.24 - 2013.10.23.09.14.13
suspect: 2013.10.23.09.24.13 - 2003.11.08.12.11.44 (bearing 1 was in suspicious health from the beginning, but showed some self-healing effects)
normal: 2003.11.08.12.21.44 - 2003.11.19.21.06.07
suspect: 2003.11.19.21.16.07 - 2003.11.24.20.47.32
imminent failure: 2003.11.24.20.57.32 - 2003.11.25.23.39.56

Bearing 2
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.24.01.01.24
suspect: 2003.11.24.01.11.24 - 2003.11.25.10.47.32
imminent failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 3
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.22.09.16.56
suspect: 2003.11.22.09.26.56 - 2003.11.25.10.47.32
Inner race failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 4
early: 2003.10.22.12.06.24 - 2003.10.29.21.39.46
normal: 2003.10.29.21.49.46 - 2003.11.15.05.08.46
suspect: 2003.11.15.05.18.46 - 2003.11.18.19.12.30
Rolling element failure: 2003.11.19.09.06.09 - 2003.11.22.17.36.56
Stage 2 failure: 2003.11.22.17.46.56 - 2003.11.25.23.39.56
"""

def getAbsoluteTime(file):
    """
    Subfunction for StateInformation
    
    Returns the "magnitude" of the time stamp 
    
    getAbsolutTime(
        file - file name that has bearing information within it
        )
    
    This function computes the magnitude of time when the IMS data was taken 
    """
    #Get needed info
    year   = int(file[0:4])
    month  = int(file[5:7])
    day    = int(file[8:10])
    hour   = int(file[11:13])
    minute = int(file[14:16])
    second = int(file[17:19])
    
    #Compute starting from the 10 month
    #in seconds don't include years taking 10 as the start month
    x = second + 60*minute + 60*60*hour + 24*60*60*day + 31*24*60*60*(month - 10)
    return x

def StateDict():
    State2Int = {
        "Early": 0,
        "Suspect": 1,
        "Normal": 2,
        "Imminent Failure": 3,
        "Inner Race Failure": 4, 
        "Rolling Element Failure": 5,
        "Stage 2 Failure": 6,
        "ERROR": 77777
    }
    
    return State2Int

def StateInformation(UserInput,BearingNum):
    """
    Returns a Dictionary of a Bearing State
    
    StateInformation(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        BearingNum - Bearing Num to know which failure type
        )
    
    This function is used to generate a known outcome for the training data.
    This function is only intended to aid in generating the training data.
    """
    #Get Needed Info 
    file = UserInput['File of Interest']
    
    #Comput time for comarison
    absolutetime = getAbsoluteTime(file)
    
    #Transitions according to the above comments
    #Bearing 1 transitions
    b1e2s  = getAbsoluteTime("2013.10.23.09.14.13")
    b1s2n  = getAbsoluteTime("2003.11.08.12.11.44")
    b1n2s  = getAbsoluteTime("2003.11.19.21.06.07")
    b1s2i  = getAbsoluteTime("2003.11.24.20.47.32")
    
    #Bearing 2 transitions
    b2e2n  = getAbsoluteTime("2003.11.01.21.41.44")
    b2n2s  = getAbsoluteTime("2003.11.24.01.01.24")
    b2s2i  = getAbsoluteTime("2003.11.25.10.47.32")
    
    #Bearing 3 transitions
    b3e2n  = getAbsoluteTime("2003.11.01.21.41.44")
    b3n2s  = getAbsoluteTime("2003.11.22.09.16.56")
    b3s2irf  = getAbsoluteTime("2003.11.25.10.47.32")
    
    #Bearing 4 transitions
    b4e2n  = getAbsoluteTime("2003.10.29.21.39.46")
    b4n2s  = getAbsoluteTime("2003.11.15.05.08.46")
    b4s2r  = getAbsoluteTime("2003.11.18.19.12.30")
    b4r2f  = getAbsoluteTime("2003.11.22.17.36.56")
    
    #Get state / output error if no state possible
    m = "ERROR"
    if BearingNum == 1:
        if absolutetime   <= b1e2s:
            m = "Early"
        elif absolutetime <= b1s2n:
            m = "Suspect"
        elif absolutetime <= b1n2s:
            m = "Normal"
        elif absolutetime <= b1s2i:
            m = "Suspect"
        elif absolutetime > b1s2i:
            m = "Imminent Failure"
    elif BearingNum == 2:
        if absolutetime   <= b2e2n:
            m = "Early"
        elif absolutetime <= b2n2s:
            m = "Normal"
        elif absolutetime <= b2s2i:
            m = "Suspect"
        elif absolutetime > b2s2i:
            m = "Imminent Failure" 
    elif BearingNum == 3:
        if absolutetime   <= b3e2n:
            m = "Early"
        elif absolutetime <= b3n2s:
            m = "Normal"
        elif absolutetime <= b3s2irf:
            m = "Suspect"
        elif absolutetime >= b3s2irf:
            m = "Inner Race Failure"   
    elif BearingNum == 4:
        if absolutetime   <= b4e2n:
            m = "Early"
        elif absolutetime <= b4n2s:
            m = "Normal"
        elif absolutetime <= b4s2r:
            m = "Suspect"
        elif absolutetime <= b4r2f:
            m = "Rolling Element Failure"
        elif absolutetime > b4r2f:
            m = "Stage 2 Failure"
    else:
        m = "ERROR"
    
    #NOT in the original model
    State2Int = StateDict()
    
    #Arrange
    x = {
        "State": State2Int[m]
    }
    return x


# In[13]:


def MotorInformation(UserInput):
    """
    Returns a Dictionary containg motor characteristics used in the IMS dataset
    
    MotorInformation(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
        
    Only valid for IMS dataset
    """
    x = {
        "Motor Type AC(1)-DC(0)": 1,
        "Shaft Speed [Hz]": 2000/60
    }
    return x


# In[14]:


def getCompleteDataFrame(UserInput,BearingNum):
    """
    Returns a Dataframe for sample
    
    getCompleteDataFrame(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        BearingNum - Bearing Num to know which failure type
        )
    
    This function is used to generate a known outcome for the training data.
    This function is only intended to aid in generating the training data.
    """
    #Call specific function order for consistency 
    UserInput1 = UserInput.copy()
    UserInput2 = RemoveDCOffset(UserInput1)
    BearingInfo = BearingInfomation(UserInput2)
    TimeDomainInfo = TimeDomainInformation(UserInput2)
    FrequecyDomainInfo = FrequencyDomainInformation(UserInput2)
    StateInfo = StateInformation(UserInput2,BearingNum)
    MotorInfo = MotorInformation(UserInput2)
    
    #Arrange
    Features = {**StateInfo,**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 


# In[15]:


def getTESTDataFrame(UserInput):
    """
    Returns a Dataframe that does not need the state
    
    getTESTDataFrame(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    
    This function generates a dataframe without knowing its state
    """
    #Call specific function order for consistency 
    UserInput1 = UserInput.copy()
    UserInput2 = RemoveDCOffset(UserInput1)
    BearingInfo = BearingInfomation(UserInput2)
    TimeDomainInfo = TimeDomainInformation(UserInput2)
    FrequecyDomainInfo = FrequencyDomainInformation(UserInput2)
    MotorInfo = MotorInformation(UserInput2)
    
    #Arrange (with no state info)
    Features = {**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 


# In[16]:


#def getPlot(X,Y,xlabel,ylabel,Title,fig):
#    """
#    Subfunction of getGraphs
#    Returns a figure
#    
#    getPlot(
#        X - Data for independent variable
#        Y - Data for dependent variable
#        xlabel - X-axis label
#        ylabel - Y-axis label
#        Title - Title of figure
#        )
#    
#    Performs plt.plot
#    """
#    
#    #Plot
#    
#    plt.figure(fig)
#    plt.plot(X,Y,c = np.random.rand(3,))
#    plt.xlabel(xlabel, fontsize=12)
#    plt.ylabel(ylabel, fontsize=12)
#    plt.title(Title)
#    plt.grid(True)
#    plt.show()
#    
#    return True

def getGraphs(UserInput):
    """
    Returns an array of figures
    
    getGraphs(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    
    This function generates a figures for:
    Raw time series
    Time series with no DC offset
    FFT
    PSD
    Correlation
    """
    #Create time series array
    t = np.arange(0,UserInput['Time of Sampling'],1/UserInput['Sampling Frequency'])
    
    #Perform FFT, PSD, Correlation, DC Offset
    UserInput1 = RemoveDCOffset(UserInput)
    x1 = FourierTransform(UserInput1)
    x2 = get_psd_values(UserInput1)
    x3 = get_autocorr_values(UserInput1)
    
    
    #Get Figures
    """figs = []
    figs.append(getPlot(t,UserInput['Signal Data of Interest'],"time (s)","Amplitude","Raw Data"))
    figs.append(getPlot(t,UserInput1['Signal Data of Interest'],"time (s)","Amplitude","Raw Data w/ Removed DC Offset"))
    figs.append(getPlot(x1['Frequency'],x1['Freq. Amp.'],'Frequency [Hz]',"time (s)","FFT"))
    figs.append(getPlot(x2['Frequency'],x2['PSD'],'Frequency [Hz]','PSD [V**2 / Hz]',"PSD"))
    figs.append(getPlot(x3['X Values'],x3['Autocorr Values'],'time delay [s]',"Autocorrelation amplitude","Autocorrelation"))
    """
    plotinfo = []
    plot2info = []
    plot3info = []
    plot4info = []
    plot5info = []
    #getPlot(t,UserInput['Signal Data of Interest'],"time (s)","Amplitude","Raw Data",GUI6.fig0)
    plotinfo.append(t)
    plotinfo.append(UserInput['Signal Data of Interest'])
    plotinfo.append("time (s)")
    plotinfo.append("Amplitude")
    plotinfo.append("Raw Data")
                                    
    #getPlot(t,UserInput1['Signal Data of Interest'],"time (s)","Amplitude","Raw Data w/ Removed DC Offset",stringfigs[1])
    plot2info.append(t)
    plot2info.append(UserInput1['Signal Data of Interest'])
    plot2info.append("time (s)")
    plot2info.append("Amplitude")
    plot2info.append("Raw Data w/ Removed DC Offset")
    
    
    #getPlot(x1['Frequency'],x1['Freq. Amp.'],'Frequency [Hz]',"time (s)","FFT",stringfigs[2])
    plot3info.append(x1['Frequency'])
    plot3info.append(x1['Freq. Amp.'])
    plot3info.append('Frequency [Hz]')
    plot3info.append("time (s)")
    plot3info.append("FFT")
    
    #getPlot(x2['Frequency'],x2['PSD'],'Frequency [Hz]','PSD [V**2 / Hz]',"PSD",stringfigs[3])
    plot4info.append(x2['Frequency'])
    plot4info.append(x2['PSD'])
    plot4info.append('Frequency [Hz]')
    plot4info.append('PSD [V**2 / Hz]')
    plot4info.append("PSD")
    
    
    #getPlot(x3['X Values'],x3['Autocorr Values'],'time delay [s]',"Autocorrelation amplitude","Autocorrelation",stringfigs[4])
    plot5info.append(x3['X Values'])
    plot5info.append(x3['Autocorr Values'])
    plot5info.append('time delay [s]')
    plot5info.append("Autocorrelation amplitude")
    plot5info.append("Autocorrelation")
    
    return plotinfo,plot2info,plot3info,plot4info,plot5info


# In[17]:


def getBarPlot(X,Y,xlabel,Title):
    """
    Subfunction of getGraphs
    Returns a figure
    
    getBarPlot(
        X - Data for independent variable
        Y - Data for dependent variable
        xlabel - X-axis label

        Title - Title of figure
        )
    
    Performs plt.barh
    """
    #Bar plot
    fig = plt.figure()
    y_pos = np.arange(len(Y))
    plt.barh(y_pos, X, align='center')
    plt.xlabel(xlabel, fontsize=12)
    plt.yticks(y_pos, Y)
    plt.title(Title)
    plt.grid(True)
    return fig


# In[18]:


def truncate(f, n):
    '''https://stackoverflow.com/questions/783897/truncating-floats-in-python/51172324#51172324'''
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


# In[19]:


def GetSplitTrainingData(UserInput, seed = 6):
    """
    Returns an X_train, X_test, Y_train, Y_test
    
    getGraphs(UserInput)
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        seed - random number for splitting of test and trainig (default = 6)
        )
    
    This returns the training and test sets
    """
    
    #Find training file name and read it

    dataset = pd.read_csv(UserInput['TrainingFileName'],header = 0,index_col = 0)

    X = dataset.values[:,1:(dataset.shape[1]-1)]
    Y = dataset.values[:,0]
    validation_size = 0.20
    seed = seed
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) 
     
    return X_train, X_test, Y_train, Y_test


# In[20]:


def GetTrainingData(UserInput):
    """
    Returns X_train, Y_train
    
    getGraphs(UserInput)
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    
    This returns the training and test sets
    """
    
    #Find training file name and read it
    for file in UserInput['Working Directory']:
        if file == UserInput['TrainingFileName']:
            dataset = pd.read_csv(file,header = 0,index_col = 0)

    X_train = dataset.values[:,1:(dataset.shape[1]-1)]
    Y_train = dataset.values[:,0]
    
    return X_train, Y_train, dataset


# In[21]:


def GetTESTDataFrameNames(UserInput):
    """
    Returns an array of strings
    
    GetTESTDataFrameNames(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    
    This returns the names of each column of the training file
    """
    for file in UserInput['Working Directory']:
        if file == UserInput['TrainingFileName']:
            dataset = pd.read_csv(file,header = 0,index_col = 0)
    names = []
    for x in dataset.columns:
        names.append(x)
    return names


# In[22]:


def TrainModel(X_train,Y_train):
    """
    Returns a classifier that has been fit
    
    TrainModel(
        X_train - Training Data
        Y_train - Results of Training Data for supervised learning
        )
    
    Currently only fits RandomForestClassifier
    """
    classifier = RandomForestClassifier(n_estimators=200)
    classifier = classifier.fit(X_train, Y_train)
    return classifier


# In[23]:


def get_key(value,dictionary): 
    """
    
    Modified from:
    https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    """
    result = []
    for x in value:
        for key, val in dictionary.items(): 
            if val == x:
                result.append(key)  
  
    return result

def PredictModel(classifier,X_test):
    """
    Returns a prediction
    
    PredictModel(
        classifier - fitted classifier
        X_test - data to be tested
        )
    """
    State2Int = StateDict()
    Y_test_pred = classifier.predict(X_test)
    Y_test_pred_string = get_key(Y_test_pred,State2Int)
    return Y_test_pred, Y_test_pred_string


# In[24]:


def PredictProbModel(classifier,X_test):
    """
    Returns a prediction probability (out of 100 not 1)
    
    PredictModel(
        classifier - fitted classifier
        X_test - data to be tested
        )
    """
    Y_test_pred_proba = classifier.predict_proba(X_test)
    
    return Y_test_pred_proba*100


# In[25]:


def GetAllModelsForComparison(X_train,Y_train):
    """
    Returns a an array of all possible ML classifiers with "default" settings
    
    getGraphs(
        X_train - Training Data
        Y_train - Results of Training Data for supervised learning
        )
    """
    #Arrange
    models = {
        'ARDRegression': ARDRegression(),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(),
        'ElasticNetCV': ElasticNetCV(),
        'Hinge': Hinge(),
        #'Huber': Huber(), #Gave an error
        'HuberRegressor': HuberRegressor(),
        'Lars': Lars(),
        'LarsCV': LarsCV(),
        'Lasso': Lasso(),
        'LassoCV': LassoCV(),
        'LassoLars': LassoLars(),
        'LassoLarsCV': LassoLarsCV(),
        'LinearRegression': LinearRegression(),
        'Log': Log(),
        'LogisticRegression': LogisticRegression(),
        'LogisticRegressionCV': LogisticRegressionCV(),
        'ModifiedHuber': ModifiedHuber(),
        'MultiTaskElasticNet': MultiTaskElasticNet(),
        'MultiTaskElasticNetCV': MultiTaskElasticNetCV(),
        'MultiTaskLasso': MultiTaskLasso(),
        'MultiTaskLassoCV': MultiTaskLassoCV(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV(),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'Perceptron': Perceptron(),
        'RANSACRegressor': RANSACRegressor(),
        #'RandomizedLasso': RandomizedLasso(), #deprecated in 0.19 and will be removed in 0.21.
        #'RandomizedLogisticRegression': RandomizedLogisticRegression(), #deprecated in 0.19 and will be removed in 0.21.
        'Ridge': Ridge(),
        'RidgeCV': RidgeCV(),
        'RidgeClassifier': RidgeClassifier(),
        'SGDClassifier': SGDClassifier(),
        'SGDRegressor': SGDRegressor(),
        'SquaredLoss': SquaredLoss(),
        'TheilSenRegressor': TheilSenRegressor(),
        'BaseEstimator': BaseEstimator(),
        'ClassifierMixin': ClassifierMixin(),
        'LinearClassifierMixin': LinearClassifierMixin(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'StandardScaler': StandardScaler(),
        'TransformerMixin': TransformerMixin(),
        'BaseEstimator': BaseEstimator(),
        'KernelRidge': KernelRidge(),
        'RegressorMixin': RegressorMixin(),
        'LinearSVC': LinearSVC(),
        'LinearSVR': LinearSVR(),
        'NuSVC': NuSVC(),
        'NuSVR': NuSVR(),
        'OneClassSVM': OneClassSVM(),
        'SVC': SVC(),
        'SVR': SVR(),
        'SGDClassifier': SGDClassifier(),
        'SGDRegressor': SGDRegressor(),
        'BallTree': BallTree(X_train), #Needs Data
        #'DistanceMetric': DistanceMetric(), #DistanceMetric is an abstract class
        'KDTree': KDTree(X_train), #Needs Data
        'KNeighborsClassifier': KNeighborsClassifier(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'KernelDensity': KernelDensity(),
        #'LSHForest': LSHForest(), #deprecated in 0.19. It will be removed in version 0.21.
        'LocalOutlierFactor': LocalOutlierFactor(),
        'NearestCentroid': NearestCentroid(),
        'NearestNeighbors': NearestNeighbors(),
        'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),
        'RadiusNeighborsRegressor': RadiusNeighborsRegressor(),
        #'GaussianProcess': GaussianProcess(), #deprecated in version 0.18 and will be removed in 0.20.
        'GaussianProcessRegressor': GaussianProcessRegressor(),
        'GaussianProcessClassifier': GaussianProcessClassifier(),
        'CCA': CCA(),
        'PLSCanonical': PLSCanonical(),
        'PLSRegression': PLSRegression(),
        'PLSSVD': PLSSVD(),
        #'ABCMeta': ABCMeta(), #Gave an error
        #'BaseDiscreteNB': BaseDiscreteNB(), #Can't instantiate abstract class BaseNB with abstract methods _joint_log_likelihood
        'BaseEstimator': BaseEstimator(),
        #'BaseNB': BaseNB(), #Can't instantiate abstract class BaseNB with abstract methods _joint_log_likelihood
        'BernoulliNB': BernoulliNB(),
        'ClassifierMixin': ClassifierMixin(),
        'GaussianNB': GaussianNB(),
        'LabelBinarizer': LabelBinarizer(),
        'MultinomialNB': MultinomialNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'ExtraTreeClassifier': ExtraTreeClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'BaggingClassifier': BaggingClassifier(),
        'BaggingRegressor': BaggingRegressor(),
        #'BaseEnsemble': BaseEnsemble(), #Can't instantiate abstract class BaseEnsemble with abstract methods __init__
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'IsolationForest': IsolationForest(),
        'RandomForestClassifier': RandomForestClassifier(),
        'RandomForestRegressor': RandomForestRegressor(),
        'RandomTreesEmbedding': RandomTreesEmbedding(),
        'VotingClassifier': VotingClassifier(estimators=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        'BaseEstimator': BaseEstimator(),
        'ClassifierMixin': ClassifierMixin(),
        'LabelBinarizer': LabelBinarizer(),
        'MetaEstimatorMixin': MetaEstimatorMixin(),
        'OneVsOneClassifier': OneVsOneClassifier(estimator=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        'OneVsRestClassifier': OneVsRestClassifier(estimator=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        'OutputCodeClassifier': OutputCodeClassifier(estimator=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        'Parallel': Parallel(),
        #'ABCMeta': ABCMeta(), #Gave an error
        'BaseEstimator': BaseEstimator(),
        'ClassifierChain': ClassifierChain(base_estimator=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        'ClassifierMixin': ClassifierMixin(),
        'MetaEstimatorMixin': MetaEstimatorMixin(),
        'MultiOutputClassifier': MultiOutputClassifier(estimator=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        #'MultiOutputEstimator': MultiOutputEstimator(), #Can't instantiate abstract class MultiOutputEstimator with abstract methods __init__
        'MultiOutputRegressor': MultiOutputRegressor(estimator=[('LR', LogisticRegression()), \
                                    ('RFC', RandomForestClassifier()), ('GBC', GradientBoostingClassifier())]), #Needs estimator
        'Parallel': Parallel(),
        'RegressorMixin': RegressorMixin(),
        'LabelPropagation': LabelPropagation(),
        'LabelSpreading': LabelSpreading(),
        'BaseEstimator': BaseEstimator(),
        'IsotonicRegression': IsotonicRegression(),
        'RegressorMixin': RegressorMixin(),
        'TransformerMixin': TransformerMixin(),
        'BernoulliRBM': BernoulliRBM(),
        'MLPClassifier': MLPClassifier(),
        'MLPRegressor': MLPRegressor()
        }
    return models


# In[26]:


def GetOnlyTwoModelsForComparison(X_train,Y_train):
    """
    Returns a an array of just TWO possible ML classifiers with "default" settings
    
    getGraphs(
        X_train - Training Data
        Y_train - Results of Training Data for supervised learning
        )
    """
    #Arrange
    models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'ExtraTreesClassifier': ExtraTreesClassifier()
    }
    
    return models


# In[27]:


def GetUserInputNames(UserInput):
    """
    Returns an array of strings
    
    GetUserInputNames(
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    
    This returns the names of the columns in UserInput
    """
    names = []
    for x in UserInput:
        names.append(x)
    return names


# In[28]:


def FeatureComparison(models,X_train, X_test, Y_train, Y_test,UserInput):
    """
    returns: 
        results - an array of classification results
        string - an array of classifier names and classification results
        string1 - an array of classifiers that do not have feature importance graphs
        time - an array of time stamps recording how long a fit/prediction/classification
                result took for an figure
        fig - an array of figures containing feature importance for classifiers
        
    FeatureComparison(
        models - an array of classifiers for testing
        X_train - Training Data
        X_test - Data to be fitted
        Y_train - Results of Training Data for supervised learning
        Y_test - results of the fitted data
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
        
    The function usefullness lies in its ability to test mass quantities of classifiers
    """
    results = []
    string = []
    string1 = []
    fig = []
    time = []
    for ModelName in models:
        
        before = datetime.now()
        
        #Fit Model
        error = False
        try:
            CTest = models[ModelName].fit(X_train, Y_train)
        except:
            print()
            error = True
        
        #Classification report
        if not error:
            try:
                Y_pred = CTest.predict(X_test)
                temporary = classification_report(Y_test,Y_pred)
                results.append(temporary)
                #CTest.score(X_test, Y_test))) #optional metric
                string.append("{} has the following results: \n\n {} \n\n".format(ModelName,temporary))
            except:
                try:
                    Y_pred = CTest.predict(X_test)
                    temporary = classification_report(Y_test,np.round_(Y_pred))
                    results.append(temporary)
                    #CTest.score(X_test, Y_test))) #optional metric
                    string.append("{} has the following results: \n\n {} \n\n".format(ModelName,temporary))
                except:    
                    string.append("{} failed during classification_report.".format(ModelName))
        else:
            string.append("{} failed during the fit.".format(ModelName))
        
        #Time fitting and scoring took
        end = datetime.now()
        time.append("{} took {} time".format(ModelName,(end-before)))
        
        #Try figures
        try:
            m = CTest.feature_importances_
            m1 = GetTESTDataFrameNames(UserInput)
            Z = [x for _,x in sorted(zip(m,m1))]
            Z1 = sorted(m)
            fig.append(getBarPlot(Z1[-10:],Z[-10:],"Relative Importance",ModelName))
        except:
            string1.append("{} has no feature importance".format(ModelName)) 
            
    return results,string,string1,time,fig


# In[29]:


def GenerateComparisonResultFiles(results,string,string1,time,fig,str1 = "Graphs.pdf",                                  str2 = "Time.txt",str3 = "NoGraphs.txt",str4 = "Scoring.txt"):
    """
    returns true 
    
    upon completion
    
    GenerateComparisonResultFiles(
        results - an array of classification results
        string - an array of classifier names and classification results
        string1 - an array of classifiers that do not have feature importance graphs
        time - an array of time stamps recording how long a fit/prediction/classification
                result took for an figure
        fig - an array of figures containing feature importance for classifiers
        str1 - title of pdf file relating to figs (Default: "Graphs.pdf")
        str2 = title of txt file relating to time (Default: Time.txt")
        str3 = title of txt file relating to string1 (Default: "NoGraphs.txt")
        str4 = title of txt file relating to string (Default: "Scoring.txt")
        )
        
    The function generates 4 files.
    This function is to be used after FeatureComparison()
    """
    #PDF of figures
    pdf = matplotlib.backends.backend_pdf.PdfPages(str1)
    i = 0
    for figure in fig:
        pdf.savefig( fig[i],dpi=300, bbox_inches = "tight")
        i += 1
    pdf.close()

    #.txt of Time
    if not(not time):
        with open(str2, 'w') as writeFile:
            for i in np.arange(len(time)):
                writeFile.write("%(t)s\n" % {"t":time[i]})
    writeFile.close()   

    #.txt of NoGraphs
    if not(not string1):
        with open(str3, 'w') as writeFile:
            for i in np.arange(len(string1)):
                writeFile.write("%(t)s\n" % {"t":string1[i]})
    writeFile.close()       

    #.txt Scoring Results
    if not(not string):
        with open(str4, 'w') as writeFile:
            for i in np.arange(len(string)):
                writeFile.write("%(t)s\n" % {"t":string[i]})
    writeFile.close()
    
    return True


# In[30]:


def SignalGenerator(t):
    """
    returns numpy.ndarray
    
    SignalGenerator(
        t - numpy.ndarray
        )
        
    The file generates a random sine wave combination with white noise
    given an input time series
    """
    #Signal generator for practice
    noise1 = np.random.randn(len(t))                # white noise 1
    noise2 = np.random.randn(len(t))                # white noise 2 
    noise3 = np.random.randn(len(t))                # white noise 3
    phase  = np.random.randn(3)                     #radians
    frequency1 = np.random.randint(1,1000)          #Hz
    frequency2 = np.random.randint(1,1000)          #Hz
    frequency3 = np.random.randint(1,1000)          #Hz
    mag = np.random.randn(3)
    base1 = mag[0] * np.sin(2 * np.pi * frequency1 * t + phase[0] ) + noise1  #base signal
    base2 = mag[1] * np.sin(2 * np.pi * frequency2 * t + phase[1] ) + noise2  #base signal
    base3 = mag[2] * np.sin(2 * np.pi * frequency3 * t + phase[2] ) + noise3  #base signal
    return base1 + base2 + base3
    #End of practice signal generator


# In[31]:


def GenerateIMSDictionary(FileOfInterest,TrainingDataFile,HomeDirectory):
    """
    returns UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
    
    GenerateIMSDictionary(
        FileOfInterest - File name a single IMS file
        TrainingDataFile - File name of TrainingDataFile
        HomeDirectory - Directory with training file 
    )
    
    Example:
        n = 2000 / 60
        N = 16
        Bd = 0.331*254
        Pd = 2.815*254
        phi = 15.17 * np.pi / 180
        SampleFrequency = 20000
        FileOfInterest = '2003.10.22.12.06.24'
        HomeDirectory = os.getcwd()
        os.chdir(HomeDirectory)
        directory = os.listdir(HomeDirectory)
        TrainingDataFile = "DELETE.csv"
        UserInput = UserInputs2WorkingForm(n,N,Bd,Pd,phi,SampleFrequency,FileOfInterest,HomeDirectory,directory,TrainingDataFile)

    This is the same as: TEST = GenerateIMSDictionary('2003.10.22.12.06.24',"DELETE.csv",os.getcwd())
    """
    n = 2000 / 60
    N = 16
    Bd = 0.331*254
    Pd = 2.815*254
    phi = 15.17 * np.pi / 180
    SampleFrequency = 20000
    FileOfInterest = FileOfInterest
    HomeDirectory = HomeDirectory
    directory = os.listdir(HomeDirectory)
    TrainingDataFile = TrainingDataFile
    UserInput = UserInputs2WorkingForm(n,N,Bd,Pd,phi,SampleFrequency,FileOfInterest,HomeDirectory,directory,TrainingDataFile)

    return UserInput


# In[32]:


def GenerateTrainingFile(string):
    HomeDirectory = "/Users/tbryan/Desktop/9 2019 Fall/ECEN 403/Programming/ProgramsForDemo"
    os.chdir(HomeDirectory)
    os.chdir('Data')
    os.chdir('IMS')
    directory = [sorted(os.listdir('1st_test')),sorted(os.listdir('2nd_test')),sorted(os.listdir('3rd_test/txt'))]

    #directory.remove(".DS_Store")
    os.chdir('1st_test')
    IMSDictionary = GenerateIMSDictionary('2003.10.22.12.06.24',"NoNegatives.csv",os.getcwd())

    m1 = []
    m2 = []
    m3 = []
    m4 = []

    for j in range(0,1):
        i = 0
        while i < len(directory[j]):
        #while i < 20:
            if directory[j][i] != ".DS_Store":
                data = pd.read_table(directory[j][i],header = None)
                DF = ReplaceSignalDataofInterest(np.transpose(data.values[:,0]),IMSDictionary,directory[j][i])
                B1X = getCompleteDataFrame(DF,1)
                DF = ReplaceSignalDataofInterest(np.transpose(data.values[:,2]),IMSDictionary,directory[j][i])
                B2X = getCompleteDataFrame(DF,2)
                DF = ReplaceSignalDataofInterest(np.transpose(data.values[:,4]),IMSDictionary,directory[j][i])
                B3X = getCompleteDataFrame(DF,3)
                DF = ReplaceSignalDataofInterest(np.transpose(data.values[:,6]),IMSDictionary,directory[j][i])
                B4X = getCompleteDataFrame(DF,4)
                ColumnTitle = B1X.columns
                m1.append(B1X.values[0,:])
                m2.append(B2X.values[0,:])
                m3.append(B3X.values[0,:])
                m4.append(B4X.values[0,:])
                i += 1
            else:
                i += 1

        B1X = pd.DataFrame(m1,columns = ColumnTitle)
        B2X = pd.DataFrame(m2,columns = ColumnTitle)
        B3X = pd.DataFrame(m3,columns = ColumnTitle)
        B4X = pd.DataFrame(m4,columns = ColumnTitle)
        TD = B1X
        TD = TD.append(B2X)
        TD = TD.append(B3X)
        TD = TD.append(B4X)

    os.chdir(HomeDirectory)
    TD.to_csv(string)

    return True

def get_type(input_data):
    try:
        return type(literal_eval(input_data))
    except (ValueError, SyntaxError):
        # A string, so return str
        return str

