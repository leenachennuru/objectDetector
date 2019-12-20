# -*- coding: utf-8 -*-
#
#
#
#from features import mfcc
#from features import logfbank
#from features import fbank
#from features import ssc
#
#from features.sigproc import framesig
#from features.sigproc import preemphasis
#from features.sigproc import magspec
#from features.sigproc import powspec
#from features.sigproc import logpowspec
#



#from librosa import feature
#from librosa import load
import DataUtil
import ImageProcessingUtil

import wave
from scipy import signal
import scipy.io.wavfile
import numpy
import pylab
import os
import cv2

from pylab import *

import scipy.signal as signal

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig("/data/datasets/0.png")  
    pylab.close()
    img = cv2.imread("/data/datasets/0.png")
    return img


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
    
    

def plotFrequencyWav(audioFile):
    print "Audio:",     audioFile
    samplerate, data = scipy.io.wavfile.read(audioFile)
    
    n = len(data)
    
    p = fft(data)
    
    nUniquePts = ceil((n+1)/2.0)
    p = p[0:nUniquePts]
    p = abs(p)
    
    p = p / float(n) # scale by the number of points so that
                 # the magnitude does not depend on the length 
                 # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 
    
    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft
    
    freqArray = arange(0, nUniquePts, 1.0) * (samplerate / n);
    plot(freqArray/1000, 10*log10(p), color='k')
    #pylab.axis('off')
    pylab.savefig("/data/datasets/0.png")  
    pylab.close()
    img = cv2.imread("/data/datasets/0.png")
    
   
   
    
#    for i in range(totalDuration/sliceSize):
#        sliceFrom = i*sliceSize*samplerate
#        sliceTo = sliceSize*(i+1)*samplerate
#        
#        data = data[sliceFrom:sliceTo]        
        
#    times = numpy.arange(len(data))/float(samplerate)
#    
#    pylab.fill_between(times, data)     
#    pylab.axis('off')    
#    pylab.savefig("/data/datasets/0.png")    
#    pylab.close()
#    img = cv2.imread("/data/datasets/0.png")
        
    return img
    
    
def plotWav(audioFile):
        
    samplerate, data = scipy.io.wavfile.read(audioFile)
    sliceSize = 1
    totalDuration = len(data) / samplerate 
    
    print "Size:", totalDuration/sliceSize
    
    
#    for i in range(totalDuration/sliceSize):
#        sliceFrom = i*sliceSize*samplerate
#        sliceTo = sliceSize*(i+1)*samplerate
#        
#        data = data[sliceFrom:sliceTo]        
        
    times = numpy.arange(len(data))/float(samplerate)
    
    pylab.fill_between(times, data)     
    pylab.axis('off')    
    pylab.savefig("/data/datasets/0.png")    
    pylab.close()
    img = cv2.imread("/data/datasets/0.png")
    
    
    return img
    
    

def delta(data, width=9, order=1, axis=-1, trim=True):
    r'''Compute delta features: local estimate of the derivative
    of the input data along the selected axis.
    Parameters
    ----------
    data      : np.ndarray
        the input data matrix (eg, spectrogram)
    width     : int >= 3, odd [scalar]
        Number of frames over which to compute the delta feature
    order     : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.
    axis      : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).
    trim      : bool
        set to `True` to trim the output matrix to the original size.
    Returns
    -------
    delta_data   : np.ndarray [shape=(d, t) or (d, t + window)]
        delta matrix of `data`.
    Examples
    --------
    Compute MFCC deltas, delta-deltas
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
    >>> mfcc_delta = librosa.feature.delta(mfcc)
    >>> mfcc_delta
    array([[  2.929e+01,   3.090e+01, ...,   0.000e+00,   0.000e+00],
           [  2.226e+01,   2.553e+01, ...,   3.944e-31,   3.944e-31],
           ...,
           [ -1.192e+00,  -6.099e-01, ...,   9.861e-32,   9.861e-32],
           [ -5.349e-01,  -2.077e-01, ...,   1.183e-30,   1.183e-30]])
    >>> mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    >>> mfcc_delta2
    array([[  1.281e+01,   1.020e+01, ...,   0.000e+00,   0.000e+00],
           [  2.726e+00,   3.558e+00, ...,   0.000e+00,   0.000e+00],
           ...,
           [ -1.702e-01,  -1.509e-01, ...,   0.000e+00,   0.000e+00],
           [ -9.021e-02,  -7.007e-02, ...,  -2.190e-47,  -2.190e-47]])
    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(mfcc)
    >>> plt.title('MFCC')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(mfcc_delta)
    >>> plt.title(r'MFCC-$\Delta$')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.specshow(mfcc_delta2, x_axis='time')
    >>> plt.title(r'MFCC-$\Delta^2$')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    '''

    data = numpy.atleast_1d(data)
#
#    if width < 3 or np.mod(width, 2) != 1:
#        raise ParameterError('width must be an odd integer >= 3')
#
#    if order <= 0 or not isinstance(order, int):
#        raise ParameterError('order must be a positive integer')

    half_length = 1 + int(width // 2)
    window = numpy.arange(half_length - 1., -half_length, -1.)

    # Normalize the window so we're scale-invariant
    window /= numpy.sum(numpy.abs(window)**2)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    width = int(width)
    padding[axis] = (width, width)
    delta_x = numpy.pad(data, padding, mode='edge')

    for _ in range(order):
        delta_x = scipy.signal.lfilter(window, 1, delta_x, axis=axis)

    # Cut back to the original shape of the input data
    if trim:
        idx = [slice(None)] * delta_x.ndim
        idx[axis] = slice(- half_length - data.shape[axis], - half_length)
        delta_x = delta_x[idx]

    return delta_x


def audioFeatureExtractor(wavDirectory, sliceSize, featureType, imageSize):
    samplerate, data = scipy.io.wavfile.read(wavDirectory)
    
    #print "SampleRate:", samplerate
    #print "Data Lenght:", len(data)
    #print "Seconds:", len(data) / samplerate     
    
    
#    data = preemphasis(data)
#    spectrum = mfcc(data,samplerate, nfft=1024)
#    spectrum = convertFloatImage(spectrum.T)            
#    spectrum = cv2.resize(spectrum,(64,64))
#
#    return spectrum
    
    totalDuration = len(data) / samplerate 
    spectrums = []
    for i in range(int(totalDuration/sliceSize)):

        sliceFrom = i*sliceSize*samplerate
        sliceTo = sliceSize*(i+1)*samplerate
        
        dataSliced = data[sliceFrom:sliceTo]
        dataSliced = preemphasis(dataSliced)
       
        dataSliced = signal.resample(dataSliced, 16000)
        #spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)
        #print "SampleRate:", samplerate
        frames = framesig(dataSliced, 16000*0.025, 16000*0.01)
        if featureType[0] == DataUtil.FEATURE_TYPE["POW"][0]:
            spectrum = logpowspec(frames,2048)
        elif featureType[0] == DataUtil.FEATURE_TYPE["MAG"][0]:
            spectrum = magspec(frames,2048)
        elif featureType[0] == DataUtil.FEATURE_TYPE["MFCC"][0]:
           spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)        
        
        deltaOrder = 0
        if deltaOrder == 1:
            spectrum = delta(spectrum, order=1)
        elif deltaOrder == 2:
            spectrum = delta(spectrum, order=2)
            
        #spectrum = 20*numpy.log10(abs(spectrum))
        spectrum = convertFloatImage(spectrum.T)  
        
        cv2.imwrite("test2.png", spectrum)          
        spectrum = cv2.imread("test2.png")         
        #spectrum = cv2.cvtColor(spectrum,cv2.COLOR_BGR2GRAY)
        print "Size:", numpy.array(spectrum).shape            
        raw_input("here")
        
        #spectrum = cv2.resize(spectrum,(imageSize[0], imageSize[1]))                   
        spectrums.append(spectrum)      
        
        
    #raw_input("here")
    return spectrums       

#    samplerate, data = scipy.io.wavfile.read(wavDirectory)
#    
#    totalDuration = len(data) / samplerate 
#    
##    sliceSize = 1    
#    
#    spectrums = []
#    
#    for i in range(totalDuration/sliceSize):
#        sliceFrom = i*sliceSize*samplerate
#        sliceTo = sliceSize*(i+1)*samplerate
#        
#        dataSliced = data[sliceFrom:sliceTo]
#        dataSliced = preemphasis(dataSliced)
#       
#        samplerate = 16000
#        dataSliced = signal.resample(dataSliced, samplerate)
#        #samplerate = 48000
#        
#        frames = framesig(dataSliced, samplerate*0.025, samplerate*0.01)
#        spectrum = logpowspec(frames,2048)
#        
#        if featureType[0] == DataUtil.FEATURE_TYPE["MFCC"][0]:
#             spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)
#        elif featureType[0] == DataUtil.FEATURE_TYPE["MFCC_Delta"][0]:             
#            spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)    
#            spectrum = delta(spectrum, order=1)
#        elif featureType[0] == DataUtil.FEATURE_TYPE["MFCC_DeltaDelta"][0]:             
#            spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)
#            spectrum = delta(spectrum, order=2)
#        elif featureType[0] == DataUtil.FEATURE_TYPE["POW"][0]:
#             frames = framesig(dataSliced, samplerate*0.025, samplerate*0.01)
#             spectrum = logpowspec(frames,2048)
#        elif featureType[0] == DataUtil.FEATURE_TYPE["MAG"][0]:
#             frames = framesig(dataSliced, samplerate*0.025, samplerate*0.01)
#             spectrum = magspec(frames, 2048)        
#          
#        #print "Directory:", wavDirectory  
#        #print "Shape spectrum:", numpy.array(spectrum).shape
#        invertedData = spectrum.T    
#        #print "Shape invertedData:", numpy.array(invertedData).shape
#        
#        imageData = convertFloatImage(invertedData)          
#        #print "Shape imageData:", numpy.array(imageData).shape
#        
#        resizedData = ImageProcessingUtil.resize(imageData,(17,128))
#        #print "Shape resizedData:", numpy.array(resizedData).shape
#        
#        cv2.imwrite("temp.png", resizedData)          
#        readedImageData = cv2.imread("temp.png")
#        #cv2.imwrite("/informatik2/wtm/home/barros/Workspace/MCCNN2/temp2.png", readedImageData)
#        
#        
#        #print "Shape readedImageData:", numpy.array(readedImageData).shape
#        #raw_input("")
#                        
#        spectrums.append(readedImageData)      
                             
       
    #return spectrums
        

def extractMFCCSingle(wavDirectory, deltaOrder, sliceSize):
    samplerate, data = scipy.io.wavfile.read(wavDirectory)
    
#    print "File:", wavDirectory    
#    print "SampleRate:", samplerate
#    print "Data Lenght:", len(data)
#    print "Seconds:", len(data) / samplerate     
            
    totalDuration = len(data) / samplerate 
    
    sliceSize = 1    
    
#    spectrum = mfcc(data,samplerate, winstep=0.5, winlen=0.25)
#    spectrum.resize(spectrum,(21,13))
#    spectrum = spectrum.reshape(1, 13,21)
#    print "Shape spectrum:", numpy.array(spectrum).shape
#    raw_input("here")
    
    spectrums = []
#    spectrums.append(spectrum)
    
    for i in range(totalDuration/sliceSize):
        sliceFrom = i*sliceSize*samplerate
        sliceTo = sliceSize*(i+1)*samplerate
        
        dataSliced = data[sliceFrom:sliceTo]
        dataSliced = preemphasis(dataSliced)
       
        dataSliced = signal.resample(dataSliced, 16000)
        #spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)
        #print "SampleRate:", samplerate
        frames = framesig(dataSliced, 16000*0.025, 16000*0.01)
        spectrum = logpowspec(frames,2048)
        #spectrum = magspec(frames, 2048)      
        if deltaOrder == 1:
            spectrum = delta(spectrum, order=1)
        elif deltaOrder == 2:
            spectrum = delta(spectrum, order=2)
            
        #spectrum = 20*numpy.log10(abs(spectrum))
        spectrum = convertFloatImage(spectrum.T)  
        
        #cv2.imwrite("/data/datasets/mini-genres/test2.png", spectrum)          
        #spectrum = cv2.imread("/data/datasets/mini-genres/test2.png")         
        #spectrum = cv2.cvtColor(spectrum,cv2.COLOR_BGR2GRAY)
#        print "Size:", numpy.array(spectrum).shape            
#        raw_input("here")
        
        spectrum = cv2.resize(spectrum,(17,128))
        spectrum = spectrum.reshape(1, 128,17)
                        
        spectrums.append(spectrum)         
        
    #raw_input("here")
    return spectrums
    
def extractMFCC(wavDirectory, saveDirectory, deltaOrder, sliceSize):
    samplerate, data = scipy.io.wavfile.read(wavDirectory)
    
    #print "SampleRate:", samplerate
    #print "Data Lenght:", len(data)
    #print "Seconds:", len(data) / samplerate     
    
    
#    data = preemphasis(data)
#    spectrum = mfcc(data,samplerate, nfft=1024)
#    spectrum = convertFloatImage(spectrum.T)            
#    spectrum = cv2.resize(spectrum,(64,64))
#
#    return spectrum
    
    totalDuration = len(data) / samplerate 
    spectrums = []
    for i in range(int(totalDuration/sliceSize)):

        sliceFrom = i*sliceSize*samplerate
        sliceTo = sliceSize*(i+1)*samplerate
        
        dataSliced = data[sliceFrom:sliceTo]
        dataSliced = preemphasis(dataSliced)
       
        dataSliced = signal.resample(dataSliced, 16000)
        #spectrum = mfcc(dataSliced,samplerate, nfft=1024, numcep=26)
        #print "SampleRate:", samplerate
        frames = framesig(dataSliced, 16000*0.025, 16000*0.01)
        spectrum = logpowspec(frames,2048)
        #spectrum = magspec(frames, 2048)
      
        if deltaOrder == 1:
            spectrum = delta(spectrum, order=1)
        elif deltaOrder == 2:
            spectrum = delta(spectrum, order=2)
            
        #spectrum = 20*numpy.log10(abs(spectrum))
        spectrum = convertFloatImage(spectrum.T)  
        
        cv2.imwrite("/data/datasets/mini-genres/test2.png", spectrum)          
        spectrum = cv2.imread("/data/datasets/mini-genres/test2.png")         
        #spectrum = cv2.cvtColor(spectrum,cv2.COLOR_BGR2GRAY)
#        print "Size:", numpy.array(spectrum).shape            
#        raw_input("here")
        
        spectrum = cv2.resize(spectrum,(17,128))
        spectrums.append(spectrum)      
        
        
    #raw_input("here")
    return spectrums
    
    #spectrum = cv2.resize(spectrum,(640,480))
    #cv2.imwrite(saveDirectory, spectrum)
    #print saveDirectory
    
#    spectrum = convertFloatImage(fbank_feat)
#    spectrum = cv2.resize(spectrum,(640,480))
#    cv2.imwrite("/data/datasets/mini-genres/test2.png", spectrum)
#    
#    print "featureSize:", numpy.array(mfcc_feat).shape
#    print "featureSize:", numpy.array(fbank_feat).shape
    
    


def convertFloatImage(image):
    scale = numpy.max(numpy.abs(image))
    if numpy.any(image < 0):
        result = 255. * ((0.5 * image / scale) + 0.5)
    else:
        result = 255. * image / scale
    return result.astype(numpy.uint8)

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n
    
    
def createSpectrum(wavDirectory):
    samplerate, data = scipy.io.wavfile.read(wavDirectory)
    
        
    data = numpy.swapaxes(data,0,1)    
    
#    Nx = len(data)
#    nsc = numpy.floor(Nx/4.5)
#    nov = numpy.floor(nsc/2);
#    nff = max(256,2^nextpow2(nsc));
#    
#    specgram(x, NFFT=256, Fs=2,detrend=mlab.detrend_none,
#        window=mlab.window_hanning, noverlap=128,
#        cmap=None, xextent=None, pad_to=None, sides='default',
#        scale_by_freq=None, mode='default')
        
        
    #pylab.figure(frameon=False)    
    fig, (ax1) = pylab.subplots(nrows=1)
    spectrum, freqs, bins, im = ax1.specgram(data,NFFT=1024, noverlap=50);
    pylab.close()
    
    #print "Shape:", numpy.array(spectrum).shape    
    
    #spectrum = cv2.resize(spectrum,(64,48))  
    #print img
    
    # First, design the Buterworth filter
    N  = 1    # Filter order
    Wn = 0.01 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    #spectrum = signal.filtfilt(B,A, spectrum)
    
    spectrum[spectrum==0] = 1
    #spectrum = convertFloatImage(spectrum)
    spectrum = 20*numpy.log10(abs(spectrum))
    return spectrum
    

    
    
def createSpectrumImage(wavDirectory):
    samplerate, data = scipy.io.wavfile.read(wavDirectory)
    
    data = numpy.swapaxes(data,0,1)    
    
    #pylab.figure(frameon=False)    
    fig, (ax1) = pylab.subplots(nrows=1)
    spectrum, freqs, bins, im = ax1.specgram(data) 
    pylab.close()
    
    # First, design the Buterworth filter
    N  = 1    # Filter order
    Wn = 0.01 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    spectrum = signal.filtfilt(B,A, spectrum)
        
    #print "Shape:", numpy.array(spectrum).shape    
    
    #spectrum = cv2.resize(spectrum,(64,48))  
    #print img
    
    #spectrum[spectrum==0] = 1
    #spectrum = convertFloatImage(spectrum)
    spectrum = 20*numpy.log10(abs(spectrum))
        
    spectrum = convertFloatImage(spectrum)
    return spectrum    
    
    #numpy.savetxt("/data/datasets/Audio_Numbers_Pablo/file.txt", spectrum)
        
    
#    print numpy.array(img).shape
    print saveDirectory
    #spectrum = cv2.resize(spectrum, (640,480))
    #cv2.imwrite(saveDirectory, spectrum)


#sequenceSizeInSeconds = 1
# 
#sourceFolder = "//data/datasets/SAVEE/AudioPerSubject/separated/DC/"
#destinationFolder = "/data/datasets/SAVEE/ALL/POW_DC_2048_"+str(sequenceSizeInSeconds)
#
#for separation in os.listdir(sourceFolder):
#    for c in os.listdir(sourceFolder+"/"+separation):
#        sampleNumber = 0
#        
#        for audioFile in os.listdir(sourceFolder+"/"+separation+"/"+c):
#            audioSource = sourceFolder+"/"+separation+"/"+c+"/"+audioFile
#            
#            #audioDestination = destinationFolder+"/"+c+"/"+str(sampleNumber)
#            audioDestination = destinationFolder+"/"+separation+"/"+c+"/"
#               
#            if not os.path.exists(audioDestination):            
#                    os.makedirs(audioDestination)
#                    
#            print "audioSource", audioSource
#            spectrum = extractMFCC(audioSource, audioDestination, 0, sequenceSizeInSeconds)
##            spectrum = graph_spectrogram(audioSource)
#            
##            audioDestination = destinationFolder+"/"+separation+"/"+c+"/"+str(sampleNumber)+".png"  
###            #spectrum = createSpectrumImage(audioSource)
##            cv2.imwrite(audioDestination, spectrum)
#            
#            for i in range(len(spectrum)):                
#                audioDestination = destinationFolder+"/"+separation+"/"+c+"/"+str(sampleNumber)+str(i)+".png"                
#                #print "Shape:", spectrum[i]
#                cv2.imwrite(audioDestination, spectrum[i])
#            
#    #        for i in range(3):
#    #            audioDestination = destinationFolder+"/"+c+"/"+str(sampleNumber)+"/"+str(i)+".png"
#                            
#                            #createSpectrumImage(audioSource, audioDestination)
#    #            spectrum = extractMFCC(audioSource, audioDestination, i)
#    #            cv2.imwrite(audioDestination, spectrum)
#            
#            #extractMFCCLibrosa(audioSource, audioDestination)
#            sampleNumber = sampleNumber+1
#        
##    raw_input("here")                

#audio = "/data/datasets/Audio_Numbers_Pablo/audio_wav/dois/Recording8"    
#save = "/data/datasets/Audio_Numbers_Pablo/cinco_8.png"

#createSpectrumImage(audio, save)



def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('/informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/spectrogram.png')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
    

def createSpectrumImage2(directory):
    samplerate, data = scipy.io.wavfile.read(directory)
    print "samplerate=", samplerate
    print "shape(data)=", numpy.shape(data)
    print "data=", data


    spectrum, freqs, bins, im = pylab.specgram(data) 
#    print numpy.shape(spectrum), numpy.shape(freqs), numpy.shape(bins), im
    
    
    spectrum_norm = numpy.floor(numpy.log10(spectrum+1) / numpy.max(numpy.log10(spectrum+1)) * 255)
#    print "spectrum_norm=", spectrum_norm
    
    return numpy.array(spectrum_norm)
    
    #cv2.imwrite("/informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/spectrum5.pnm", numpy.array(spectrum_norm))
#    f = open("/informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/spectrum2.pnm", 'wb')
#    f.write("P2\n")
#    f.write('%d %d\n' % (numpy.shape(spectrum_norm)[1],numpy.shape(spectrum_norm)[0]))
#    f.write("255\n")
#    for i in range(numpy.shape(spectrum_norm)[0]):
#        for j in range(numpy.shape(spectrum_norm)[1]):
#            f.write("%d " % spectrum_norm[i][j])
#    f.close()
    
    
#img = cv2.imread("/informatik2/wtm/home/barros/Documents/Experiments/avLetters/data/avletters/Audio/mfcc/Clean/A1_Anya.mfcc") 
#
#f = file("/informatik2/wtm/home/barros/Documents/Experiments/avLetters/data/avletters/Audio/mfcc/Clean/A1_Anya.mfcc", 'rb')
#firtLine = f.readline()
#print "F:", firtLine
##print "Shape:", numpy.array(f.read()).shape
#
#   
#    
##graph_spectrogram("/informatik2/wtm/home/barros/Documents/Experiments/avLetters/data/avletters/Audio/mfcc/Clean/A1_Anya.mfcc")
    