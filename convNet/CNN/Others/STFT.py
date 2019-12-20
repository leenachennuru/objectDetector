# -*- coding: utf-8 -*-

import scipy, pylab
import numpy
import scipy.io.wavfile
import os
import cv2


def convertFloatImage(image):
    scale = numpy.max(numpy.abs(image))
    if numpy.any(image < 0):
        result = 255. * ((0.5 * image / scale) + 0.5)
    else:
        result = 255. * image / scale
    return result.astype(numpy.uint8)
    

def stft(x, fs, framesz, hop):
    """x is the time-domain signal
    fs is the sampling frequency
    framesz is the frame size, in seconds
    hop is the the time between the start of consecutive frames, in seconds
    """
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    """X is the short-time Fourier transform
    fs is the sampling frequency
    T is the total length of the time-domain output in seconds
    hop is the the time between the start of consecutive frames, in seconds
    """
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

if __name__ == '__main__':
    f0 = 440         # Compute the STFT of a 440 Hz sinusoid
    fs = 2        # sampled at 8 kHz
    T = 1            # lasting 5 seconds
    framesz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.050      # and hop size of 20 milliseconds.
 
    # Create test signal and STFT.
    t = scipy.linspace(0, T, T*fs, endpoint=False)
    x = scipy.sin(2*scipy.pi*f0*t)
    
   
    sourceFolder = "/data/datasets/Audio_Numbers_Pablo/audio_wav/"
    destinationFolder = "/data/datasets/Audio_Numbers_Pablo/spectogram_SFTT" 
    
    for c in os.listdir(sourceFolder):
        sampleNumber = 0
        for audioFile in os.listdir(sourceFolder+"/"+c):
            audioSource = sourceFolder+"/"+c+"/"+audioFile
            
            audioDestination = destinationFolder+"/"+c+"/"
            
            if not os.path.exists(audioDestination):            
                os.makedirs(audioDestination)
            
            audioDestination = destinationFolder+"/"+c+"/"+str(sampleNumber)+".png"
            
            samplerate, data = scipy.io.wavfile.read(audioSource)            
            fs = samplerate
            data = numpy.swapaxes(data,0,1)    
            data = data[0]    
    
            X = stft(data, fs, framesz, hop)
            
            spectrum = convertFloatImage(X.T)
    
            print numpy.array(X).shape
            print numpy.array(scipy.absolute(X.T)).shape
            
            img = cv2.resize(spectrum,(64,48))  
            
            #img = cv2.resize(spectrum,(650,500))  
    
    
        #    print numpy.array(img).shape
            print audioDestination
            cv2.imwrite(audioDestination, img)
            
            
#            fig, (ax1) = pylab.subplots(nrows=1)
#            ax1.set_axis_off()
#            ax1.axes.get_xaxis().set_visible(False)
#            ax1.axes.get_yaxis().set_visible(False)
#            
#        
#            
#            spectrum2 = 20*numpy.log10(scipy.absolute(X.T))
#            # Plot the magnitude spectrogram.
#            pylab.figure()
#            pylab.imshow(spectrum2, origin='lower', aspect='auto',
#                         interpolation='nearest')
#            
#            #pylab.show()
#            pylab.savefig(audioDestination, bbox_inches='tight', pad_inches = 0)
#            pylab.close()
            sampleNumber = sampleNumber+1
    
    

#    # Compute the ISTFT.
#    xhat = istft(X, fs, T, hop)
#
#    # Plot the input and output signals over 0.1 seconds.
#    T1 = int(T*fs)
#
#    pylab.figure()
#    pylab.plot(t[:T1], x[:T1], t[:T1], xhat[:T1])
#    pylab.xlabel('Time (seconds)')
#
#    pylab.figure()
#    pylab.plot(t[-T1:], x[-T1:], t[-T1:], xhat[-T1:])
#    pylab.xlabel('Time (seconds)')

