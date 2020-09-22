import scipy.io.wavfile
import scipy.signal
import scipy
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi
from scipy.interpolate import InterpolatedUnivariateSpline

F = 30000         # Carrier frequency
section = 0.005   # Audio section
N = 4             # Times of sample rate

# Function that calculate fourier transformation
def fourier(signal, dt):
    FFT = scipy.fftpack.fft(signal)
    FFT = scipy.fftpack.fftshift(FFT)
    freqs = scipy.fftpack.fftfreq(len(signal), dt)
    freqs = scipy.fftpack.fftshift(freqs)
    return abs(FFT), freqs

# Function that show a graphic
def showGraph(x, y, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

# Function that interpolate a signal
def interpolate(x, y, x0, xf, xsample):
    inter = InterpolatedUnivariateSpline(x, y, k=3)
    xInter = np.linspace(x0, xf, xsample)
    yInter = inter(xInter)
    return xInter, yInter

# Function that read a audio
def inputSignal(name):
    return scipy.io.wavfile.read(name)

# Function that applies a filter low pass
def filterLowPass(signal, fc):
    nyq = 0.5 * N * F
    w = fc / nyq
    b, a = scipy.signal.butter(9, w, 'low')
    filtered = scipy.signal.lfilter(b, a, signal)
    return filtered

if __name__ == '__main__':

    # Read input signal
    rate, signal = inputSignal('handel.wav')

    # Calculation of necessary parameters
    muestra = len(signal)
    dt = 1/(N*F)
    sampleRate = int(N*F*muestra/rate)
    time = np.arange(0, muestra / rate, 1.0 / rate)

    # Interpolation of the signal
    time, signal = interpolate(time, signal, 0, muestra / rate, sampleRate)

    # Input signal section
    timeSection = time[0:int(N * F * section)]
    signalSection = signal[0:int(N * F * section)]

    # Input signal section graph
    showGraph(timeSection, signalSection,
              title='Section of the input signal',
              xlabel='Time',
              ylabel='Amplitude',
              ylim=(-7000, 7000))

    # Fourier of the input signal
    FFT, freqs = fourier(signal, dt)
    showGraph(freqs, FFT,
              title='Fourier transform of the input signal',
              xlabel='Frequency',
              xlim=(-20000, 20000))


    # Carrier
    carrier = cos(2*pi*F*time)

    # Carrier section
    carrierSection = carrier[0:len(timeSection)]
    showGraph(timeSection, carrierSection,
              title='Section of the carrier',
              xlabel='Time',
              ylabel='Amplitude')

    # Fourier of the carrier
    FFTCar, freqsCar = fourier(carrier, dt)
    showGraph(freqsCar, FFTCar,
              title='Fourier transform of the carrier',
              xlabel='Frequency')

    # Modulated
    modulated = signal * carrier

    # Modulated signal section
    modulatedSection = modulated[0:len(timeSection)]
    showGraph(timeSection, modulatedSection,
              title='Section of the modulated signal',
              xlabel='Time',
              ylabel='Amplitude',
              ylim=(-7000, 7000))

    # Fourier of the modulated signal
    FFTMod, freqsMod = fourier(modulated, dt)
    showGraph(freqsMod, FFTMod,
              title='Fourier transform of the modulated signal',
              xlabel='Frequency')

    # Demodulated
    demodulated = modulated * carrier

    # Demodulated signal section
    demodulatedSection = demodulated[0:len(timeSection)]
    showGraph(timeSection, demodulatedSection,
              title='Section of the demodulated signal',
              xlabel='Time',
              ylabel='Amplitude',
              ylim=(-7000, 7000))

    # Fourier of the demodulated signal
    FFTDeMod, freqsDeMod = fourier(demodulated, dt)
    showGraph(freqsDeMod, FFTDeMod,
              title='Fourier transform of the demodulated signal',
              xlabel='Frequency')

    # Filter to demodulated signal
    fc = 10000
    filtered = filterLowPass(demodulated, fc)

    # Fourier of the filter to demodulated signal
    FFTFil, freqsFil = fourier(filtered, dt)
    showGraph(freqsFil, FFTFil,
              title='Fourier transform of the filter to demodulated signal',
              xlabel='Frequency',
              xlim=(-20000, 20000))