import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt
import random
import scipy.integrate as integrate
import math
import sys

FREQUENCY = 6000  # Signal Frequency.
K = 4             # Sampling rate.

# Function that generates a random array of bits.
# Input: Long of array.
# Output: Random array of bits.
def generateInput(long):
    bits = []
    i = 0
    while i < long:
        bits.append(random.randint(0, 1))
        i += 1
    return bits

# Function that makes a digital modulation OOK.
# Input: Amplitude, bit rate, array of bits.
# Output: Modulated signal.
def modulator(A, rate, input):
    time = np.arange(0, 1 / rate, step=1/(K * FREQUENCY))
    signal0 = 0 * cos(2 * pi * FREQUENCY * time)
    signal1 = A * cos(2 * pi * FREQUENCY * time)

    signal = []

    t = 0
    while t < len(input):
        if input[t] == 0:
            signal.append(signal0)
        elif input[t] == 1:
            signal.append(signal1)
        t += 1

    realsignal = np.linspace(0, len(signal) * len(signal[0]), len(signal) * len(signal[0]))
    i = 0
    l = 0
    while i < len(signal):
        j = 0
        while j < len(signal[i]):
            realsignal[l] = signal[i][j]
            j += 1
            l += 1
        i += 1
    return realsignal

# Function that generate AWGN.
# Input: Power of signal, SNR lineal, long of signal.
# output: AWGN.
def AWGN(power, SNRlineal, long):
    variance = math.sqrt(power / SNRlineal)
    noise = variance * np.random.normal(0, 1, long)
    return noise

# Function that calculate a signal power.
# Input: Signal, array time, time.
# output: Signal power.
def signalPower(signal, time, T):
    square = signal ** 2
    energy = integrate.simps(square, time)
    power = (1 / T) * energy
    return power

# Function that transforms SNR db to SNR lineal.
# Input: SNR db.
# Output: SNR lineal.
def SNRdb2lineal(SNR):
    SNRlineal = 10 ** (SNR / 10)
    return SNRlineal

# Function that simulate a real channel.
# Input: Signal, noise.
# Output: Real signal.
def channel(signal, noise):
    return noise + signal

# Function that demodulated a signal OOK.
# Input: Signal, bit rate, amplitude, bit amount.
# Output: Array of bits.
def demodulator(signal, rate, A, data):
    read = []
    i = 0
    while i < data:
        section = signal[int((i * K * FREQUENCY) / rate): int((i * K * FREQUENCY) / rate + (K * FREQUENCY) / rate)]
        avg = np.average(np.absolute(section))
        if avg <= A*0.35:
            read.append(0)
        else:
            read.append(1)
        i += 1
    return read

# Function that count a errors.
# Input: Original input array of bits, demodulated array of bits.
# Output: Number of errors.
def errorsCount(input, read):
    i = 0
    errors = 0
    while i < len(read):
        if read[i] != input[i]:
            errors += 1
        i += 1
    return errors / len(input)

# Function that show a graph.
# Input: Time, signal, title, axis x label, axis y label.
# Output: Graph.
def showGraph(time, signal, title, xlabel, ylabel):
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Section to test the functions.
if len(sys.argv) == 2 and sys.argv[1] == "test":
    long = 5
    bits = generateInput(long)
    SNR = 10
    rate = 500
    A = 10
    T = len(bits) / rate

    signal = modulator(A, rate, bits)
    time = np.linspace(0, T, num=len(signal), endpoint=False)

    showGraph(time, signal,
              title="Señal modulada (Tiempo de bit: 0.002 s )",
              xlabel="Tiempo (s)",
              ylabel="Señal"
              )

    power = signalPower(signal, time, T)
    SNRlineal = SNRdb2lineal(SNR)
    noise = AWGN(power, SNRlineal, len(time))

    showGraph(time, noise,
              title="Rudio blanco gausiano aditivo",
              xlabel="Tiempo (s)",
              ylabel="Ruido"
              )

    realSignal = channel(signal, noise)

    showGraph(time, realSignal,
              title="Señal con ruido",
              xlabel="Tiempo (s)",
              ylabel="Señal"
              )

    read = demodulator(realSignal, rate, A, long)

    print("Bits originales: ", bits)
    print("Bits demodulados: ", read)

    exit()

if __name__ == "__main__":
    long = 100000
    bits = generateInput(long)
    SNRlist = np.arange(-2,9, step=0.5)
    rateList = [1000, 2000, 3000]

    errors = np.empty((len(rateList), len(SNRlist)))

    A = 10
    i = 0
    while i < len(rateList):
        rate = rateList[i]
        T = len(bits) / rate
        print("-------------", rate,"-------------")
        j = 0
        while j < len(SNRlist) :
            SNRdb = SNRlist[j]
            SNRlineal = SNRdb2lineal(SNRdb)
            print("             ", SNRdb, "             ")
            signal = modulator(A, rate, bits)
            time = np.linspace(0, T, num=len(signal), endpoint=False)
            power = signalPower(signal, time, T)
            noise = AWGN(power, SNRlineal, len(time))
            realSignal = channel(signal, noise)
            read = demodulator(realSignal, rate, A, long)
            errors[i][j] = errorsCount(bits, read)
            j += 1
        i += 1

    i = 0
    while i < len(errors):
        plt.plot(SNRlist, errors[i], 'o-', label=rateList[i])
        i += 1
    plt.yscale('log')
    plt.title("BER vs SNR (db)")
    plt.xlabel("SNR (db)")
    plt.ylabel("BER")
    plt.legend()
    plt.show()

