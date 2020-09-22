####  Gabriel Bustamante Toledo
####  Redes de computadores
####  Laboratorio 1
import scipy.io.wavfile
import scipy.signal as signal
import scipy
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt

#########  Apertura #############
# Es solicitado el nombre del archivo de audio con su extension apropiada.
archivo = input('\n\nArchivo de sonido con su extension: ' )

# El archivo de audio es abierto, el cual retorna dos valores almacenados inmediatamente.
try:
    muestreo, senal = scipy.io.wavfile.read(archivo)
except:
    print("\n\n\nEl archivo de audio no pudo ser abierto," + 
          " verifica el nombre que has ingresado y si el " + 
          "archivo de audio esta disponible.\n")
    exit()
    
#####  Parametros de la senal.
muestra = len(senal)
dt = 1.0/muestreo
tiempo = np.arange(0, muestra*dt , dt)
##### Grafica senal.
plt.plot( tiempo, senal, lw = 0.4)
plt.title('Senal de audio ingresada')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.show()


##### Calculo de la T. de Fourier.
FFT = scipy.fftpack.fft(senal)
freqs = scipy.fftpack.fftfreq( len( abs(FFT) ), dt)
##### Grafica de la T. de Fourier.
plt.plot( freqs, abs(FFT), lw = 0.4)
plt.title('T. de Fourier')
plt.xlabel('Frecuencia')
plt.show()


##### Calculo de la T.I. de Fourier.
IFFT = scipy.fftpack.ifft(FFT)
#####  Grafica T.I. de Fourier.
plt.plot( tiempo, IFFT, lw = 0.4)
plt.title('T.I. de Fourier')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.show()


##### Obtencion del espectrograma.
f, t, Sxx = signal.stft( senal, muestreo)
#####  Grafica del espectrograma.
plt.pcolormesh(t, f, abs(Sxx), vmin=0, vmax=5000)
plt.title('Espectrograma del audio ingresado')
plt.ylabel('Frequencia')
plt.xlabel('Tiempo')
plt.show()



###################    SENAL FILTRADA    #######################

#### Creacion del filtro.
fc = 1700
w = fc / (muestreo / 2)
b, a = signal.butter(9, w, 'low')


#### Creacion de la senal filtrada.
output = signal.lfilter(b, a, senal)
#### Grafica senal filtrada.
plt.plot( tiempo, output, lw = 0.4)
plt.title('Senal de audio filtrado')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.show()


#### Calculo de T. de Fourier a senal filtrada.
FFTfilt = scipy.fftpack.fft(output)
freqsfilt = scipy.fftpack.fftfreq( len( abs(FFTfilt) ), (1.0/muestreo))
#### Grafica de T. de Fourier a senal filtrada.
plt.plot( freqsfilt, abs(FFTfilt) , lw = 0.4)
plt.title('T. de Fourier filtrada')
plt.xlabel('Frecuencia')
plt.show()


#### Obtencion del espectrograma de la senal filtrada.
ffilt, tfilt, Sxxfilt = signal.stft( output, muestreo)
#### Grafica del espectrograma de la senal fitlrada.
plt.pcolormesh(tfilt, ffilt, abs(Sxxfilt), vmin=0, vmax=5000)
plt.title('Espectrograma del audio filtrado')
plt.ylabel('Frequencia')
plt.xlabel('Tiempo')
plt.show()


#### Guardado de senal filtrada.
scipy.io.wavfile.write("filtrada.wav", muestreo, output)