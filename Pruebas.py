import numpy
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

sample_rate, signal = wavfile.read("C:/Users/rober/PycharmProjects/Programa_Tesis/venv/Audios/03-01-01-01-01-01-01.wav")
signal = signal[0:int(3.5 * sample_rate)]

axis_x = numpy.arange(0, signal.size, 1)
plt.plot(axis_x, signal, linewidth=5)
plt.title("Time domain plot")
plt.xlabel("Time", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('Time domain plot.png')
plt.show()

"""
Pre-énfasis
 El primer paso es aplicar un filtro de preacentuación a la señal para amplificar las altas frecuencias. El filtro de énfasis previo es útil de varias formas:
 (1) Equilibre el espectro, porque las frecuencias altas suelen tener una amplitud menor que las frecuencias bajas;
 (2) Evite problemas numéricos durante la operación de transformada de Fourier;
 (3) También puede mejorar la relación señal-ruido (SNR).
 El filtro de preacentuación se puede aplicar a la señal x utilizando el filtro de primer orden en la siguiente fórmula:
            y(t)=x(t) -αx(t-1)
 Se puede implementar fácilmente utilizando las siguientes líneas de código, donde el valor típico del coeficiente de filtro (α) es 0,95 o 0,97,
"""
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

axis_x = numpy.arange(0, emphasized_signal.size, 1)
plt.plot(axis_x, emphasized_signal, linewidth=5)
plt.title("Pre-Emphasis")
plt.xlabel("Time", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig("Pre-Emphasis.png")
plt.show()

"""
 Después del pre-énfasis, necesitamos dividir la señal en cuadros cortos. El principio básico de este paso es que la frecuencia de la señal cambiará con el tiempo.
 Por lo tanto, en la mayoría de los casos, la transformada de Fourier de toda la señal no tiene sentido.
 Porque perdemos el perfil de frecuencia de la señal con el tiempo. 
 Para evitar esta situación, podemos asumir que la frecuencia de la señal se fija en poco tiempo. 
 Por lo tanto, al realizar la transformada de Fourier en esta trama corta, se puede obtener una buena aproximación del perfil de frecuencia de la señal concatenando tramas adyacentes.
 El tamaño de cuadro típico en el procesamiento de voz es de 20 ms a 40 ms, con un 50% (+/- 10%) de superposición entre cuadros consecutivos. 
 Los ajustes comunes son un tamaño de fotograma de 25 milisegundos, frame_size = 0.025 y un intervalo de 10 milisegundos (superpuesto 15 milisegundos),
"""
frame_stride = 0.01
frame_size = 0.025
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(
    numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z)
# Llene la señal para asegurarse de que todos los fotogramas tengan el mismo número de muestras sin truncar ninguna muestra en la señal original

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
    numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

"""
 Después de cortar la señal en cuadros, aplicamos funciones de ventana como la ventana de Hamming a cada cuadro. La ventana de Hamming tiene la siguiente forma:
            w[n]=0.54-0.46cos(2*pi*n/(N-1))
 Donde 0 <= n <= N-1, N es la longitud de la ventana
 Hay muchas razones para aplicar la función de ventana a estos marcos, especialmente para compensar el cálculo de FFT infinito y reducir la fuga de espectro.
"""
frames *= numpy.hamming(frame_length)
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

# Transformada Fourier y espectro de potencia
"""
 Ahora, podemos realizar FFT de N puntos en cada cuadro para calcular el espectro de frecuencia, que también se llama transformada de Fourier de tiempo corto (STFT),
 Donde N suele ser 256 o 512, NFFT = 512; luego use la siguiente fórmula para calcular el espectro de potencia (periodograma):
            P=|FFT(xi)|^2/N
 Entre ellos, xi es el i-ésimo fotograma de la señal x. Esto se puede lograr con las siguientes líneas:
"""
NFFT = 512
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

# Filter Group Filter Banks
"""
 El último paso para calcular el banco de filtros es aplicar un filtro triangular (generalmente 40 filtros, nfilt = 40 en el nivel Mel) al espectro de potencia para extraer la banda de frecuencia. 
 El propósito de la escala Mel es imitar la percepción del sonido por el oído humano a bajas frecuencias al ser más discriminativo a bajas frecuencias y menos discriminativo a altas frecuencias.
   Podemos usar la siguiente fórmula para convertir entre Hertz (f) y Mel (m):
            m = 2595log10(1+f/700)
            f = 700*(10^(m/2595)-1)
"""
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])  # left
    f_m = int(bin[m])  # center
    f_m_plus = int(bin[m + 1])  # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB

plt.title("filter_banks")
plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1,extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])  # Figura
plt.xlabel("Frames", fontsize=14)
plt.ylabel("Dimension", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('filter_banks.png')
plt.show()

#       Coeficientes cepstrales de frecuencia Mel (MFCC)
"""
 Resulta que los coeficientes del banco de filtros calculados en el paso anterior están altamente correlacionados, lo que puede causar problemas en algunos algoritmos de aprendizaje automático. 
 Por lo tanto, podemos aplicar la Transformada Discreta de Coseno (DCT) para descorrelacionar los coeficientes del banco de filtros y producir una representación comprimida del banco de filtros. 
 Generalmente, para el reconocimiento automático de voz (ASR), los coeficientes cepstrum resultantes 2-13 se mantendrán y el resto se descartará; num_ceps = 12.
 La razón para descartar otros coeficientes es que representan cambios rápidos en los coeficientes del banco de filtros, y estos detalles sutiles no son útiles para el reconocimiento automático de voz (ASR).
"""
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

"""
 El amplificador sinusoidal 1 se puede aplicar a MFCC para enfatizar MFCC excesivamente altos, lo que puede mejorar el reconocimiento de voz en señales ruidosas.
"""
cep_lifter = 22
(nframes, ncoeff) = mfcc.shape
n = numpy.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
mfcc *= lift  # *

plt.title("mfcc")
plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.05,
           extent=[0, mfcc.shape[1], 0, mfcc.shape[0]])  # Figura
plt.xlabel("Frames", fontsize=14)
plt.ylabel("Dimension", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('mfcc.png')
plt.show()

# Normalización media Normalización media
"""
 Como se mencionó anteriormente, para equilibrar el espectro y mejorar la relación señal / ruido (SNR), simplemente podemos restar el valor promedio de cada coeficiente de todos los cuadros.
"""
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

plt.savefig("filter_banks_mean.png")
plt.title("filter_banks_mean")
plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1,
           extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])  # Figura
plt.xlabel("Frames", fontsize=14)
plt.ylabel("Dimension", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('filter_banks_mean.png')
plt.show()

plt.title("mfcc_mean")
plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.05,
           extent=[0, mfcc.shape[1], 0, mfcc.shape[0]])  # Figura
plt.xlabel("Frames", fontsize=14)
plt.ylabel("Dimension", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('mfcc_mean.png')
plt.show()