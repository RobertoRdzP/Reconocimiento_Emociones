import math
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wave
from scipy import stats
from scipy.fftpack import dct
from scipy.fft import rfft, rfftfreq, irfft

#Lectura de archvios .wav
def leerArchivo(nombre):
    r, d = wave.read(nombre)
    n = len(d)
    return r, d, n

#Metodo para graficar
def graficar(dato):
    plt.plot(dato)
    plt.show()

#Metodo de la Trannsformada rapida de Fourier
def tr_Fourier(data, rate, n):
    yf = rfft(data)
    yf = yf[0:int(len(yf)/2)]
    x = irfft(yf)
    graficar(x)
    xf = rfftfreq(n, 1 / rate)

    return yf, xf

#Normaizacion de silencio, |NO SE USA|
def norm_silencio(data_n):
    data_abs = np.abs(data_n)
    i = 0

    while i != (len(data_abs)-30):
        i += 1
        bool_alto = 1
        for j in range(29):
            if data_abs[i+j] > 30:
                bool_alto = 0
        if bool_alto :
            data_abs[i] = 0
            data_n[i] = 0
        else:
            i += 29

    return data_n

def no_silencio(data_n):
    data_abs = np.abs(data_n)
    i = 0

    while i != (len(data_abs)-30):
        i += 1
        bool_alto = 1
        for j in range(29):
            if data_abs[i+j] > 30:
                bool_alto = 0
        if bool_alto :
            data_abs[i] = 0
            data_n[i] = 0
        else:
            i += 29

    return data_n

def recorte_audio(data_n):
    data_abs = np.abs(data_n)
    cont_sil = 0
    n_com = 0
    n_fin = 0

    for i in range(len(data_abs)):
        if data_abs[i] > 50:
            cont_sil += 1
            if cont_sil == 1000:
                n_com = i-1000
                break


    cont_sil = 0

    for i in range(n_com, len(data_abs)):
        if data_abs[i] < 50:
            cont_sil += 1
            if cont_sil == 10000:
                n_fin = i-10000
                break
        else:
            cont_sil = 0

    data_n = data_n[n_com:n_fin]

    return data_n

#Pre-enfasis se aplica un filtro de pre enfasis para acentuar las señales altas
def pre_enfasis(data_n):
    var_pre = 0.97


    señal_enf = numpy.append(data_n[1], data_n[1:] - var_pre * data_n[:-1])

    enfa_x = numpy.arange(0, señal_enf.size, 1)

    return señal_enf, enfa_x

def GFB(data_n, rate_n ):
    print("22222")
    print(data_n)
    print(rate_n)
    f_stride = 0.01
    f_size = 0.025
    f_length, f_step = f_size * rate_n, rate_n * f_stride
    tam_data = len(data_n)
    f_length = int(round(f_length))
    f_step = int(round(f_step))
    num_frames = int(numpy.ceil(float(numpy.abs(tam_data - f_length)) /  f_step))

    pad_signal_length = num_frames * f_step + f_length
    z = numpy.zeros((pad_signal_length - tam_data))
    pad_signal = numpy.append(data_n, z)

    indices = numpy.tile(numpy.arange(0, f_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * f_step, f_step), (f_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    frames *= numpy.hamming(f_length)

    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    ##############################

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (rate_n  / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / rate_n )

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
    plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1, extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])  # Figura
    plt.xlabel("Frames", fontsize=14)
    plt.ylabel("Dimension", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('filter_banks.png')
    plt.show()

    #       Coeficientes cepstrales de frecuencia Mel (MFCC)
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

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

    # Normalización media

    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

    #plt.savefig("filter_banks_mean.png")
    plt.title("filter_banks_mean")
    plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=1,
               extent=[0, filter_banks.shape[0], 0, filter_banks.shape[1]])  # Figura
    plt.xlabel("Frames", fontsize=14)
    plt.ylabel("Dimension", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('filter_banks_mean.png')
    #plt.show()

    plt.title("mfcc_mean")
    plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=10,
               extent=[0, mfcc.shape[0], 0, mfcc.shape[1]])  # Figura
    plt.ylabel("Frames", fontsize=14)
    plt.xlabel("Dimension", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('mfcc_mean.png')
    plt.show()

    print(mfcc.shape)

#Main___________________________________________________________________________________________________________________

#Leemos el archivo, retorna el rate, data, y tamaño
#rate, data, n_data = leerArchivo("C:/Users/rober/PycharmProjects/Programa_Tesis/venv/Audios/03-01-08-02-02-02-02.wav")
rate, data, n_data = leerArchivo("C:/Users/rober/PycharmProjects/Programa_Tesis/venv/Audios/Actor_01/03-01-01-01-01-01-01.wav")

#Graficamos el audio
graficar(data)

#Impementamos la transformada rapida
yf, xf = tr_Fourier(data, rate, n_data)

graficar(yf)

#Normalizamos el silencio
#data_mod = norm_silencio(data)

#Recorte de audio
data_mod2 =  recorte_audio(data)

#graficar(data_mod2)

#Se manda a pre enfatizar la señal
data_enf, enf_x = pre_enfasis(data_mod2)

#graficar(data_enf)

#Se crea el  ventaneo
GFB(data_enf, rate)



