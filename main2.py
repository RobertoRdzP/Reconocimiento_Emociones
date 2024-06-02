import math
import os
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wave
from scipy import stats
from scipy.fftpack import dct
from scipy.fft import rfft, rfftfreq, irfft
import time
import statistics
import pandas as pd

#Lectura de archvios .wav
def leer_Audios(carpetas):
    tup_aux = [0,0,0]
    d_audios = []
    v_emocion = []
    ubicacion = "C:/Users/rober/PycharmProjects/Programa_Tesis/venv/Audios/"  #+ list_audios[1])

    for j in range(len(carpetas)):
        actor_audios = os.listdir(ubicacion + carpetas[j])
        for i in range(len(actor_audios)):
            tup_aux[0], tup_aux[1] = wave.read(ubicacion + carpetas[j]+"/"+actor_audios[i])
            tup_aux[2] = len(tup_aux[1])

            #Se lee cual emocion es
            v_aux = actor_audios[i].split("-")
            emocion = int(v_aux[2])
            v_emocion.append(emocion)

            d_audios.append(tup_aux.copy())


    return d_audios, v_emocion

#Metodo para graficar
def graficar(dato):
    plt.plot(dato)
    plt.show()

#Metodo de la Trannsformada rapida de Fourier
def tr_Fourier(data, rate, n):
    yf = rfft(data)
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

def ajuste_audio(data_n, tam):
    mitad = int(len(np.abs(data_n))/2)
    tam_ajuste = int((tam-2)/2)

    data_n = data_n[(mitad-tam_ajuste):(mitad+tam_ajuste)]

    return data_n

def recorte_silencio(data_n):
    data_abs = abs(data_n)
    cont_sil = 0
    n_com = 0
    n_fin = 0

    for i in range(len(data_abs)):
        if isinstance(data_abs[i], numpy.int16) :
            if data_abs[i] > 50:
                n_com = i
                break
        elif isinstance(data_abs[i], numpy.ndarray):
            if data_abs[i][0] > 50:
                n_com = i
                break



    cont_sil = 0
    inicio_for = int(len(data_abs)/2)

    for i in range(inicio_for, len(data_abs)):
        if isinstance(data_abs[i], numpy.int16):
            if data_abs[i] < 50:
                cont_sil += 1
                if cont_sil == 3000:
                    n_fin = i - 3000
                    break
            else:
                cont_sil = 0
        elif isinstance(data_abs[i], numpy.ndarray):
            if data_abs[i][0] < 50:
                cont_sil += 1
                if cont_sil == 3000:
                    n_fin = i - 3000
                    break
            else:
                cont_sil = 0




    data_n = data_n[n_com:n_fin]
    #print(n_com,"---",n_fin)

    return data_n, len(data_n)

def recorte_silencio_alter(data_n):
    data_abs = abs(data_n)
    cont_sil = 0
    n_com = 0
    n_fin = 0

    for i in range(len(data_abs)):
        if isinstance(data_abs[i], numpy.int16):
            if data_abs[i] > 50:
                n_com = i
                break
        elif isinstance(data_abs[i], numpy.ndarray):
            if data_abs[i][0] > 50:
                n_com = i

                break


    for i in range(len(data_abs)-1, n_com ,-1):
        if isinstance(data_abs[i], numpy.int16):
            if data_abs[i] > 50:
                n_fin = i
                break
        elif isinstance(data_abs[i], numpy.ndarray):
            if data_abs[i][0] > 50:
                n_fin = i
                data_n = data_n[:][1]

                break

    data_n = data_n[n_com:n_fin]


    return data_n, len(data_n)

#Pre-enfasis se aplica un filtro de pre enfasis para acentuar las señales altas
def pre_enfasis(data_n):
    var_pre = 0.97

    señal_enf = numpy.append(data_n[1], data_n[1:] - var_pre * data_n[:-1])
    enfa_x = numpy.arange(0, señal_enf.size, 1)

    return señal_enf, enfa_x

def GFB(data_n, rate_n ):
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

    #Coeficientes cepstrales de frecuencia Mel (MFCC)
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  # *

    # Normalización media

    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

    return mfcc

#Se calcula media, mediana, min, max
def calc_parametros(data_n, emocion):
    vector = []
    vec_param = []
    prom = 0

    for i in range(len(data_n[1])):
        for j in range(len(data_n)):
            vector.append(data_n[j][i])
        #vector.sort()

        #Se calculan parametros con el vector ordenado por coeficiente
        # Promedio
        vec_param.append(round(statistics.mean(vector)*100)/100)
        # Desv standard
        vec_param.append(round(statistics.stdev(vector)*100)/100)
        # Mediana
        vec_param.append(round(statistics.median(vector)*100)/100)
        #Se agrega min,max
        vec_param.append(round(min(vector)*100)/100)
        vec_param.append(round(max(vector)*100)/100)
        vector.clear()

    vec_param.append(emocion)

    return vec_param

#Main___________________________________________________________________________________________________________________

#Abrimos la carpeta donde se encuentran los audios
carpeta_audios = os.listdir('C:/Users/rober/PycharmProjects/Programa_Tesis/venv/Audios')

#Leemos los archivo, retorna el rate, data, y tamaño
list_audios, list_emociones = leer_Audios(carpeta_audios)

#print(len(list_audios))

#Graficamos el audio
#graficar(list_audios[4][1])

#Impementamos la transformada rapida
#yf, xf = tr_Fourier(data, rate, n_data)

#Normalizamos el silencio
#data_mod = norm_silencio(data)

l_audios_n_sil = []
tam_men = len(list_audios[1][1])
indice = 0
inicio = time.time()
print(list_audios[1][1])
#Recorte del silencio del audio
for i in range(len(list_audios)):
    audio, tam_act= recorte_silencio_alter(list_audios[i][1])
    l_audios_n_sil.append(audio)
    #print(tam_act)
    if tam_act < tam_men:
        indice = i
        tam_men = tam_act

l_aud_ajust = []
#print(tam_men)
fin = time.time()
print(fin-inicio)
#Ajusta la longitud del audio
for i  in range(len(l_audios_n_sil)):
    data_aj =  ajuste_audio(l_audios_n_sil[i], tam_men)
    #print(len(data_aj))
    l_aud_ajust.append(data_aj)

print("----------")
#print(len(l_aud_ajust))

#graficar(data_mod2)

datas_enf = [0,0]
l_aud_enfa = []
#Se manda a pre enfatizar la señal
for i in range(len(l_aud_ajust)):
    datas_enf[0], datas_enf[1] = pre_enfasis(l_aud_ajust[i])
    #print(x)
    l_aud_enfa.append(datas_enf.copy())

#graficar(data_enf)

print("----------")
l_aud_mfcc = []
#Se crea el  ventaneo
for i in range(len(l_aud_enfa)):
    l_aud_mfcc.append(GFB(l_aud_enfa[i][0], list_audios[i][0]))

l_aud_par = []
lbl_emocion = ["neutral", "tranquilo", "feliz", "triste", "enojado", "temeroso", "asco", "soprendido"]
#Se calcula media, mediana, min, max, desv estandart
for i in range(len(l_aud_mfcc)):
    l_aud_par.append(calc_parametros(l_aud_mfcc[i],lbl_emocion[list_emociones[i]-1]))

#Encabezado
etiquetas = ["Promedio","DesviacionEstandar","Media","Minimo", "Maximo"]
encabezado = []
for i in range(12):
    for j in range(5):
        encabezado.append(etiquetas[j] + str(i+1))

#Guardar los datos en un csv
encabezado.append("Emocion")
archivo = pd.DataFrame(l_aud_par, index=None, columns = encabezado)
print(archivo)
archivo.to_csv('DB_Tesis_Index_lbl.csv',index=False)