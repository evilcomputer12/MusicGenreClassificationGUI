import shutil
from tkinter import *
from tkinter.ttk import *
import os
from tkinter import filedialog

import librosa
import numpy as np
import torch
import sys
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import librosa
from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db
from librosa import amplitude_to_db

from model import genreNet
from config import MODELPATH
from config import GENRES

import warnings
import threading

import librosa.display

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time


warnings.filterwarnings("ignore")

fname = None
default_path = "tempdata"

master = Tk()
v = StringVar()
Label(master, textvariable=v).pack()
v.set("Otvori muzicka datoteka i otkri go nejziniot muzicki zanr")
v2 = StringVar()
Label(master, textvariable=v2).pack()


def main():
    b1.configure(state='disabled')
    b.configure(state='disabled')
    progressbar = Progressbar(master, length=200, mode='indeterminate')
    progressbar.pack()
    progressbar.start()
    le = LabelEncoder().fit(GENRES)
    # ------------------------------- #
    ## LOAD TRAINED GENRENET MODEL
    net = genreNet()
    net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))
    # ------------------------------- #
    ## LOAD AUDIO
    audio_path = fname
    y, sr = load(audio_path, mono=True, sr=22050)
    ## AUDIO SPECTOGRAM
    fspec1 = melspectrogram(y, sr=sr, n_mels=128)
    #plt.subplot(3, 1, 1)
    spec1_amp = amplitude_to_db(fspec1, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec1_amp, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Power Spectogram')
    plt.tight_layout()
    if os.path.exists('../spectograms/mel_pow_spectogram.png'):
        plt.savefig('../spectograms/mel_pow_spectogram_{}.png'.format(int(time.time())), dpi=300)
    else:
        plt.savefig('../spectograms/mel_pow_spectogram.png', dpi=300)
    #plt.subplot(3, 1, 2)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(fspec1 ** 2, sr=sr, y_axis='log')
    plt.colorbar()
    plt.title('Power Spectogram')
    plt.tight_layout()
    if os.path.exists('../spectograms/pow_spectogram.png'):
        plt.savefig('../spectograms/pow_spectogram_{}.png'.format(int(time.time())), dpi=300)
    else:
        plt.savefig('../spectograms/pow_spectogram.png', dpi=300)
    #plt.subplot(3, 1, 3)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(power_to_db(fspec1 ** 2, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Power Spectrogram')
    plt.tight_layout()
    if os.path.exists('../spectograms/log_pow_spectogram.png'):
        plt.savefig('../spectograms/log_pow_spectogram_{}.png'.format(int(time.time())), dpi=300)
    else:
        plt.savefig('../spectograms/log_pow_spectogram.png', dpi=300)
    # ------------------------------- #
    ## GET CHUNKS OF AUDIO SPECTROGRAMS
    S = melspectrogram(y, sr).T
    S = S[:-1 * (S.shape[0] % 128)]
    num_chunk = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)
    # ------------------------------- #
    ## CLASSIFY SPECTROGRAMS
    genres = list()
    for i, data in enumerate(data_chunks):
        data = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds = net(data)
        pred_val, pred_index = preds.max(1)
        pred_index = pred_index.data.numpy()
        pred_val = np.exp(pred_val.data.numpy()[0])
        pred_genre = le.inverse_transform(pred_index).item()
        if pred_val >= 0.5:
            genres.append(pred_genre)
    # ------------------------------- #
    s = float(sum([v for k, v in dict(Counter(genres)).items()]))
    pos_genre = sorted([(k, v / s * 100) for k, v in dict(Counter(genres)).items()], key=lambda x: x[1], reverse=True)
    for genre, pos in pos_genre:
        print("%10s: \t%.2f\t%%" % (genre, pos))
        text.configure(state="normal")
        text.insert(END, "%10s: \t%.2f\t%%\n" % (genre, pos))
        progressbar.destroy()
        text.configure(state="disable")
        b1.configure(state='enabled')
        b.configure(state='enabled')
    return

def loadfile():
    global fname
    name = filedialog.askopenfilename(
        title="Otvori muzicka datoteka",
        filetypes=(
        ("Music files", "*.mp3"), ("All files", "*.*")))
    if len(name) != 0:
        fname = name
        v2.set(os.path.basename(fname))
        text.configure(state="normal")
        text.delete("1.0", END)
        b.configure(state='enabled')
    else:
        v2.set("")
        b.configure(state='disabled')

def thread():
    t1 = threading.Thread(target=main, args=())
    t1.start()
    return

b1 = Button(master, text="Otvori datoteka", width=20, command=loadfile)
b1.pack()


b = Button(master, text="Pronajdi zanr", width=20, command=thread)
b.pack()

text = Text(master, width=20, height=15)
text.pack()

master.title("Detekcija na zanr")
master.geometry("400x400")
master.maxsize(400, 400)
master.mainloop()

if __name__ == '__main__':
    sys.exit()
    main()