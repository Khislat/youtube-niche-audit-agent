"""Modulyatsiya spektri: envelope tebranishlari (1-60 Hz) cho'qqilari. x = cho'qqi / mahalliy fon. x>=3 sezilarli.
Usage: python modspec.py a.wav b.webm ..."""
import sys, numpy as np, soundfile as sf, librosa
from scipy.signal import find_peaks
def run(p):
    y, sr = sf.read(p, always_2d=True); y = y.mean(axis=1)
    hop = 256; n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    ff = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fs_env = sr/hop
    res = {}
    for name,(lo,hi) in {'100-400':(100,400),'400-1500':(400,1500),'1500-5000':(1500,5000)}.items():
        env = S[(ff>=lo)&(ff<hi)].sum(axis=0)
        env = np.log(env+1e-9); env = env - env.mean()
        # welch-ish: average FFT over 20 s windows
        w = int(20*fs_env); spec = []
        for s in range(0, len(env)-w+1, w//2):
            seg = env[s:s+w]*np.hanning(w); spec.append(np.abs(np.fft.rfft(seg))**2)
        spec = np.mean(spec, axis=0); fr = np.fft.rfftfreq(w, 1/fs_env)
        m = (fr>=1)&(fr<=60); fr, spec = fr[m], spec[m]
        # prominence vs local median (smoothed baseline)
        base = np.convolve(spec, np.ones(25)/25, mode='same')
        ratio = spec/base
        pk,_ = find_peaks(ratio, height=3)
        top = sorted(pk, key=lambda i: -ratio[i])[:4]
        res[name] = [(round(float(fr[i]),2), round(float(ratio[i]),1)) for i in top]
    print('==', p); [print('  ', k, 'Hz:', ', '.join(f'{f} Hz (x{r})' for f,r in v) or 'cho\'qqi yo\'q') for k,v in res.items()]
for p in sys.argv[1:]: run(p)
