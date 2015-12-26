import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import scipy
from utilities import *
from pydub import AudioSegment
from scikits.audiolab import *
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

""" plot spectrogram"""
def plotstft(samples, samplerate, infile, outfolder, binsize=2**10, plotpath=None, colormap="Greys"):
    s = stft(samples, 256)
    
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    #return ims
    timebins, freqbins = np.shape(ims)
    #scipy.misc.toimage(imnp.transpose(ims)s, cmin=0.0, cmax=...).save('infile.jpg')
    #scipy.misc.imsave('infile.jpg', np.transpose(ims))
    
    
    slash = infile.rfind('/')
    outpath = outfolder + infile[slash + 1:-4] + ".png"
    create_path(outpath)
    #plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.imsave(infile[:slash] + "/imgs/" + infile[slash + 1:-4], np.transpose(ims), cmap="Greys")
    #plt.close()
    #return ims
    #plt.xlabel("time (s)")
    #plt.ylabel("frequency (hz)")

    #plt.xlim([0, timebins-1])
    #plt.ylim([0, freqbins])
    fig = plt.figure(figsize=(10, 5 ))
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.close()
    #plt.colorbar()

    #plt.xlabel("time (s)")
    #plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins/2.5])
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    #print infile
    #plt.show()

    plt.savefig(outpath, frameon=False, bbox_inches=extent, pad_inches=0, format="png")
    plt.close(fig)
    
    plt.clf()
#song = AudioSegment.from_mp3("/Users/quinnjarrell/datasets/Music/segmented/Users/quinnjarrell/datasets/Music/alternative/50_Minutes-Colours1.mp3")
#sound_file = Sndfile("/Users/quinnjarrell/datasets/Music/segmented/Users/quinnjarrell/datasets/Music/alternative/50_Minutes-Colours0.wav", 'r')
#signal = sound_file.read_frames(sound_file.nframes)
#plotstft(signal[:,], 22050, "outs3.jpg")



