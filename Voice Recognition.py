# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:28:38 2018

@author: ramse
"""

# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012

from __future__ import division
import numpy
import decimal
import math
import logging
from scipy.fftpack import dct
import scipy.io.wavfile
from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import wave
import sys
import pyaudio

def mfcc(signal,samplerate,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):
    """Compute MFCC features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:numpy.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    signal = preemphasis(signal,preemph)
    frames = framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = powspec(frames,nfft)
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    """Compute log Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,winlen=0.025,winstep=0.01,
        nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
        winfunc=lambda x:numpy.ones((x,))):
    """Compute Spectral Subband Centroid features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    highfreq= highfreq or samplerate/2
    signal = preemphasis(signal,preemph)
    frames = framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = powspec(frames,nfft)
    pspec = numpy.where(pspec == 0,numpy.finfo(float).eps,pspec) # if things are all zeros we get problems

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))

    return numpy.dot(pspec*R,fb.T) / feat

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.
    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

_size_data=0
_dim=0

def generate_codebook(data, size_codebook, epsilon=0.00001):

    global _size_data,_dim

    _size_data =len(data)
    assert _size_data>0

    _dim= len(data[0])
    assert _dim>0


    codebook = []
    codebook_abs_weights = [_size_data]
    codebook_rel_weights = [1.0]

    # calculate initial codevector: average vector of whole input data
    c0= avg_vec_of_vecs(data, _dim, _size_data)
    codebook.append(c0)

    # calculate the average distortion
    avg_dist = avg_distortion_c0(c0,data)

    #split codevectors until we have enough codevectors
    while len(codebook) < size_codebook:
        codebook, codebook_abs_weights, codebook_rel_weights, avg_dist = split_codebook(data, codebook, epsilon, avg_dist)

    return codebook, codebook_abs_weights, codebook_rel_weights


def matching(data,codebook):
        closest_c_list = [None] * len(data)#_size_data # list that contains the nearest codevector for each input data vector
        for i,vec in enumerate(data):   #for each input vector
            min_dist = None
            for i_c, c in enumerate(codebook):      #for each codevector
                d = euclid_squared(vec, c)
                if min_dist is None or d< min_dist:
                    min_dist = d
                    closest_c_list[i] = c
        vq_dist = avg_distortion_c_list(closest_c_list, data)

        return vq_dist
#def matching2(data,codebook):
#        closest_c_list = [] # list that contains the nearest codevector for each input data vector
#        for i,vec in enumerate(data):   #for each input vector
#            min_dist = None
#            for i_c, c in enumerate(codebook):      #for each codevector
#                d = euclid_squared(vec, c)
#                if min_dist is None or d< min_dist:
#                    min_dist = d
#                    min_temp = c
#            closest_c_list.append(min_temp)
#        vq_dist = avg_distortion_c_list(closest_c_list, data)
#
#        return vq_dist

def split_codebook(data,  codebook, epsilon, initial_avg_dist):

    #split codevectors
    new_codevectors = []
    for c in codebook:
        # the new codevectors c1 and c2 will moved by epsilon and -epsilon
        #so to be apart from each other
        c1 = new_codevector(c, epsilon)
        c2 = new_codevector(c, -epsilon)
        new_codevectors.extend((c1,c2))

    codebook = new_codevectors
    len_codebook = len(codebook)
    abs_weights = [0]* len_codebook
    rel_weights = [0.0] * len_codebook

    print('> splitting to size', len_codebook)

    # try to reach a convergence by minimizing the average distortion. this is
    # done by moving the codevectors step by step to the center of the points
    # in their proximity
    avg_dist = 0
    err = epsilon + 1
    num_iter = 0
    while err > epsilon:
        # find closest codevectors for each vector in data (find the proximity of each codevector)
        closest_c_list = [None] * _size_data # list that contains the nearest codevector for each input data vector
        vecs_near_c = defaultdict(list) # list with codevector index -> input data vector mapping
        vecs_idxs_near_c = defaultdict(list) # list with codevector index -> input data index mapping

        for i,vec in enumerate(data):   #for each input vector
            min_dist = None
            closest_c_index =None
            for i_c, c in enumerate(codebook):      #for each codevector
                d = euclid_squared(vec, c)
                if min_dist is None or d< min_dist:
                    min_dist = d
                    closest_c_list[i] = c
                    closest_c_index = i_c
            vecs_near_c[closest_c_index].append(vec)
            vecs_idxs_near_c[closest_c_index].append(i)

        #update codebook: recalculate each codevector so that it sits in the center of the points in their proximity
        for i_c in range(len_codebook): #for each codevector index
            vecs = vecs_near_c.get(i_c) or [] #get its proximity input vectors
            num_vecs_near_c = len(vecs)
            if num_vecs_near_c > 0:
                new_c = avg_vec_of_vecs(vecs, _dim) #calculate the new center
                codebook[i_c] = new_c              #update in codebook
                for i in vecs_idxs_near_c[i_c]:      #update in input vector index -> codevector mapping list
                    closest_c_list[i] = new_c

                # update the weights
                abs_weights[i_c] = num_vecs_near_c
                rel_weights[i_c] = num_vecs_near_c / _size_data

            #recalculate average distortion value
            prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
            avg_dist = avg_distortion_c_list(closest_c_list, data)

            #recalculate the new error value
            err = (prev_avg_dist - avg_dist) / prev_avg_dist
            # print(closest_c_list)
            # print('> iteration', num_iter, 'avg_dist', avg_dist, 'prev_avg_dist', 'err' , err)

            num_iter +=1

        return codebook, abs_weights, rel_weights, avg_dist

    
def avg_vec_of_vecs(vecs, dim=None, size=None):

    size = size or len(vecs)
    dim = dim or len(vecs[0])
    avg_vec =[0.0] * dim
    for vec in vecs:
        for i, x in enumerate(vec):
            avg_vec[i] += x/size


    return avg_vec


def new_codevector(c,e):
    return [x* (1.0 + e) for x in c]

def avg_distortion_c0(c0, data, size=None):
    size =size or _size_data
    return reduce(lambda s,d: s+d / size,
                  (euclid_squared(c0,vec)
                   for vec in data),
                  0.0)


def avg_distortion_c_list(c_list, data, size=None):
    size=size or _size_data
    return reduce(lambda s, d: s+d/size,
                  (euclid_squared(c_i, data[i])
                   for i, c_i in enumerate(c_list)),
                  0.0)


def euclid_squared(a, b):
    return sum((x_a - x_b)**2 for x_a, x_b in zip(a,b))





def read_file(str):
    sample_rate, signal1 = scipy.io.wavfile.read(str)  # File assumed to be in the same directory
    print("RERSRSFFDGDGDHHFHH",sample_rate)
#    signal1 = signal1[0:int(2 * sample_rate)]  # Keep the first 2 seconds
    signal1 = signal1[0:int(2 * sample_rate)]  # Keep the first 2 seconds
    feat1=mfcc(signal1,sample_rate,0.025,0.01,13, 26,1024,0,None,0.97,22,True)
    return feat1
feat1=read_file('1.wav')
print(feat1.shape)
feat2=read_file('2.wav')
feat3=read_file('3.wav')
feat4=read_file('03b03Tc.wav')

#
#
codebook1, codebook_abs_weights, codebook_rel_weights = generate_codebook(feat2, size_codebook=10, epsilon=0.00001)

#Writing data to pickle file
def write_file():
    Input1=input("Enter speaker name")
    Ip=input("Enter file name where your voice is stored") 
    Input2=read_file(Ip)
    codes, codebook_abs_weights, codebook_rel_weights = generate_codebook(Input2, size_codebook=10, epsilon=0.00001)
    dictionary=defaultdict()
    
    try:
        with open("Voicerecognition.pkl","rb") as pkl:
            print("tryZEROTHITERATION")
            G=pickle.load(pkl)
            print("TRYYYYYYYYYYYYY")
            print(G,"\n\n\n")
            print(type(G))
            G[Input1]=codes
            print("FINAL TRYYYYYYYYYYYYYYY")
            return G
    except:
        with open("Voicerecognition.pkl","wb") as pkl:
            dictionary={}
            print("EXCEPTTTTTTT")
            dictionary[Input1]=codes
            return dictionary
#            pickle.dump(dictionary,pkl)
def read_from_file():    
    Input1=input("Enter speaker name")    
    with open("Voicerecognition.pkl","rb") as pkl2:
        global G
        G=pickle.load(pkl2)
        try:
            G=G[Input1]
            print(G)
        except:
            raise ValueError("This voice is not available in our database")
        return G
def file_call():
    try:
        G1=write_file()
        with open("Voicerecognition.pkl","wb") as pkl3:
            pickle.dump(G1,pkl3)
        print("DONEEEE")
    except:
        print("EXCEPTT")
        write_file()
def process_new_input():
    Ip1=input("Enter speaker name")
    Ip0=input("Enter file name where your voice is stored")
    Ip2=read_file(Ip0)
    
    return Ip1,Ip2
def matching_check_file(InputMFCC):
    with open("Voicerecognition.pkl","rb") as pkl4:
        data=pickle.load(pkl4)
        mini=None
        name=None
        for i in data:
            print(InputMFCC.shape)
            print(i,matching(InputMFCC,data[i]),"FINALMATCHES")
            value=matching(InputMFCC,data[i])
            if mini==None or value<mini:
                mini=value
                name=i
    return mini,name
def print_file():
    with open("Voicerecognition.pkl","rb") as pkl5:
        data=pickle.load(pkl5)
        print(data)
def del_from_file(speaker_name):
    with open("Voicerecognition.pkl","rb") as pkl6:
        data=pickle.load(pkl6)
        del data[speaker_name]
        return data

def microphone_integration(file_name):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 48000
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = file_name+ ".wav"
     
    audio = pyaudio.PyAudio()
     
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
     
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
 
 
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
     
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def write_file_for_microphone():
    Input1=input("Enter speaker name")
    microphone_integration(Input1)
    Ip=Input1+".wav"
    Input2=read_file(Ip)
    codes, codebook_abs_weights, codebook_rel_weights = generate_codebook(Input2, size_codebook=10, epsilon=0.00001)
    dictionary=defaultdict()
    
    try:
        with open("Voicerecognition.pkl","rb") as pkl:
            print("tryZEROTHITERATION")
            G=pickle.load(pkl)
            print("TRYYYYYYYYYYYYY")
            print(G,"\n\n\n")
            print(type(G))
            G[Input1]=codes
            print("FINAL TRYYYYYYYYYYYYYYY")
            return G
    except:
        with open("Voicerecognition.pkl","wb") as pkl:
            dictionary={}
            print("EXCEPTTTTTTT")
            dictionary[Input1]=codes
            return dictionary
#            pickle.dump(dictionary,pkl)
    
def file_call_microphone():
    try:
        G1=write_file_for_microphone()
        with open("Voicerecognition.pkl","wb") as pkl3:
            pickle.dump(G1,pkl3)
        print("DONEEEE")
    except:
        print("EXCEPTT")
        write_file_for_microphone()

def process_new_input_microphone():
    Ip1=input("Enter speaker name")
    microphone_integration(Ip1)
    Ip0=Ip1+".wav"
    Ip2=read_file(Ip0)
    
    return Ip1,Ip2

Choice=input("WHAT DO YOU WANT TO DO?\n\nEnter 1 to insert voices in our file\nEnter 2 to read Voices from our file\nEnter 3 to Check matching with other voices by inputting a new voice\n Enter 4 to print file\n Enter 5 to delete a voice from Database\n Enter 6 to check shape of file\n Enter 7 to Store a new voice to database live(Through Microphone)\n Enter 8 to check best matching voice by inputting a live new voice(Through Microphone)")
if Choice=='1':      
    file_call()
elif Choice=='2':    
    codebook1=read_from_file() 
elif Choice=='3':
    Inputname,InputMFCC=process_new_input()
    mini,name=matching_check_file(InputMFCC)
    print("\n\n","The Closest Match to",Inputname,"is:")
    print(name,mini)
elif Choice=='4':
    print_file()
elif Choice=='5':
    spkr_name=input("Enter the voice to delete")
    data=del_from_file(spkr_name)
    with open("Voicerecognition.pkl","wb") as pkl7:
        pickle.dump(data,pkl7)
    
elif Choice=='6':
    ip=input("ENTER FILE NAME")
    print(read_file(ip).shape)   
    
elif Choice=='7':
    file_call_microphone()

elif Choice=='8':
    Inputname,InputMFCC=process_new_input_microphone()
    mini,name=matching_check_file(InputMFCC)
    print("\n\n","The Closest Match to",Inputname,"is:")
    print(name,mini)
    
    



#print(matching(feat2,codebook1),"MATCHING1")
#print(matching(feat3,codebook1),"MATCHING2")
#print(matching(feat4,codebook1),"MATCHING3")
spf = wave.open('1.wav','r')#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = numpy.fromstring(signal, 'Int16')
#If Stereo
if spf.getnchannels() == 2:   #There are 2 channels for audio 
    print('Just mono files')    
    sys.exit(0)
plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()
    
fig=plt.figure()
ax=fig.add_subplot(221)
cb=numpy.asarray(codebook1)
print(cb.shape)
#199 Blue points and 32/16/8 Red points
ax.scatter(feat1[:,2],feat1[:,10],color='blue',marker='.')#All points
ay=fig.add_subplot(222)#Taking only 2 dimensions at a time. [:,5]- Takes all values of 5th dimension
ay.scatter(cb[:,2],cb[:,10],color='red',marker='.')#Rate points 
plt.show()
