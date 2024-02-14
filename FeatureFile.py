"""
functions to load and save olMEGA feature files
License: BSD 3-clause 
Version 1.0.0 Sven Franz @ Jade HS
Version 1.0.1 JB bug fixed with non valid filenames
"""
import os
import math
import os
import struct
import zlib

import numpy
import scipy

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

class FeatureFile():
    def __init__(self) -> None:
        self.FrameSizeInSamples = 0
        self.HopSizeInSamples = 0
        self.fs = 0
        self.SystemTime = None
        self.calibrationInDb = [0, 0]
        self.AndroidID = ''
        self.BluetoothTransmitterMAC = ''
        self.TransmitterSamplingrate = 0
        self.AppVersion = ''
        self.nBytesHeader = 0
        self.ProtokollVersion = 0
        self.nDimensions = 0
        self.nBlocks = 0
        self.StartTime = None
        self.nFramesPerBlock = 0
        self.nFrames = 0
        self.vFrames = []
        self.BlockSizeInSamples = 0
        self.mBlockTime = None
        self.dataTimestamps = []
        self.data = []

    def wavSignal(self):
        if type(self.data) is numpy.ndarray:
            n = [int(self.data.shape[1] / 2), int(self.data.shape[1] / 4)]
            Pxy = self.data[:, 0 : n[0] : 2]+ self.data[:, 1 : n[0] : 2] * 1j
            Pxx = self.data[:, n[0] : n[0] + n[1]].clip(0, None)
            Pyy = self.data[:, n[0] + n[1] : ].clip(0, None)
            blocklen    = math.floor(.025 * self.fs)
            nFFT        = next_power_of_2(blocklen)
            spec = numpy.sqrt(Pxx)
            spec = scipy.signal.resample(spec, int(spec.shape[0] * (0.125 * 1000 * 2) / (0.025 * 1000))).clip(0, None)
            phase = 2 * math.pi * numpy.random.rand(spec.shape[0], spec.shape[1])
            phase[:, 0] = 0
            spec = numpy.multiply(spec, numpy.exp(1j * phase))
            spec = numpy.concatenate((spec, numpy.fliplr(spec[:, 1 : -1]).conj()), 1)
            blocks = numpy.real(numpy.fft.ifft(spec * nFFT, nFFT))
            TimeSignal = numpy.zeros(int(blocks.shape[0] * .025 / 2 * self.fs))
            if TimeSignal.shape[0] % blocklen != 0:
                TimeSignal = numpy.concatenate((TimeSignal, numpy.zeros(int(blocklen - (TimeSignal.shape[0] % blocklen)))))
            TimeSignal = numpy.concatenate((TimeSignal, numpy.zeros(int(blocklen / 2))))
            idx = 0
            win = numpy.hanning(int(blocklen))
            for blockIdx in range(blocks.shape[0]):
                TimeSignal[idx : (idx + blocklen)] += numpy.multiply(blocks[blockIdx, 0 : blocklen], win)
                idx += int(blocklen / 2)
            TimeSignal = TimeSignal[0 : int(blocks.shape[0] * .025 / 2 * self.fs)]
            return TimeSignal

def load(file):
    data = None
    featureFile = FeatureFile()
    if type(file) is str:
        if file[-5:] == '.feat':
            if os.path.isfile(file):
                with open(file, mode='rb') as filereader:
                    data = filereader.read()
            else:
                return None
        else:
            data = bytearray(file.encode())
    elif type(file) == bytearray:
        data = file

    if not data is None and len(data):
        veryOldHeaderSizes = [29, 36]
        if len(data) >= 2 and hex(data[0]) == '0x78' and hex(data[1]) in ['0x01', '0x9c', '0xda']:
            data = zlib.decompress(data)
        featureFile.vFrames = [int.from_bytes(data[0:4], byteorder='big', signed=True)]
        featureFile.nDimensions = int.from_bytes(data[4:8], byteorder='big', signed=True)
        if len(data) - featureFile.vFrames[0] * featureFile.nDimensions * 4 in veryOldHeaderSizes:
            featureFile.ProtokollVersion = veryOldHeaderSizes.index(len(data) - featureFile.vFrames[0] * featureFile.nDimensions * 4)
            featureFile.nBytesHeader = veryOldHeaderSizes[featureFile.ProtokollVersion]
            featureFile.FrameSizeInSamples = int.from_bytes(data[8:12], byteorder='big', signed=True)
            featureFile.HopSizeInSamples = int.from_bytes(data[12:16], byteorder='big', signed=True)
            featureFile.fs = int.from_bytes(data[16:20], byteorder='big', signed=True)
            featureFile.mBlockTime = "".join(map(chr, data[20:featureFile.nBytesHeader]))
            featureFile.SystemTime = featureFile.mBlockTime
        else:
            featureFile.ProtokollVersion = int.from_bytes(data[0:4], byteorder='big', signed=True)
            featureFile.vFrames = [int.from_bytes(data[4:8], byteorder='big', signed=True)]
            featureFile.nDimensions = int.from_bytes(data[8:12], byteorder='big', signed=True)
            featureFile.FrameSizeInSamples = int.from_bytes(data[12:16], byteorder='big', signed=True)
            featureFile.HopSizeInSamples = int.from_bytes(data[16:20], byteorder='big', signed=True)
            featureFile.fs = int.from_bytes(data[20:24], byteorder='big', signed=True)
            featureFile.mBlockTime = "".join(map(chr, data[24:40]))
            featureFile.SystemTime = featureFile.mBlockTime
            if featureFile.ProtokollVersion >= 2:
                featureFile.calibrationInDb = [struct.unpack('f', data[56:60])[0], struct.unpack('f', data[60:64])[0]]
                featureFile.nBytesHeader = 64
            if featureFile.ProtokollVersion >= 3:
                featureFile.SystemTime = "".join(map(chr, data[40:56]))
                featureFile.nBytesHeader = 56
            if featureFile.ProtokollVersion >= 4:
                featureFile.AndroidID = "".join(map(chr, data[64:80]))
                featureFile.BluetoothTransmitterMAC = "".join(map(chr, data[80:97]))
                featureFile.nBytesHeader = 97
            if featureFile.ProtokollVersion >= 5:
                featureFile.TransmitterSamplingrate = int.from_bytes(data[97:101], byteorder='big', signed=True)
                featureFile.nBytesHeader = 101
            if featureFile.ProtokollVersion >= 6:
                featureFile.AppVersion = "".join(map(chr, data[101:121]))
                featureFile.nBytesHeader = 121
        featureFile.nBlocks = 1
        featureFile.nFrames = sum(featureFile.vFrames)
        featureFile.nFramesPerBlock = featureFile.vFrames[0]
        featureFile.BlockSizeInSamples = featureFile.nFrames * featureFile.HopSizeInSamples + featureFile.FrameSizeInSamples
        featureFile.StartTime = featureFile.mBlockTime
        if len(data) - featureFile.vFrames[0] * featureFile.nDimensions * 4 == featureFile.nBytesHeader:
            data = numpy.frombuffer(data, dtype='>f4', offset = featureFile.nBytesHeader, count=-1).reshape([featureFile.nFrames, featureFile.nDimensions])
            featureFile.dataTimestamps = data[ :: , 0 : 2]
            featureFile.data = data[ :: , 2 :: ]
        else:
            return None
    return featureFile

def save(featureFile, filename, compressOnServer = False):
    if type(featureFile) is FeatureFile:
        featureFile.nBlocks = 1
        featureFile.BlockSizeInSamples = featureFile.nFrames * featureFile.HopSizeInSamples + featureFile.FrameSizeInSamples
        header = bytearray()
        header += int(6).to_bytes(4, "big")
        header += featureFile.nFrames.to_bytes(4, "big")
        header += (featureFile.nDimensions + 2).to_bytes(4, "big")
        header += featureFile.FrameSizeInSamples.to_bytes(4, "big")
        header += featureFile.HopSizeInSamples.to_bytes(4, "big")
        header += featureFile.fs.to_bytes(4, "big")
        header += bytearray((featureFile.mBlockTime + " "*(16-len(featureFile.mBlockTime)))[:16], "utf-8")
        header += bytearray((featureFile.SystemTime + " "*(16-len(featureFile.SystemTime)))[:16], "utf-8")
        header += bytearray(struct.pack("f", featureFile.calibrationInDb[0]))
        header += bytearray(struct.pack("f", featureFile.calibrationInDb[1]))
        header += bytearray((featureFile.AndroidID + " "*(16-len(featureFile.AndroidID)))[:16], "utf-8")
        header += bytearray((featureFile.BluetoothTransmitterMAC + " "*(17-len(featureFile.BluetoothTransmitterMAC)))[:17], "utf-8")

        header += featureFile.TransmitterSamplingrate.to_bytes(4, "big")
        header += bytearray((featureFile.AppVersion + " "*(20-len(featureFile.AppVersion)))[:20], "utf-8")

        featureFile.dataTimestamps = numpy.linspace(0, featureFile.nFrames * featureFile.HopSizeInSamples / featureFile.fs - featureFile.HopSizeInSamples / featureFile.fs, featureFile.data.shape[0])
        featureFile.dataTimestamps = numpy.concatenate((featureFile.dataTimestamps, featureFile.dataTimestamps + featureFile.FrameSizeInSamples / featureFile.fs), axis=0)
        featureFile.dataTimestamps = featureFile.dataTimestamps.reshape(featureFile.data.shape[0], 2)
        content = bytearray()
        for val in numpy.concatenate((featureFile.dataTimestamps, featureFile.data), axis = 1).flatten():
            content += bytearray(struct.pack(">f", val))
        data = header + content
        if compressOnServer:
            data = zlib.compress(data)
        with open(filename, mode='wb') as filewriter:
            filewriter.write(data)

if __name__ == "__main__":
    feat = load('AE210040_AkuData/PSD_20221024_075625249.feat')

    dates = []
    for file in os.listdir('AE210040_AkuData'):            
        if file.startswith("PSD_"):
            feat = load(os.path.join("AE210040_AkuData", file))
            if feat:
                dates.append(feat.SystemTime)
        
    pass