import os
import struct
import zlib

import numpy

class FeatureFile():
    def __init__(self) -> None:
        self.FrameSizeInSamples = 0
        self.HopSizeInSamples = 0
        self.fs = 0
        self.SystemTime = None
        self.calibrationInDb = [0, 0]
        self.AndroidID = ''
        self.BluetoothTransmitterMAC = ''
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

def load(file):
    featureFile = FeatureFile()
    if type(file) is str and os.path.isfile(file):
        with open(file, mode='rb') as filereader:
            data = filereader.read()
    elif type(file) == bytearray:
        data = file
    elif type(file) is str:
        data = bytearray(file.encode())
    if len(data):
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
        header += int(4).to_bytes(4, "big")
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