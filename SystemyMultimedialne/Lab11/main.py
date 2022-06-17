from base64 import decode
import numpy as np
import glob
from parse import parse
import operator as op
from functools import reduce
from pip import main
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

def encode_rle(_data):

    data_flatten = _data.copy()
    shape = data_flatten.shape
    dimension = len(shape)

    data_flatten = data_flatten.flatten()
    
    # structure = [dimension, shape[0], shape[1] + ... + shape[x] + (repeats + bit) + (repeats + bit) + ...]
    # worst case = 2 times more bits than in original data + dimensions + shape[0] + shape[1] + ... + shape[x]
    encoded_data = np.zeros(data_flatten.shape[0]*2 + dimension + 1).astype(_data.dtype)
    encoded_data[0] = dimension

    for counter, sh in enumerate(shape):
        encoded_data[counter + 1] = sh

    # pair = bit and how many times bit occured in row [1, 1, 4, 5, 5, 5] = [2, 1, 1, 4, 3, 5]
    pair_index = dimension + 1
    bit_index = 0
    with tqdm(total=data_flatten.shape[0]) as pbar:
        while bit_index < data_flatten.shape[0]:
            current_bit = data_flatten[bit_index]
            repeats = 1
            while bit_index + repeats < data_flatten.shape[0] and current_bit == data_flatten[bit_index + repeats]:
                repeats += 1

            bit_index += repeats
            pbar.update(bit_index)
            encoded_data[pair_index] = repeats
            encoded_data[pair_index + 1] = current_bit
            pair_index += 2

    return encoded_data[:pair_index]
    

def decore_rle(_data):
    
    data = _data.copy()

    dimension = data[0]
    shape = np.zeros(dimension).astype(np.int)

    for i in range(dimension):
        shape[i] = data[i + 1]

    size = 1
    for s in shape:
        size *= s

    decoded_data = np.zeros(size).astype(_data.dtype)

    decoded_index = 0
    for i in range(dimension + 1, data.shape[0], 2):
        repeats = data[i]
        bit = data[i+1]
        for j in range(repeats):
            decoded_data[decoded_index + j] = bit
        decoded_index += repeats

    decoded_data = np.reshape(decoded_data, shape).astype(data.dtype)

    return decoded_data



class CorrectionCodes:
    ########################################################################
    def data_to_bits(data):
        return np.unpackbits(np.frombuffer(data.tobytes(), dtype=np.uint8))

    def bits_to_data(bits,dtype=np.uint8):
        return np.frombuffer(np.packbits(bits.astype(np.uint8)).tobytes(), dtype=dtype)
    
    def parity(data):
        if data.shape[0]==1:
            return data
        return reduce(op.xor,data)

    ########################################################################
    def Write(data,filePrefix='',bit_count=4): 
        bits=CorrectionCodes.data_to_bits(data)
        m0=np.array([])
        m1=np.array([])
        m2=np.array([])    
        m3=np.array([])
        m4=np.array([])
        mH=np.array([])
        codes=np.array([bit_count,0])
        for i in tqdm(range(0,bits.size,bit_count),desc="Coding for {} \t bits: ".format(bit_count)):
            sample=bits[i:i+bit_count]         
            sample=np.append(sample,np.zeros(((bit_count-(sample.shape[0]%bit_count))%bit_count,),dtype=np.uint8))
            m0=np.append(m0,CorrectionCodes.coding_none(sample))
            m1=np.append(m1,CorrectionCodes.coding_M1(sample))
            m2=np.append(m2,CorrectionCodes.coding_M2(sample))
            if bit_count>1:
                m3=np.append(m3,CorrectionCodes.coding_M3(sample))
                m4=np.append(m4,CorrectionCodes.coding_M4(sample))
            mHt,codes=CorrectionCodes.coding_Hamming(sample,codes)
            mH=np.append(mH,mHt)
        with open(filePrefix+"M0_"+str(bit_count)+".hex", 'wb') as f:
            m0_t=CorrectionCodes.bits_to_data(m0)
            m0_t.tofile(f)
        with open(filePrefix+"M1_"+str(bit_count)+".hex", 'wb') as f:
            m1_t=CorrectionCodes.bits_to_data(m1)
            m1_t.tofile(f)
        with open(filePrefix+"M2_"+str(bit_count)+".hex", 'wb') as f:
            m2_t=CorrectionCodes.bits_to_data(m2)
            m2_t.tofile(f)
        with open(filePrefix+"M3_"+str(bit_count)+".hex", 'wb') as f:
            m3_t=CorrectionCodes.bits_to_data(m3)
            m3_t.tofile(f)
        with open(filePrefix+"M4_"+str(bit_count)+".hex", 'wb') as f:
            m4_t=CorrectionCodes.bits_to_data(m4)
            m4_t.tofile(f)
        with open(filePrefix+"MH_"+str(codes[0])+"."+str(codes[1])+".hex", 'wb') as f:
            mH_t=CorrectionCodes.bits_to_data(mH)
            mH_t.tofile(f)
    
    def Read(filePrefix=''):
        fl=glob.glob(filePrefix+"*.hex")
        decoded_data={}
        for file in fl:
            pr=parse('M{}_{}.hex',file[len(filePrefix):])
            decoded_data[file[len(filePrefix):-4]]=None
            if pr[0]=='0':               
                with open(file, 'rb') as f:
                    m=CorrectionCodes.data_to_bits(np.fromfile(f,dtype=np.uint8))
                    dm=np.array([],dtype=np.uint8)
                    is_correct=True
                    bit_count=int(pr[1])
                    for i in tqdm( range(0,m.shape[0], bit_count),desc="Decoding file {}: \t\t".format(file)):
                        temp=CorrectionCodes.decoding_none(m[i:i+bit_count])
                        if temp is not None:
                            dm=np.append(dm,temp)
                        else:
                            is_correct=False
                            break
                    if is_correct:
                      decoded_data[file[len(filePrefix):-4]]=CorrectionCodes.bits_to_data(dm)   
            elif pr[0]=='1':
                with open(file, 'rb') as f:
                    m=CorrectionCodes.data_to_bits(np.fromfile(f,dtype=np.uint8))
                    dm=np.array([],dtype=np.uint8)
                    is_correct=True
                    bit_count=int(pr[1])
                    for i in tqdm(range(0,m.shape[0], bit_count*3),desc="Decoding file {}: \t\t".format(file)):
                        temp=CorrectionCodes.decoding_M1(m[i:i+bit_count*3])
                        if temp is not None:
                            dm=np.append(dm,temp)
                        else:
                            is_correct=False
                            break
                    if is_correct:
                      decoded_data[file[len(filePrefix):-4]]=CorrectionCodes.bits_to_data(dm)  
            elif pr[0]=='2':
                with open(file, 'rb') as f:
                    m=CorrectionCodes.data_to_bits(np.fromfile(f,dtype=np.uint8))
                    dm=np.array([],dtype=np.uint8)
                    is_correct=True
                    bit_count=int(pr[1])
                    for i in tqdm(range(0,m.shape[0]-3, bit_count+2),desc="Decoding file {}: \t\t".format(file)):
                        temp=CorrectionCodes.decoding_M2(m[i:i+bit_count+2])
                        if temp is not None:
                            dm=np.append(dm,temp)
                        else:
                            is_correct=False
                            break
                    if is_correct:
                      decoded_data[file[len(filePrefix):-4]]=CorrectionCodes.bits_to_data(dm)  
            elif pr[0]=='3':
                with open(file, 'rb') as f:
                    m=CorrectionCodes.data_to_bits(np.fromfile(f,dtype=np.uint8))
                    dm=np.array([],dtype=np.uint8)
                    is_correct=True
                    bit_count=int(pr[1])
                    for i in tqdm(range(0,m.shape[0]-3, bit_count+2),desc="Decoding file {}: \t\t".format(file)):
                        temp=CorrectionCodes.decoding_M3(m[i:i+bit_count+2])
                        if temp is not None:
                            dm=np.append(dm,temp)
                        else:
                            is_correct=False
                            break
                    if is_correct:
                      decoded_data[file[len(filePrefix):-4]]=CorrectionCodes.bits_to_data(dm)  
            elif pr[0]=='4':
                with open(file, 'rb') as f:
                    m=CorrectionCodes.data_to_bits(np.fromfile(f,dtype=np.uint8))
                    dm=np.array([],dtype=np.uint8)
                    is_correct=True
                    bit_count=int(pr[1])
                    for i in tqdm(range(0,m.shape[0]-5, (bit_count+4)),desc="Decoding file {}: \t\t".format(file)):
                        temp=CorrectionCodes.decoding_M4(m[i:i+bit_count+4])
                        if temp is not None:
                            dm=np.append(dm,temp)
                        else:
                            is_correct=False
                            break
                    if is_correct:
                      decoded_data[file[len(filePrefix):-4]]=CorrectionCodes.bits_to_data(dm)  
            elif pr[0]=='H':
                with open(file, 'rb') as f:
                    m=CorrectionCodes.data_to_bits(np.fromfile(f,dtype=np.uint8))
                    dm=np.array([],dtype=np.uint8)
                    is_correct=True
                    pr_lv2=parse('{}.{}',pr[1])
                    bit_count=int(pr_lv2[1])
                    codes=np.array([int(pr_lv2[0]),int(pr_lv2[1])])
                    for i in tqdm(range(0,m.shape[0], bit_count),desc="Decoding file {}:  \t".format(file)):
                        temp=CorrectionCodes.decoding_Hamming(m[i:i+bit_count],codes)
                        if temp is not None:
                            dm=np.append(dm,temp)
                        else:
                            is_correct=False
                            break
                    if is_correct:
                      decoded_data[file[len(filePrefix):-4]]=CorrectionCodes.bits_to_data(dm)
        return decoded_data

    ########################################################################
    def coding_none(data):
        return data
    def decoding_none(data):
        return data
    ########################################################################
    def coding_M1(data):
        out=np.zeros((data.shape[0]*3,))
        out[0::3]=data
        out[1::3]=data
        out[2::3]=data
        return out
    def decoding_M1(data):
        out=np.array([],dtype=np.uint8)
        for i in range(0,data.shape[0],3):
            out=np.append(out,(np.sum(data[i:i+3])/3)>0.5)
        return out
    ########################################################################
    def coding_M2(data):
        out=data.copy()
        out= np.append(out,CorrectionCodes.parity(out)) 
        out= np.append(out,CorrectionCodes.parity(out))
        return out
    def decoding_M2(data):
        if (CorrectionCodes.parity(data[0:-2])==data[-2]) and (CorrectionCodes.parity(data[0:-1])==data[-1]):
            return data[0:-2]
        return None
    ########################################################################
    def coding_M3(data):
        out=data.copy()
        out= np.append(out,CorrectionCodes.parity(data[0::2])) 
        out= np.append(out,CorrectionCodes.parity(data[1::2]))
        return out
    def decoding_M3(data):
        if (CorrectionCodes.parity(data[0:-2:2])==data[-2]) and (CorrectionCodes.parity(data[1:-2:2])==data[-1]):
            return data[0:-2]
        return None
    ########################################################################    
    def coding_M4(data):
        out=data.copy()
        m=(data.shape[0])//2
        out= np.append(out,CorrectionCodes.parity(data[0::2])) 
        out= np.append(out,CorrectionCodes.parity(data[1::2]))
        out= np.append(out,CorrectionCodes.parity(data[0:m])) 
        out= np.append(out,CorrectionCodes.parity(data[m:]))
        return out
    def decoding_M4(data):
        m=(data.shape[0]-4)//2
        if ((CorrectionCodes.parity(data[0:-4:2])==data[-4]) and (CorrectionCodes.parity(data[1:-4:2])==data[-3]) 
            and (CorrectionCodes.parity(data[0:m])==data[-2]) and (CorrectionCodes.parity(data[m:-4])==data[-1])):
            return data[0:-4]
        return None
    ########################################################################
    def coding_Hamming(data,codes=np.array([0,0])):
        if codes[1]==0:
            if codes[0]==0:
                codes[0]=data.shape[0]
            if codes[0]==1:
                codes=np.array([1,4])
            elif codes[0]<=4:
                codes=np.array([4,8])
            elif codes[0]<=11:
                codes=np.array([11,16])
            elif codes[0]<=26:
                codes=np.array([26,32])
            elif codes[0]<=57:
                codes=np.array([57,64])
            elif codes[0]<=120:
                codes=np.array([120,128])
            elif codes[0]<=247:
                codes=np.array([247,256])
            else:
                return None
        out=np.zeros((codes[1],),dtype=np.uint8)
        idx=np.arange(0,codes[1])
        del_idx=np.array([0,1])
        for i in range(1,(codes[1]-codes[0])-1):
            del_idx=np.append(del_idx,2**i)
        idx = np.delete(idx,del_idx)
        out[idx[:data.shape[0]]]=data
        for i in del_idx[1:]:
            out[i]=CorrectionCodes.parity(out[idx[np.bitwise_and(idx,i)>0]])
        out[0]=CorrectionCodes.parity(out[1:])
        return out, codes
    def decoding_Hamming(data,codes):
        # out=np.zeros(codes[0],)
        idx=np.arange(0,codes[1])
        del_idx=np.array([0,1])
        for i in range(1,(codes[1]-codes[0])-1):
            del_idx=np.append(del_idx,2**i)
        idx = np.delete(idx,del_idx)
        if not ((data==0).all()):
            fix=reduce(op.xor,[i for (i,b) in enumerate(data) if b])
            if fix>0:
                data[fix] = not data[fix]
            elif (fix==0) and (data[0]  != CorrectionCodes.parity(data[1:])):
                data[fix] = not data[fix]
        if (data[0]  != CorrectionCodes.parity(data[1:])):
            return None
        return data[idx]
    ########################################################################

def load_image(path, infilename) :
    img = Image.open(path + infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def main_text():

    correctionCodes = CorrectionCodes()

    # path = 'Lab11/'
    # title = 'rysunek.png'
    # image = load_image(path, title)
    
    # encoded = encode_rle(image)
    # decoded = decore_rle(encoded)
    # print(image.size)
    # print(encoded.size)
    # print(decoded.size)

    # img = (img[:, :, 0]).flatten()

    # plt.imshow(img)
    # plt.show()
    
    # CorrectionCodes.Write(img, filePrefix="Lab11/icon247", bit_count=247)

    # decoded = CorrectionCodes.Read(filePrefix="Lab11/icon247")

    with open('Lab11/text.txt', 'r', encoding='utf-8') as text_file:
        text_string = text_file.read()
    
    text = np.frombuffer(str.encode(text_string), dtype=np.uint8)

    # print(img.size)


    # CorrectionCodes.Write(text, filePrefix="Lab11/text", bit_count=11)
    decoded = CorrectionCodes.Read(filePrefix="Lab11/text")

    # print("m0", decoded['M0_26'].size)
    # print("m1", decoded['M1_26'].size)
    # print("m2", decoded['M2_26'].size)
    # print("m3", decoded['M3_26'].size)
    # print("m4", decoded['M4_26'].size)
    # print("MH", decoded['MH_26.32'].size)

    string = ''
    
    # for i in decoded['MH_11.16']:
    #     string += str(chr(i))

    for i in decoded['MH_11.16']:
        string += str(chr(i))

    # print(decoded['MH_26.32'])

    print(string)

    # m0_11 = np.frombuffer(str.decode(text_string), dtype=np.uint8)

    # print(''.join(m0_11))

    # f = open("M0_11_decoded.txt", "a")
    # f.write("Now the file has more content!")
    # f.close()

    # plt.imshow(img)
    # plt.show()
    

if __name__ == "__main__":
    main_text()
