from pickletools import uint8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from jpeg import *

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat='Lab09/'                               # katalog z plikami wideo
plik="clip_1.mp4"                       # nazwa pliku
ile=16                                # ile klatek odtworzyc? <0 - calosc
key_frame_counter=3                  # co ktora klatka ma byc kluczowa i nie podlegac kompresji
plot_frames=np.array([])           # automatycznie wyrysuj wykresy
auto_pause_frames=np.array([])        # automatycznie zapauzuj dla klatki i wywietl wykres
subsampling="4:4:0"                     # parametry dla chorma subsamplingu
wyswietlaj_kaltki=True                  # czy program ma wyswietlac kolejene klatki
rle = False

##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        self.Y=None
        self.Cb=None
        self.Cr=None


def compress_frame(Y,Cb,Cr, key_frame_Y, key_frame_Cb, key_frame_Cr, rle, key):

    info = ImageInfo(Y.shape[0], Y.shape[1])

    y_frame = Y.copy().astype(np.int)
    cb_frame = Cb.copy().astype(np.int)
    cr_frame = Cr.copy().astype(np.int)

    if not key:
        y_frame = (y_frame - key_frame_Y)
        print(y_frame[256:268, 256:268])
        cb_frame = (cb_frame- key_frame_Cb)
        print(cb_frame[256:268, 256:268])
        cr_frame = (cr_frame - key_frame_Cr)
        print(cr_frame[256:268, 256:268])
        print("******")

    data.Y=compress(y_frame, subsampling, QY, rle, True)
    data.Cb=compress(cb_frame, subsampling, QC, rle, False)
    data.Cr=compress(cr_frame, subsampling, QC, rle, False)

    return data, info

def decompress_frame(data,  key_frame_Y, key_frame_Cb, key_frame_Cr , info, rle, key):

    y_frame = decompress(data.Y, subsampling, QY, info, rle, True).astype(np.int)
    cb_frame = decompress(data.Cb, subsampling, QC, info, rle, False).astype(np.int)
    cr_frame = decompress(data.Cr, subsampling, QC, info, rle, False).astype(np.int)

    if not key:
        y_frame = (key_frame_Y + y_frame)
        cb_frame = (key_frame_Cb + cb_frame)
        cr_frame = (key_frame_Cr + cr_frame)

    # print(y_frame[500:510, 500:510])
    # print(cb_frame[500:510, 500:510])
    # print(cr_frame[500:510, 500:510])
    # print("******")

    y_frame = np.clip(y_frame, 0, 255)
    return np.dstack([y_frame,cb_frame,cr_frame]).astype(np.uint8)


##############################################################################
####     Głowna petla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(kat+'\\'+plik)

if ile<0:
    ile=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Normal Frame')
cv2.namedWindow('Decompressed Frame')

compression_information=np.zeros((3,ile))

frame_counter = 0

for i in range(ile):
    ret, frame = cap.read()
    if wyswietlaj_kaltki:
        cv2.imshow('Normal Frame',frame)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    if (i % key_frame_counter)==0: # pobieranie klatek kluczowych
        key_frame = frame
        cdata, info = compress_frame(frame[:,:,0],frame[:,:,2],frame[:,:,1], 0, 0, 0, rle, True)
        cY=cdata.Y
        cCb=cdata.Cb
        cCr=cdata.Cr
        d_frame = decompress_frame(cdata, 0, 0, 0, info, rle, True)
        key_frame = d_frame
    else: # kompresja
        cdata, info=compress_frame(frame[:,:,0],frame[:,:,2],frame[:,:,1], key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], rle, False)
        cY=cdata.Y
        cCb=cdata.Cb
        cCr=cdata.Cr
        d_frame= decompress_frame(cdata, key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], info, rle, False)
    
    compression_information[0,i]= (frame[:,:,0].size - cY.size)/frame[:,:,0].size
    compression_information[1,i]= (frame[:,:,2].size - cCb.size)/frame[:,:,2].size
    compression_information[2,i]= (frame[:,:,1].size - cCr.size)/frame[:,:,1].size  
    if wyswietlaj_kaltki:
        print(frame_counter)
        frame_counter += 1
        cv2.imshow('Decompressed Frame', cv2.cvtColor(d_frame, cv2.COLOR_YCrCb2RGB))
        #cv2.imshow('Decompressed Frame',d_frame)
    
    if np.any(plot_frames==i): # rysuj wykresy
        # bardzo słaby i sztuczny przyklad wykrozystania tej opcji
        fig, axs = plt.subplots(1, 3 , sharey=True   )
        fig.set_size_inches(16,5)
        axs[0].imshow(frame)
        axs[2].imshow(d_frame) 
        diff=frame.astype(float)-d_frame.astype(float)
        print(np.min(diff),np.max(diff))
        axs[1].imshow(diff,vmin=np.min(diff),vmax=np.max(diff))
        
    if np.any(auto_pause_frames==i):
        cv2.waitKey(-1) #wait until any key is pressed
    
    k = cv2.waitKey(1) & 0xff
    
    if k==ord('q'):
        break
    elif k == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

plt.figure()
plt.plot(np.arange(0,ile),compression_information[0,:]*100)
plt.plot(np.arange(0,ile),compression_information[1,:]*100)
plt.plot(np.arange(0,ile),compression_information[2,:]*100)
plt.show()