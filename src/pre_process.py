from tqdm import tqdm
import json
import random
import numpy as np
import os
import wave

import librosa
import librosa.display
import matplotlib.pyplot as plt

res_path  = "/home/brsf11/Hdd/ML/Dataset/MobvoiHotwords/mobvoi_hotword_dataset_resources/"
data_path = "/home/brsf11/Hdd/ML/Dataset/MobvoiHotwords/mobvoi_hotword_dataset/"

output_path = "/home/brsf11/Hdd/ML/Dataset/MobvoiHotwords/output/"

elem = ["-","garbage","hi","xiao","wen"]

class Dataset(object):
    def __init__(self,res_path,type="train",set_size=10000,p_portion=0.1):
        n_file_name = "n_"+type+".json"
        p_file_name = "p_"+type+".json"

        n_file = open(res_path+n_file_name,'r')
        p_file = open(res_path+p_file_name,'r')

        n_json = json.loads(n_file.read())
        p_json = json.loads(p_file.read())

        n_file.close()
        p_file.close()

        self.set_size = set_size
        self.set_p_portion = p_portion
        self.set_p_size = int(self.set_p_portion * self.set_size)
        self.set_n_size = self.set_size - self.set_p_size

        self.n_data_size = len(n_json)
        self.p_data_size = len(p_json)

        self.p_data_tag0_size = 0
        for p_data in p_json:
            if p_data['keyword_id'] == 0:
                self.p_data_tag0_size += 1
        self.p_data_tag1_size = self.p_data_size - self.p_data_tag0_size

        self.n_index = random.sample(np.arange(self.n_data_size).tolist(),self.set_n_size)
        self.p_index = random.sample(np.arange(self.p_data_tag0_size).tolist(),self.set_p_size)

        self.n_file = []
        for index in self.n_index:
            self.n_file.append(n_json[index]['utt_id']+".wav")
        self.p_file = []
        for index in self.p_index:
            self.p_file.append(p_json[index]['utt_id']+".wav")

def melSpec(file):
    y, sr = librosa.load(file,sr=16000)
    # plt.plot(y);
    # plt.title('Signal');
    # plt.xlabel('Time (samples)');
    # plt.ylabel('Amplitude');

    duration = 3
    if len(y)/sr < duration:
        y = np.append(y,np.zeros(((duration * sr) - len(y) - 1),dtype=np.dtype(y[0])))
    else:
        y = y[0:(sr * duration - 1)]

    NFFT = 512
    window_length = int(0.025 * sr)
    hop_length = int(0.01 * sr)
    # spec = np.abs(librosa.stft(y, n_fft=NFFT,hop_length=hop_length,win_length=window_length))
    # spec = librosa.amplitude_to_db(spec, ref=np.max)
    # librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log');
    # plt.colorbar(format='%+2.0f dB');
    # plt.title('Spectrogram');
    
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=NFFT,hop_length=hop_length,win_length=window_length,n_mels=40)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)


    # librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
    # plt.title('Mel Spectrogram');
    # plt.colorbar(format='%+2.0f dB');

    # plt.show()
    return mel_spect

if __name__ == "__main__":
    set_size = 20
    batch_size = 64
    for i in tqdm(range(set_size),position=1,ascii=True,leave=False):
        ds = Dataset(res_path=res_path,set_size=batch_size,p_portion=0.1)
        # portion = 0
        # for file in ds.p_file:
        #     wav = wave.open(data_path+file,'r')
        #     if wav.getnframes() /  wav.getframerate()> 3:
        #         print(f"{wav.getnframes() /  wav.getframerate():>4f} s")
        #         portion += 1

        # portion /= 1000
        # print(f"Portion = {portion:>4f}")
        # os.system("aplay "+data_path+ds.p_file[0])
        data = np.zeros((40,300),dtype=float).tolist()
        n_tag = "1"
        p_tag = "0203040"
        # print(json.dumps(out_temp,separators=[',',':']))
        output = []

        for file in tqdm(ds.n_file,position=0,ascii=True,leave=False):
            out_temp = {'data': melSpec(data_path+file).tolist(),'tag': n_tag}
            output.append(out_temp)
        for file in tqdm(ds.p_file,position=0,ascii=True,leave=False):
            out_temp = {'data': melSpec(data_path+file).tolist(),'tag': p_tag}
            output.append(out_temp)

        out_file = open(output_path+"output"+str(i)+".json",'w')
        out_file.write(json.dumps(output,separators=[',',':'])) 
        out_file.close()

    print("\n\n")
    
