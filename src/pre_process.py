from tqdm import tqdm
import json
import random
import numpy as np
import scipy.io.wavfile

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
    sr, y = scipy.io.wavfile.read(file)


    duration = 3
    if len(y)/sr < duration:
        y = np.append(y,np.zeros(((duration * sr) - len(y) - 1),dtype=np.dtype(y[0])))
    else:
        y = y[0:(sr * duration - 1)]

    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 分帧
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = int(round(frame_size*sr))
    frame_step = int(round(frame_stride*sr)) 
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))

    indices = np.tile(np.arange(0,frame_length),(num_frames,1))+np.tile(np.arange(0,num_frames*frame_step,frame_step), (frame_length, 1)).T
    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

    # 加汉明窗
    frames *= np.hamming(frame_length)
    # Explicit Implementation
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))

    # 傅里叶变换和功率谱
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)


    # 将频率转换为Mel频率
    low_freq_mel = 0

    nfilt = 40
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sr)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    mel_spect = 20 * np.log10(filter_banks)  # dB
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
        n_tag = [1,1,1,1,1,1,1]
        p_tag = [0,2,0,3,0,4,0]
        # print(json.dumps(out_temp,separators=[',',':']))
        output = []

        for file in tqdm(ds.n_file,position=0,ascii=True,leave=False):
            out_temp = {'data': melSpec(data_path+file).tolist(),'tag': n_tag,'len': 1}
            output.append(out_temp)
        for file in tqdm(ds.p_file,position=0,ascii=True,leave=False):
            out_temp = {'data': melSpec(data_path+file).tolist(),'tag': p_tag,'len': 7}
            output.append(out_temp)

        out_file = open(output_path+"output"+str(i)+".json",'w')
        out_file.write(json.dumps(output,separators=[',',':'])) 
        out_file.close()

    print("\n\n")
    
