from datasets import load_dataset
# from transformers import WhisperProcessor
from audio_preprocessing.datasets import AudioDataset
# import librosa

# FSD50K File

def load_fsd50k() -> AudioDataset:
    fsd50k_ds = load_dataset("Chand0320/fsd50k_hf")
    fsd50k = AudioDataset(data=fsd50k_ds)
    # print(type(fsd50k_ds["train"]))
    return fsd50k

largest_size = 0

## A lot of this was me testing FSD50k Loading
## However, this caused issues with our environment.
## Trouble is librosa hasn't had updates for a year and a half
## We should move to other things to handle what librosa used to handle
# if __name__ == "__main__":
#     fsd50k = load_fsd50k()
#     print(fsd50k["train"][0]["audio"])
#     for i in range(fsd50k["train"].shape[0]):
#         samples = fsd50k["train"][i]["audio"].get_all_samples()
        
#         waveform = samples.data
#         sample_rate = samples.sample_rate


#         largest_size = max(waveform.shape[1], largest_size)

#     #    y, sr = librosa.load(librosa.ex('trumpet'))
#         print(librosa.feature.melspectrogram(y=waveform.numpy(), sr=sample_rate).shape)
#         print(samples)
#         print(largest_size)


#         processor = WhisperProcessor.from_pretrained("openai/whisper-base")


#         input_features = processor(
#             waveform[0],
#             sampling_rate=16000,
#             return_tensors="pt",
#             truncation=True,
#             max_length=16000*5
#         ).input_features
#         print(input_features.shape)
#         input()

    

    