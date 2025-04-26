# Voice-to-voice-StyleTransfer - Streaming Puretalk TTS+ v0.1.2


Direct Wave Neural Style Transfer (DWavNST) invented by Puretalk team



## Voice conversion - direct wav (no conversion to MEL spectrum or vocoder)
## idea. take a synthetic voice and apply the style of any speaker
The input ***(tts)*** voice is disentangled to content representation and the target voice ***(gabby)*** is used to reconstruct the voice.  The speech is only converted and true sounding speech is only refective of the content speech. The purpose is to be able to effectively extract style/features of a target voice and apply to a disentagled TTS voice.

Each voice will have it's own latent space that contains the unique style of the speaker.  The latent space, the style transfer, is controlled by a reference wav file.

The voice is wav file in to wav file out, no intermediate sound representation.


## Docker image - nnabla and supporting python libraries.
Build the docker image
```bash
bash scripts/docker_build.sh

Need to pip install pyloudnorm as it is not in the docker file.
```

## Data set
We use the [VCTK data set](https://datashare.ed.ac.uk/handle/10283/3443). Download the dataset, then run the following command to prepare trianing and validation sets.  The data is used to augment internal voice data.

Internal data sets gabby (gab) and TTS (tts) (text-to-speech) used to obtain tts to gabby voice conversion.

```bash
python preprocess.py -i <path to `VCTK-Corpus/wav48/`> \
       -o <path to save precomputed inputs> \
       -s data/list_of_speakers.txt \
       --make-test
```
- List of speakers used for training the model can be found [here](data/list_of_speakers.txt).
- List of speakers used for the traditional subjective evaluation can be found [here](data/list_of_sub.txt).
- Gender information about speakers can be found [here](data/speaker-info.txt).


## Training
All hyper-parameters used for training are defined at [hparams.py](hparams.py). These parameters can also be changed in the command line.
```bash
python main.py -c cudnn -d <list of GPUs e.g., 0,1,2,3> \
       --output_path log/baseline/ \
       --batch_size 8 \
       ...
```

## Inference
The conversion can be performed as follows.
```bash
python inference.py -c cudnn -d <list of GPUs e.g., 0,1> \
       -m <path to pretrained model> \
       -i <input audio file> \
       -r <reference audio file> \
       -o <output audio file>
```

```
Directories:
wav_in  - tts input - Content
wav_out - neural style transfered output wavs
wav_ref - reference wav for the specific voice style to choose for neural style transfer
```


            +-------------------+        +------------------+
TTS Input   | Content Encoder    |        | Style Encoder     |    Reference Speaker
(Voice.wav) | (1D CNN over Wav)   |        | (1D CNN over Wav) |    (Voice.wav)
            +---------+----------+        +---------+--------+
                      |                           |
                Content Embedding             Style Embedding
                      |                           |
                      +-------------+-------------+
                                    |
                               Fusion Module
                       (AdaIN-like or FiLM over embeddings)
                                    |
                         +----------v-----------+
                         | Waveform Decoder      |
                         | (Upsampling 1D ConvNet |
                         |   or Diffusion Model)  |
                         +----------+------------+
                                    |
                             Stylized Output
                               (Voice.wav)

