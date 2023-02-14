from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlenlp import Taskflow
import paddle
from ppgan.apps.wav2lip_predictor import Wav2LipPredictor

# # https://github.com/JiehangXie/PaddleBoBo/blob/0.1/PaddleTools/GAN.py
import base64
import librosa
import soundfile as sf

def resample_rate(path,new_sample_rate = 16000):

    signal, sr = librosa.load(path, sr=None)
    wavfile = path.split('/')[-1]
    wavfile = wavfile.split('.')[0]
    file_name = wavfile + '_new.wav'
    new_signal = librosa.resample(signal, sr, new_sample_rate) # 
    #librosa.output.write_wav(file_name, new_signal , new_sample_rate) 
    sf.write(file_name, new_signal, new_sample_rate, subtype='PCM_24')
    print(f'{file_name} has download.')
    return file_name




asr = ASRExecutor()
tts = TTSExecutor()
dialogue = Taskflow("dialogue")
wav2lip_predictor = Wav2LipPredictor(face_det_batch_size = 2,
                                 wav2lip_batch_size = 16,
                                 face_enhancement = True)


def audio2text(input_file):

    # input_file="input.wav"
    # file = open(input_file,"wb")   # 写入二进制文件
    # text = base64.b64decode(b)   # 进行解码
    # file.write(text)
    # file.close()

    input_file=resample_rate(input_file,new_sample_rate = 16000)

    result = asr(audio_file=input_file,device=paddle.get_device())
    print('audio2text')
    return result

def text2audio(text):
    # out_file="output.wav"
    out_file =tts(text)
    file = open(out_file,"wb")
    content=file.read()
    b = base64.b64encode(content)   # 进行解码
    file.close()
    print('text2audio')
    return out_file

def reply(t):
    data = [t]
    result = dialogue(data)
    print('reply',result[0])
    return result[0]

def wav2lip(input_video,input_audio):
    out_file='video.mp4'
    wav2lip_predictor.run(input_video, input_audio, out_file)
    # file = open(out_file,"wb")
    # content=file.read()
    # b = base64.b64encode(content)   # 进行解码
    # file.close()

    return out_file

def write_wav(data, samplerate,wav_file):
    
    # data, samplerate = sf.read('existing_file.wav')
    sf.write(wav_file, data, samplerate)
    return wav_file

import os
import gradio as gr

def greet(wav_file_input):
    wav_file=write_wav(wav_file_input[1],wav_file_input[0],'./wav_file.wav')
    text=audio2text(wav_file)
    q=reply(text)
    input_audio=text2audio(q)
    result=wav2lip(os.path.join(os.path.dirname(__file__), 
                                     '00001-4231494536.png'),input_audio)
    # os.path.join(os.path.dirname(__file__),  "video.mp4")

    return result
iface = gr.Interface(fn=greet, inputs=[gr.Audio(label="录音",type="numpy")], outputs=[gr.Video()])
iface.launch()


# from fastapi import FastAPI, APIRouter,Body
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse

# app = FastAPI()
# api_router = APIRouter()

# origins = ['*']

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @api_router.post("/api", status_code=200)
# def api(data=Body(None)) -> dict:
#     text=audio2text(data['audio'])
#     q=reply(text)
#     # input_audio=text2audio(q)
#     # result=wav2lip('00001-4231494536.png',input_audio)
#     return {"msg":q}

# @api_router.get("/", response_class=HTMLResponse)
# def root() -> dict:
#     html_file = open("index.html", 'r').read()
#     return html_file


# app.include_router(api_router)



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5555)