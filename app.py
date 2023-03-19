import os

from modules import talking_face
# print(talking_face.TTS)

def get_all_driving_video_path():
  output,filename = os.path.split(os.path.abspath(__file__))
  res=[]
  for f in os.listdir(os.path.join(output,'driving_video/')):
     res.append(os.path.join(output,'driving_video/'+f))
  return res
def get_driving_video_path(driving_video_type='base'):
  output,filename = os.path.split(os.path.abspath(__file__))
  fp=os.path.join(output,'driving_video/'+driving_video_type+'.mp4')
  return fp
def get_portrait_video_path():
  output,filename = os.path.split(os.path.abspath(__file__))
  fp=os.path.join(output,'data/portrait_video.mp4')
  return fp
def get_user_portrait_img_root_path():
  output,filename = os.path.split(os.path.abspath(__file__))
  fp=os.path.join(output,'data/')
  return fp
def get_wav_file_path():
  output,filename = os.path.split(os.path.abspath(__file__))
  fp=os.path.join(output,'data/wav_file.wav')
  return fp
def get_gif_path():
  output,filename = os.path.split(os.path.abspath(__file__))
  fp=os.path.join(output,'data/portrait_gif.gif')
  return fp


fom=talking_face.FOM(get_portrait_video_path())
tts=talking_face.TTS()
wav2lip=talking_face.Wav2lip(get_wav_file_path())

def get_all_driving_video():
  data=[]
  for file in get_all_driving_video_path():
     output,filename = os.path.split(file)
     data.append({
        'base64':'data:video/mp4;base64,'+talking_face.encode_base64(file),
        'emotion':filename.split('.')[0]
     })
  return data

def create_avatar(portrait_file,driving_video,result_type='gif',is_base64=True):
  fom.update_driving_video(driving_video)
  portrait_video=fom.run(portrait_file,result_type=='gif',is_base64)
  return portrait_video

def create_avatar_video(portrait_file,driving_video,text):
  fom.update_driving_video(driving_video)
  portrait_video=fom.run(portrait_file,False,False)
  input_audio=tts.text2audio(text)
  # print(input_audio)
  res=wav2lip.run(portrait_video,input_audio)

  return res 

   

# pip install fastapi
# pip install uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"message": "Hello World"}



class Avatar(BaseModel):
    name: str
    dialogue:str
    base64:str
    filename:str
    emotion:str
    type:str
    isBase64:bool

@app.post("/driving_video")
async def driving_video_api():
    data=get_all_driving_video()
    data=[{
       "base64":d['base64'],
       "element":'video',
       "emotion":d['emotion']
    } for d in data]
    return {"data": data}

@app.post("/create_avatar")
async def create_avatar_api(avatar: Avatar):
    # print(avatar)
    fp=talking_face.save_img_base64(avatar.base64,os.path.join(get_user_portrait_img_root_path(),avatar.filename))
    if avatar.type=='gif':
      res=create_avatar(fp,get_driving_video_path(avatar.emotion),avatar.type,avatar.isBase64)
    elif avatar.type=='mp4':
      res=create_avatar_video(fp,get_driving_video_path(avatar.emotion),avatar.dialogue)
    return {
       "data":{
       "base64":res,
       "type":avatar.type
       }
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app',host="127.0.0.1",port=8080, reload=True)



# with gr.Blocks() as demo:
    
#     with gr.Column():
#         # avatar
#         # input_text=gr.Textbox()
#         input_i=gr.Image(type='filepath')
#         input_file_type=gr.Radio(["mp4", "gif"],value='mp4', label="文件类型")
#         input_driving_video=gr.Video(format='mp4')
#         #input_driving_video_type=gr.Radio(["normal"],value='base', label="文件类型")
#         # output_video1=gr.Video(format='mp4')
#         # output_gif=gr.Image(type="numpy")
#         gr.Examples(
#             examples=[get_driving_video_path(x) for x in ['normal','1']],
#             inputs=input_driving_video,
#         )
        
#         btn = gr.Button(value="创建形象")
#         btn.click(create_avatar, inputs=[input_i,input_file_type,input_driving_video], outputs=[gr.HTML()],api_name='create_avatar')
#     with gr.Column():
#         # talking
#         input_file_type=gr.Radio(["wav", "text"],value='wav', label="输入类型")
#         input_audio=gr.Audio(label="录音",type="numpy",source='microphone')
#         input_talk_text=gr.Textbox()
#         output_video2=gr.Video()
#         btn2 = gr.Button(value="生成视频")
#         btn2.click(test2, inputs=[input_talk_text,input_audio,input_file_type], outputs=[output_video2],api_name='create_talk_face')
 