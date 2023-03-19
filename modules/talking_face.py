import base64,os
import cv2
from PIL import Image


class TTS(object):
    from paddlespeech.cli.tts.infer import TTSExecutor
    tts = TTSExecutor()

    def __init__(self, device='cpu'):
        self.device = device

    def text2audio(self,text):
        print('text2audio - run',text)
        out_file =self.tts(text,device=self.device)
        print('out_file',out_file)
        return out_file

class FOM(object):

    def __init__(self,output_path):
        from ppgan.apps.first_order_predictor import FirstOrderPredictor
        output,filename = os.path.split(output_path)
        self.output_path=output_path
        self.first_order_predictor = FirstOrderPredictor(output = output,filename = filename, 
                              face_enhancement = True, 
                              ratio = 0.4,
                              relative = True,
                              image_size=512,
                              adapt_scale = True)
        
    def update_driving_video(self,fp):
        self.driving_video=fp
    
    def run(self, source_image,is_gif=False,is_base64=False):
        self.first_order_predictor.run(source_image, self.driving_video)

        if is_gif:
            gif_file= convert_mp4_to_gif(self.output_path)
            if is_base64:
                return 'data:image/gif;base64,'+encode_base64(gif_file)
            else:
                return gif_file
        else:
            if is_base64:
                return 'data:video/mp4;base64,'+encode_base64(self.output_path)
            else:
                return self.output_path
            
class Wav2lip(object):

    def __init__(self,out_file):
        from ppgan.apps.wav2lip_predictor import Wav2LipPredictor
        self.wav2lip_predictor = Wav2LipPredictor(face_det_batch_size = 2,wav2lip_batch_size = 16,face_enhancement = True)
       
        self.out_file=out_file
        
    def run(self,input_video,input_audio):
        self.wav2lip_predictor.run(input_video, input_audio, self.out_file)
        return self.out_file


# 转video 2 gif
def convert_mp4_to_gif(input_file):
    '''
    传参 :  input_file 视频文件名
            output_file gif文件名
            duration 每帧图像的停留时间 毫秒ms 
            step 跳帧,降低采样率,减小gif体积
    返回 : 无
    '''
    output,filename = os.path.split(input_file)
    video_capture = cv2.VideoCapture(input_file)
    rate=video_capture.get(5)
    still_reading, image = video_capture.read()
    frame_count = 0
    i = 0
    frames = []

    #单位是秒
    step=int(rate*0.2)
    duration=1000*step/rate
 
    print('rate',rate)
    print('step',step)
    print('duration',duration)
        
    while still_reading:
        still_reading, image = video_capture.read()
        if not still_reading:
            break
        if i>step:
            frames.append(Image.fromarray(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)))
            frame_count += 1
            i=0
        i+=1
    
    # v_duration=frame_count/rate
    output_file=os.path.join(output,filename.split('.')[0]+'.gif')
    frame_one = frames[0]
    frame_one.save(output_file, format="GIF", append_images=frames[1:],save_all=True, duration=duration, loop=0)
    return output_file


# 读取文件 编码成base64
def encode_base64(file):
    with open(file, 'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        # print(type(base64_data))
        # print(base64_data)
        # 如果想要在浏览器上访问base64格式图片，需要在前面加上：data:image/jpeg;base64,
        base64_str = str(base64_data, 'utf-8')
        # print(base64_str)
        return base64_str
    
def save_img_base64(data,file_path):
    imagedata = base64.b64decode(data)
    # print(imagedata)
    file = open(file_path,"wb")
    file.write(imagedata)
    file.close()
    return file_path