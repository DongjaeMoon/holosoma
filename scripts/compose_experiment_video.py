from pathlib import Path
import subprocess
import cv2
import numpy as np
import shutil

cv2.setNumThreads(1)
root=Path('/workspace/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513/evaluation')
models=('biHAM','sym_aug_HAM','HAM')
colors=((220,140,40),(40,160,240),(100,200,80))
caps=[[cv2.VideoCapture(str(next((root/'videos'/m/'model_0010000'/(d+'_0.6mps')).glob('*.mp4')))) for m in models] for d in ('left','right')]
fps=caps[0][0].get(5)
frames=min(int(c.get(7)) for group in caps for c in group)
out=root/'training_vs_deploy.mp4'
process=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x400','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
for index in range(frames):
    panels=[]
    for side,group in enumerate(caps):
        images=[]
        for cap in group:
            ok,frame=cap.read()
            assert ok
            images.append(frame[78:])
        gray=cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
        panel=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR).astype(np.float32)*.48
        panel+=np.array((48,40,32) if side==0 else (38,43,48),np.float32)
        for img,color in zip(images,colors):
            small=cv2.resize(img,(320,141))
            mask=np.zeros((141,320),np.uint8)
            cv2.grabCut(small,mask,(112,25,96,115),np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),1,cv2.GC_INIT_WITH_RECT)
            alpha=cv2.resize(((mask==1)|(mask==3)).astype(np.float32),(640,282))
            alpha=cv2.GaussianBlur(alpha,(3,3),0)*.63
            intensity=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            tinted=np.asarray(color,np.float32)[None,None,:]*(.45+.75*intensity[...,None])
            panel=panel*(1-alpha[...,None])+tinted*alpha[...,None]
        canvas=np.full((400,640,3),(24,22,20),np.uint8)
        canvas[94:376]=np.clip(panel,0,255).astype(np.uint8)
        cv2.putText(canvas,'TRAIN COMMAND' if side==0 else 'DEPLOY COMMAND',(28,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(238,238,238),1,cv2.LINE_AA)
        start,end=((400,63),(240,63)) if side==0 else ((240,63),(400,63))
        cv2.arrowedLine(canvas,start,end,(245,245,245),4,cv2.LINE_AA,tipLength=.16)
        for x,name,color in zip((28,210,455),('biHAM','sym-aug HAM','HAM'),colors):
            cv2.circle(canvas,(x,389),4,color,-1,cv2.LINE_AA)
            cv2.putText(canvas,name,(x+10,394),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
        panels.append(canvas)
    combined=np.concatenate(panels,axis=1)
    combined[:,638:642]=(225,225,225)
    process.stdin.write(combined.tobytes())
    if index==int(fps*5): cv2.imwrite(str(root/'training_vs_deploy_preview.png'),combined)
process.stdin.close()
assert process.wait()==0
for group in caps:
    for cap in group: cap.release()
print(out,flush=True)
