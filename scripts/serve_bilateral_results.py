"""Host-side localhost gallery and TensorBoard proxy for the Docker experiment queue."""
import argparse
from functools import partial
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from urllib.parse import quote

PAGE = '''<!doctype html><meta charset="utf-8"><title>Holosoma experiments</title>
<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;background:#f6f7fa;color:#172133}select{padding:8px;margin:8px}video{width:100%;background:#111}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;padding:14px;border-radius:10px}pre{white-space:pre-wrap;font-size:13px}</style>
<h1>Holosoma experiments · GPU 3</h1>
<p><a id="tb">TensorBoard</a> · <a href="/?files=1">Files and logs</a></p>
<p>Training: vanilla → bilateral → bilateral without rightward commands. Videos appear after training.</p>
<label>Experiment <select id="run"></select></label><label>Checkpoint <select id="checkpoint"></select></label>
<div class="grid" id="videos"></div><h2>Queue and VRAM reports</h2><pre id="status"></pre>
<script>
let data={videos:[]};const run=document.querySelector('#run'),checkpoint=document.querySelector('#checkpoint');
document.querySelector('#tb').href='http://'+location.hostname+':6006';
function chooseCheckpoint(){let old=checkpoint.value;let names=[...new Set(data.videos.filter(x=>x.run===run.value).map(x=>x.checkpoint))].sort().reverse();checkpoint.replaceChildren(...names.map(x=>new Option(x,x)));if(names.includes(old))checkpoint.value=old;showVideos()}
function showVideos(){let box=document.querySelector('#videos');box.replaceChildren();let clips=data.videos.filter(x=>x.run===run.value&&x.checkpoint===checkpoint.value);for(let x of clips){let card=document.createElement('div');card.className='card';let title=document.createElement('h3');title.textContent=x.direction;let video=document.createElement('video');video.controls=true;video.preload='none';video.src=x.url;let link=document.createElement('a');link.href=x.metrics;link.textContent='Evaluation metrics';card.append(title,video,link);box.append(card)}if(!clips.length)box.textContent='No videos yet. Training is running; checkpoint videos will be generated automatically afterward.'}
async function refresh(){let response=await fetch('/api/manifest');data=await response.json();document.querySelector('#status').textContent=JSON.stringify(data.status,null,2);let old=run.value;let names=[...new Set(data.videos.map(x=>x.run))].sort();run.replaceChildren(...names.map(x=>new Option(x,x)));if(names.includes(old))run.value=old;chooseCheckpoint()}
run.onchange=chooseCheckpoint;checkpoint.onchange=showVideos;refresh();setInterval(()=>{if(![...document.querySelectorAll('video')].some(v=>!v.paused))refresh()},30000);
</script>'''


class Gallery(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/manifest':
            videos = []
            for path in self.root.rglob('videos/*/model_*/*/*.mp4'):
                relative = path.relative_to(self.root)
                parts = relative.parts
                index = parts.index('videos')
                videos.append(dict(run=parts[index-1]+'/'+parts[index+1], checkpoint=parts[index+2],
                    direction=parts[index+3], url='/'+quote(str(relative)),
                    metrics='/'+quote(str(relative.parent/'metrics.json'))))
            status = [json.loads(p.read_text()) for p in self.root.glob('*/*.json')]
            status += [dict(log=str(p.relative_to(self.root)), tail=p.read_text(errors='replace')[-1800:])
                       for p in self.root.glob('gpu3_launcher.log')]
            body = json.dumps(dict(videos=videos, status=status)).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == '/':
            body = PAGE.encode()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            super().do_GET()


class TensorBoardProxy(BaseHTTPRequestHandler):
    def proxy(self):
        connection = HTTPConnection(self.container_ip, 16006, timeout=60)
        try:
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            headers = {k:v for k,v in self.headers.items() if k.lower() not in {'host','connection'}}
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse(); data = response.read()
            self.send_response(response.status)
            for key,value in response.getheaders():
                if key.lower() not in {'transfer-encoding','connection','content-length'}:
                    self.send_header(key,value)
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        except OSError as exc:
            self.send_error(502, 'TensorBoard is starting: '+str(exc))
        finally:
            connection.close()
    do_GET = do_POST = do_HEAD = proxy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container',default='ycho-holosoma')
    parser.add_argument('--logdir',type=Path,default=Path('logs/bilateral_queue'))
    parser.add_argument('--port',type=int,default=8008)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--gallery-only',action='store_true',help='Serve videos inside Docker without starting a TensorBoard proxy.')
    parser.add_argument('--tensorboard-proxy-only',action='store_true',help='Forward this port to the existing TensorBoard on localhost:16006.')
    args=parser.parse_args()
    if args.tensorboard_proxy_only:
        TensorBoardProxy.container_ip='127.0.0.1'
        proxy=ThreadingHTTPServer((args.host,args.port),TensorBoardProxy)
        print(f'TensorBoard proxy: http://{args.host}:{args.port} -> localhost:16006',flush=True)
        proxy.serve_forever()
        return
    root=args.logdir.resolve(); root.mkdir(parents=True,exist_ok=True)
    Gallery.root=root
    gallery=ThreadingHTTPServer((args.host,args.port),partial(Gallery,directory=str(root)))
    if args.gallery_only:
        print(f'Videos: http://{args.host}:{args.port}',flush=True)
        gallery.serve_forever()
        return
    info=json.loads(subprocess.check_output(['docker','inspect',args.container],text=True))[0]
    TensorBoardProxy.container_ip=next(iter(info['NetworkSettings']['Networks'].values()))['IPAddress']
    proxy=ThreadingHTTPServer((args.host,6006),TensorBoardProxy)
    command='source /workspace/holosoma/scripts/source_isaacsim_setup.sh && exec python /workspace/holosoma/scripts/launch_tensorboard.py --logdir /workspace/holosoma/logs/bilateral_queue --host 0.0.0.0 --port 16006'
    with Path('/tmp/holosoma_tensorboard_server.log').open('a') as log:
        process=subprocess.Popen(['docker','exec','-e','CUDA_VISIBLE_DEVICES=',args.container,'bash','-lc',command],stdout=log,stderr=subprocess.STDOUT)
        threading.Thread(target=proxy.serve_forever,daemon=True).start()
        print(f'Videos: http://localhost:{args.port} · TensorBoard: http://localhost:6006',flush=True)
        try:
            gallery.serve_forever()
        finally:
            proxy.shutdown(); process.terminate()


if __name__=='__main__':
    main()
