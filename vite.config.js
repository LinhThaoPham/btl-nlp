import {defineConfig} from 'vite';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {existsSync, mkdirSync, createWriteStream, renameSync, unlinkSync} from 'node:fs';
import {resolve, basename} from 'node:path';
import {pipeline} from 'node:stream/promises';
import {Transform} from 'node:stream';

function localNlp() {
  let child, serial=0;
  const pending=new Map();
  function start() {
    if(child)return;
    const local=resolve(process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
    if(!existsSync(local))throw new Error('Chưa cài NLP. Xem README: tạo .venv và cài requirements-nlp.txt.');
    child=spawn(local,['-u','server/nlp_service.py'],{cwd:process.cwd(),windowsHide:true,stdio:['pipe','pipe','pipe']});
    createInterface({input:child.stdout}).on('line',line=>{
      try {const data=JSON.parse(line),task=pending.get(data.id);if(task){clearTimeout(task.timer);pending.delete(data.id);data.error?task.reject(new Error(data.error)):task.resolve(data.result);}}catch{}
    });
    child.stderr.on('data',b=>process.stderr.write(b));
    const failed=()=>{child=null;for(const task of pending.values()){clearTimeout(task.timer);task.reject(new Error('Dịch vụ NLP đã dừng. Kiểm tra cài đặt Python.'));}pending.clear();};
    child.on('error',failed);child.on('exit',failed);
  }
  function request(data) {
    start();return new Promise((resolve,reject)=>{const id=++serial,timer=setTimeout(()=>{pending.delete(id);reject(new Error('Xử lý NLP quá 5 phút.'));},300000);pending.set(id,{resolve,reject,timer});child.stdin.write(JSON.stringify({...data,id})+'\n');});
  }
  function configure(server) {
    server.httpServer?.on('close',()=>child?.kill());
    server.middlewares.use('/api/nlp',async(req,res)=>{
      res.setHeader('Content-Type','application/json; charset=utf-8');
      const reply=(status,data)=>{res.statusCode=status;res.end(JSON.stringify(data));};
      // This API is local-only, even when Vite is intentionally bound to a LAN address.
      const ip=req.socket.remoteAddress;
      if(!['127.0.0.1','::1','::ffff:127.0.0.1'].includes(ip))return reply(403,{error:'NLP chỉ cho phép truy cập từ máy này.'});
      const origin=req.headers.origin;
      if(origin && new URL(origin).host!==req.headers.host)return reply(403,{error:'Nguồn yêu cầu không hợp lệ.'});
      try {
        if(req.method==='GET')return reply(200,await request({action:'status'}));
        if(req.method!=='POST')return reply(405,{error:'Chỉ hỗ trợ GET và POST.'});
        if(req.url?.startsWith('/upload?')){
          const name=basename(new URL(req.url,'http://localhost').searchParams.get('name')||'');
          if(!/^[\p{L}\p{N}_. -]+\.(bin|vec|txt)$/u.test(name))throw new Error('Chọn tệp Word2Vec .bin/.vec/.txt (đã giải nén).');
          mkdirSync(resolve('models'),{recursive:true});
          const target=resolve('models',`${Date.now()}-${name}`),temp=target+'.part';let bytes=0;
          try {await pipeline(req,new Transform({transform(chunk,encoding,callback){bytes+=chunk.length;callback(bytes>2*1024**3?new Error('Tối đa 2 GB.'):null,chunk);}}),createWriteStream(temp));renameSync(temp,target);}
          catch(error){if(existsSync(temp))unlinkSync(temp);throw error;}
          return reply(200,{name:basename(target)});
        }
        let body='';for await(const chunk of req){body+=chunk;if(Buffer.byteLength(body)>4*1024**2)throw new Error('Yêu cầu quá lớn.');}
        return reply(200,await request(JSON.parse(body)));
      }catch(error){reply(400,{error:error.message});}
    });
  }
  return {name:'local-vietnamese-nlp',configureServer:configure,configurePreviewServer:configure,closeBundle(){child?.kill();}};
}
export default defineConfig({plugins:[localNlp()],server:{host:'127.0.0.1'},preview:{host:'127.0.0.1'}});
