import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT); _sys.path.insert(0, _ROOT)  # resolve cl/, vectors.json, hwpx_cracker

import numpy as np, pyopencl as cl, hashlib, json
ctx=cl.create_some_context(interactive=False); q=cl.CommandQueue(ctx)
src = open("cl/sha.cl").read()+open("cl/kdf.cl").read()+r"""
#define STRIDE 256
// PBKDF2 test: start_key(32) + salt in globals -> dk(dklen)
__kernel void t_pbkdf2(__global const uchar* sk,__global const uchar* salt,uint saltlen,
                       uint iters,uint dklen,__global uchar* out){
    uchar k[32]; for(int i=0;i<32;i++)k[i]=sk[i];
    uchar s[32]; for(uint i=0;i<saltlen;i++)s[i]=salt[i];
    uchar dk[40]; pbkdf2_hmac_sha1(k,32u,s,saltlen,iters,dk,dklen);
    for(uint i=0;i<dklen;i++)out[i]=dk[i];
}
// partial key: password bytes -> 16-byte AES key
__kernel void t_partial(__global const uchar* w,__global const uint* ln,__global uchar* out){
    int g=get_global_id(0); uchar buf[STRIDE]; uint L=ln[g];
    for(uint i=0;i<L;i++)buf[i]=w[g*STRIDE+i];
    uchar d[20]; sha1_hash(buf,L,d);
    uchar key[16]; derive_partial_key_from_pwdigest(d,key);
    for(int i=0;i<16;i++)out[g*16+i]=key[i];
}
"""
prg=cl.Program(ctx,src).build(); mf=cl.mem_flags
V=json.load(open("vectors.json"))

# --- PBKDF2 ---
ok=True
for name,d in V["document_pbkdf2"].items():
    sk=bytes.fromhex(d["start_key"]); salt=bytes.fromhex(d["salt"])
    skb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(sk,np.uint8))
    sab=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(salt,np.uint8))
    ob=cl.Buffer(ctx,mf.WRITE_ONLY,d["klen"])
    prg.t_pbkdf2(q,(1,),None,skb,sab,np.uint32(len(salt)),np.uint32(d["iters"]),np.uint32(d["klen"]),ob)
    r=np.empty(d["klen"],np.uint8); cl.enqueue_copy(q,r,ob); q.finish()
    g=r.tobytes().hex(); s= g==d["dk"]; ok&=s
    print(f"PBKDF2 {name:16s} {'OK' if s else 'FAIL  gpu='+g+' exp='+d['dk']}")

# --- partial key ---
pws=list(V["partial"].keys()); n=len(pws); STRIDE=256
w=np.zeros((n,STRIDE),np.uint8); ln=np.zeros(n,np.uint32)
for i,pw in enumerate(pws):
    b=pw.encode(); w[i,:len(b)]=np.frombuffer(b,np.uint8); ln[i]=len(b)
wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=w.ravel())
lb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ln)
ob=cl.Buffer(ctx,mf.WRITE_ONLY,n*16)
prg.t_partial(q,(n,),None,wb,lb,ob)
r=np.empty(n*16,np.uint8); cl.enqueue_copy(q,r,ob); q.finish()
for i,pw in enumerate(pws):
    g=r[i*16:(i+1)*16].tobytes().hex(); e=V["partial"][pw]["aes_key"]; s=g==e; ok&=s
    print(f"partkey {pw[:18]:18s} {'OK' if s else 'FAIL gpu='+g+' exp='+e}")
print("ALL","PASS" if ok else "FAIL")
