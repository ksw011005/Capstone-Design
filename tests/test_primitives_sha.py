import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT); _sys.path.insert(0, _ROOT)  # resolve cl/, vectors.json, hwpx_cracker

import numpy as np, pyopencl as cl, hashlib, os
ctx = cl.create_some_context(interactive=False); q = cl.CommandQueue(ctx)
src = open("cl/sha.cl").read() + r"""
#define STRIDE 256
__kernel void t_sha1(__global const uchar* w,__global const uint* ln,__global uchar* o){
    int g=get_global_id(0); uchar buf[STRIDE]; uint L=ln[g];
    for(uint i=0;i<L;i++) buf[i]=w[g*STRIDE+i];
    uchar d[20]; sha1_hash(buf,L,d);
    for(int i=0;i<20;i++) o[g*20+i]=d[i];
}
__kernel void t_sha256(__global const uchar* w,__global const uint* ln,__global uchar* o){
    int g=get_global_id(0); uchar buf[STRIDE]; uint L=ln[g];
    for(uint i=0;i<L;i++) buf[i]=w[g*STRIDE+i];
    uchar d[32]; sha256_hash_p(buf,L,d);
    for(int i=0;i<32;i++) o[g*32+i]=d[i];
}
"""
prg = cl.Program(ctx, src).build()
STRIDE=256
msgs = [b"", b"abc", b"12345", "한글".encode(), b"a"*55, b"a"*56, b"a"*63, b"a"*64,
        b"a"*100, ("가나다라마바사"*5).encode(), os.urandom(200)]
n=len(msgs)
w=np.zeros((n,STRIDE),np.uint8); ln=np.zeros(n,np.uint32)
for i,m in enumerate(msgs): w[i,:len(m)]=np.frombuffer(m,np.uint8); ln[i]=len(m)
mf=cl.mem_flags
wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=w.ravel())
lb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ln)
o1=cl.Buffer(ctx,mf.WRITE_ONLY,n*20); o2=cl.Buffer(ctx,mf.WRITE_ONLY,n*32)
prg.t_sha1(q,(n,),None,wb,lb,o1); prg.t_sha256(q,(n,),None,wb,lb,o2)
r1=np.empty(n*20,np.uint8); r2=np.empty(n*32,np.uint8)
cl.enqueue_copy(q,r1,o1); cl.enqueue_copy(q,r2,o2); q.finish()
ok=True
for i,m in enumerate(msgs):
    g1=r1[i*20:(i+1)*20].tobytes().hex(); e1=hashlib.sha1(m).hexdigest()
    g2=r2[i*32:(i+1)*32].tobytes().hex(); e2=hashlib.sha256(m).hexdigest()
    s1="OK" if g1==e1 else "FAIL"; s2="OK" if g2==e2 else "FAIL"
    ok &= (g1==e1 and g2==e2)
    print(f"len={len(m):3d} sha1={s1} sha256={s2}")
print("ALL", "PASS" if ok else "FAIL")
