import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT); _sys.path.insert(0, _ROOT)  # resolve cl/, vectors.json, hwpx_cracker

import numpy as np, pyopencl as cl, os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
ctx=cl.create_some_context(interactive=False); q=cl.CommandQueue(ctx)
src=open("cl/aes.cl").read()+r"""
__kernel void t_enc(__global const uchar* key,uint Nk,__global const uchar* in,__global uchar* out){
    uchar rk[AES_MAXRK]; uchar k[32]; for(uint i=0;i<Nk*4u;i++)k[i]=key[i];
    aes_expand(k,Nk,rk); uchar s[16]; for(int i=0;i<16;i++)s[i]=in[i];
    aes_encrypt_block(s,rk,Nk+6u); for(int i=0;i<16;i++)out[i]=s[i];
}
__kernel void t_dec(__global const uchar* key,uint Nk,__global const uchar* in,__global uchar* out){
    uchar rk[AES_MAXRK]; uchar k[32]; for(uint i=0;i<Nk*4u;i++)k[i]=key[i];
    aes_expand(k,Nk,rk); uchar s[16]; for(int i=0;i<16;i++)s[i]=in[i];
    aes_decrypt_block(s,rk,Nk+6u); for(int i=0;i<16;i++)out[i]=s[i];
}
// CBC decrypt of NB blocks (chaining), ciphertext + iv in globals
__kernel void t_cbc_dec(__global const uchar* key,uint Nk,__global const uchar* iv,
                        __global const uchar* ct,uint nb,__global uchar* pt){
    uchar rk[AES_MAXRK]; uchar k[32]; for(uint i=0;i<Nk*4u;i++)k[i]=key[i];
    aes_expand(k,Nk,rk); uint Nr=Nk+6u;
    uchar prev[16]; for(int i=0;i<16;i++)prev[i]=iv[i];
    for(uint b=0;b<nb;b++){
        uchar s[16],c[16]; for(int i=0;i<16;i++){c[i]=ct[b*16+i]; s[i]=c[i];}
        aes_decrypt_block(s,rk,Nr);
        for(int i=0;i<16;i++) pt[b*16+i]=s[i]^prev[i];
        for(int i=0;i<16;i++) prev[i]=c[i];
    }
}
"""
prg=cl.Program(ctx,src).build(); mf=cl.mem_flags
def run(kname,key,blk):
    Nk=len(key)//4
    kb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(key,np.uint8))
    ib=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(blk,np.uint8))
    ob=cl.Buffer(ctx,mf.WRITE_ONLY,16)
    getattr(prg,kname)(q,(1,),None,kb,np.uint32(Nk),ib,ob)
    r=np.empty(16,np.uint8); cl.enqueue_copy(q,r,ob); q.finish(); return r.tobytes()
ok=True
for Nk,klen in [(4,16),(6,24),(8,32)]:
    for _ in range(3):
        key=os.urandom(klen); pt=os.urandom(16)
        enc=Cipher(algorithms.AES(key),modes.ECB()).encryptor()
        ct_ref=enc.update(pt)+enc.finalize()
        ct_gpu=run("t_enc",key,pt)
        dec=Cipher(algorithms.AES(key),modes.ECB()).decryptor()
        pt_ref=dec.update(ct_ref)+dec.finalize()
        pt_gpu=run("t_dec",key,ct_ref)
        e=(ct_gpu==ct_ref); d=(pt_gpu==pt_ref); ok&=e and d
    print(f"AES-{klen*8} ecb enc={'OK' if e else 'FAIL'} dec={'OK' if d else 'FAIL'}")
# CBC decrypt 64 blocks AES-256
key=os.urandom(32); iv=os.urandom(16); pt=os.urandom(64*16)
enc=Cipher(algorithms.AES(key),modes.CBC(iv)).encryptor(); ct=enc.update(pt)+enc.finalize()
kb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(key,np.uint8))
ivb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(iv,np.uint8))
ctb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(ct,np.uint8))
ptb=cl.Buffer(ctx,mf.WRITE_ONLY,len(pt))
prg.t_cbc_dec(q,(1,),None,kb,np.uint32(8),ivb,ctb,np.uint32(64),ptb)
r=np.empty(len(pt),np.uint8); cl.enqueue_copy(q,r,ptb); q.finish()
cbc_ok=(r.tobytes()==pt); ok&=cbc_ok
print("AES-256 CBC-decrypt 64 blocks:", "OK" if cbc_ok else "FAIL")
print("ALL", "PASS" if ok else "FAIL")
