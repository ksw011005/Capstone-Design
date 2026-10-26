// =====================================================================
//  hwpx_kernels.cl
//  GPU verifiers for the three HWPX protection profiles, in two input
//  modes each:
//     *_dict : candidate bytes are supplied by the host (wordlist / rules)
//     *_mask : candidates are generated on the device from a Hashcat-style
//              mask (mixed-radix decode of the linear index), so even the
//              SHA-1-only distribution profile is not host-bound. Mask
//              tokens are byte strings, so Korean/UTF-8 multibyte charsets
//              work natively.
//
//  All candidates are already byte-encoded by the host (UTF-8 etc.); no
//  Unicode logic runs on the device.
//
//  Build defines: PW_STRIDE (bytes/slot, default 64),
//                 PART_MAXBITS (CFB1 prefix bits, default 512).
//  Match: g_found starts at INT_MAX; a hit does atomic_min(g_found, gid)
//  where gid is the index within the current dispatch.
// =====================================================================

#include "sha.cl"
#include "aes.cl"
#include "kdf.cl"

#ifndef PW_STRIDE
#define PW_STRIDE 64
#endif
#ifndef PART_MAXBITS
#define PART_MAXBITS 1024      // >= the 528-bit partial prefix; purely a loop bound
#endif

// ============================ verifier cores =========================

inline bool verify_dist(const uchar *pw, uint L,
                        __global const uchar *verifiers, uint nver){
    uchar d[20];
    sha1_hash(pw, L, d);
    for(uint v=0; v<nver; v++){
        bool eq=true;
        #pragma unroll
        for(int i=0;i<20;i++) if(d[i]!=verifiers[v*20+i]){ eq=false; break; }
        if(eq) return true;
    }
    return false;
}

inline bool verify_partial(const uchar *pw, uint L,
                           __global const uchar *ct_prefix,
                           __global const uchar *exp_prefix, uint nbits){
    uchar d[20];
    sha1_hash(pw, L, d);
    uchar key[16];
    derive_partial_key_from_pwdigest(d, key);

    uchar rk[AES_MAXRK];
    aes_expand(key, 4u, rk);
    uint Nr = 10u;

    uchar reg[16];
    #pragma unroll
    for(int i=0;i<16;i++) reg[i]=0;

    uint bits = nbits < PART_MAXBITS ? nbits : PART_MAXBITS;
    for(uint bit=0; bit<bits; bit++){
        uint byteidx=bit>>3, bidx=7u-(bit&7u);
        uchar ct_bit =(ct_prefix[byteidx] >>bidx)&1u;
        uchar exp_bit=(exp_prefix[byteidx]>>bidx)&1u;
        uchar blk[16];
        #pragma unroll
        for(int i=0;i<16;i++) blk[i]=reg[i];
        aes_encrypt_block(blk, rk, Nr);
        uchar ks_bit=(blk[0]>>7)&1u;
        if((uchar)(ct_bit^ks_bit)!=exp_bit) return false;
        uchar carry=ct_bit;
        for(int j=15;j>=0;j--){ uchar nb=(uchar)((reg[j]<<1)|carry); carry=(reg[j]>>7)&1u; reg[j]=nb; }
    }
    return true;
}

inline bool verify_document(const uchar *pw, uint L,
                            __global const uchar *salt, uint saltlen,
                            uint iters, uint key_size, uint start_algo, uint start_size,
                            __global const uchar *iv,
                            __global const uchar *ct, uint nb, uint hash_len,
                            __global const uchar *checksum){
    uchar sk[32];
    if(start_algo==1u){ uchar d[20]; sha1_hash(pw,L,d); for(uint i=0;i<start_size&&i<20u;i++) sk[i]=d[i]; }
    else              { uchar d[32]; sha256_hash_p(pw,L,d); for(uint i=0;i<start_size&&i<32u;i++) sk[i]=d[i]; }
    uint avail = (start_algo==1u) ? 20u : 32u;      // bytes actually produced
    uint skl = start_size < avail ? start_size : avail;   // == reference digest[:start_size]

    uchar salt_p[32];
    for(uint i=0;i<saltlen;i++) salt_p[i]=salt[i];
    uchar aeskey[40];
    pbkdf2_hmac_sha1(sk, skl, salt_p, saltlen, iters, aeskey, key_size);

    uchar rk[AES_MAXRK];
    uint Nk=key_size/4u;
    aes_expand(aeskey, Nk, rk);
    uint Nr=Nk+6u;

    uchar pt[1024];
    uchar prev[16];
    for(int i=0;i<16;i++) prev[i]=iv[i];
    for(uint b=0;b<nb;b++){
        uchar s[16],c[16];
        for(int i=0;i<16;i++){ c[i]=ct[b*16+i]; s[i]=c[i]; }
        aes_decrypt_block(s, rk, Nr);
        for(int i=0;i<16;i++) pt[b*16+i]=(uchar)(s[i]^prev[i]);
        for(int i=0;i<16;i++) prev[i]=c[i];
    }
    uchar got[32];
    sha256_hash_p(pt, hash_len, got);
    #pragma unroll
    for(int i=0;i<32;i++) if(got[i]!=checksum[i]) return false;
    return true;
}

// ============================ mask decoder ===========================
// Reconstruct candidate bytes for linear index `lin`. Enumeration order
// matches Python itertools.product: the leftmost position varies slowest.
inline uint mask_decode(ulong lin, uint P,
                        __global const uint  *pos_count,
                        __global const ulong *pos_weight,
                        __global const uint  *pos_base,
                        __global const uint  *tok_off,
                        __global const uint  *tok_len,
                        __global const uchar *cset,
                        uchar *pw){
    uint L=0;
    for(uint i=0;i<P;i++){
        uint t=(uint)((lin / pos_weight[i]) % (ulong)pos_count[i]);
        uint k=pos_base[i]+t;
        uint off=tok_off[k], len=tok_len[k];
        for(uint j=0;j<len && L<PW_STRIDE;j++) pw[L++]=cset[off+j];
    }
    return L;
}

// ============================ dictionary kernels =====================

inline void load_pw(__global const uchar *words, uint gid, uint len, uchar *out){
    for(uint i=0;i<len;i++) out[i]=words[gid*PW_STRIDE+i];
}

__kernel void dist_dict(__global const uchar *words, __global const uint *lens, uint n,
                        __global const uchar *ver, uint nver, volatile __global int *g_found){
    uint gid=get_global_id(0); if(gid>=n) return;
    uchar pw[PW_STRIDE]; uint L=lens[gid]; load_pw(words,gid,L,pw);
    if(verify_dist(pw,L,ver,nver)) atomic_min(g_found,(int)gid);
}

__kernel void partial_dict(__global const uchar *words, __global const uint *lens, uint n,
                           __global const uchar *ct, __global const uchar *exp, uint nbits,
                           volatile __global int *g_found){
    uint gid=get_global_id(0); if(gid>=n) return;
    uchar pw[PW_STRIDE]; uint L=lens[gid]; load_pw(words,gid,L,pw);
    if(verify_partial(pw,L,ct,exp,nbits)) atomic_min(g_found,(int)gid);
}

__kernel void document_dict(__global const uchar *words, __global const uint *lens, uint n,
                            __global const uchar *salt, uint saltlen, uint iters, uint key_size,
                            uint start_algo, uint start_size, __global const uchar *iv,
                            __global const uchar *ct, uint nb, uint hash_len,
                            __global const uchar *checksum, volatile __global int *g_found){
    uint gid=get_global_id(0); if(gid>=n) return;
    uchar pw[PW_STRIDE]; uint L=lens[gid]; load_pw(words,gid,L,pw);
    if(verify_document(pw,L,salt,saltlen,iters,key_size,start_algo,start_size,
                       iv,ct,nb,hash_len,checksum)) atomic_min(g_found,(int)gid);
}

// ============================ mask kernels ===========================
// Common mask args are threaded through; batch_start offsets the linear
// index so keyspaces > 2^31 can be split across dispatches.

#define MASK_ARGS \
    uint P, __global const uint *pos_count, __global const ulong *pos_weight, \
    __global const uint *pos_base, __global const uint *tok_off, \
    __global const uint *tok_len, __global const uchar *cset, \
    ulong batch_start, uint batch_n

__kernel void dist_mask(MASK_ARGS,
                        __global const uchar *ver, uint nver, volatile __global int *g_found){
    uint gid=get_global_id(0); if(gid>=batch_n) return;
    uchar pw[PW_STRIDE];
    uint L=mask_decode(batch_start+gid,P,pos_count,pos_weight,pos_base,tok_off,tok_len,cset,pw);
    if(verify_dist(pw,L,ver,nver)) atomic_min(g_found,(int)gid);
}

__kernel void partial_mask(MASK_ARGS,
                           __global const uchar *ct, __global const uchar *exp, uint nbits,
                           volatile __global int *g_found){
    uint gid=get_global_id(0); if(gid>=batch_n) return;
    uchar pw[PW_STRIDE];
    uint L=mask_decode(batch_start+gid,P,pos_count,pos_weight,pos_base,tok_off,tok_len,cset,pw);
    if(verify_partial(pw,L,ct,exp,nbits)) atomic_min(g_found,(int)gid);
}

__kernel void document_mask(MASK_ARGS,
                            __global const uchar *salt, uint saltlen, uint iters, uint key_size,
                            uint start_algo, uint start_size, __global const uchar *iv,
                            __global const uchar *ct, uint nb, uint hash_len,
                            __global const uchar *checksum, volatile __global int *g_found){
    uint gid=get_global_id(0); if(gid>=batch_n) return;
    uchar pw[PW_STRIDE];
    uint L=mask_decode(batch_start+gid,P,pos_count,pos_weight,pos_base,tok_off,tok_len,cset,pw);
    if(verify_document(pw,L,salt,saltlen,iters,key_size,start_algo,start_size,
                       iv,ct,nb,hash_len,checksum)) atomic_min(g_found,(int)gid);
}
