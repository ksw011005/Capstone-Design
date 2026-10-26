// =====================================================================
//  sha.cl  --  SHA-1 / SHA-256 device functions for the HWPX cracker
//  Big-endian, bounded-length messages held in private/global uchar bufs.
//  Validated against Python hashlib inside the project test harness.
// =====================================================================
#ifndef HWPX_SHA_CL
#define HWPX_SHA_CL

#define ROTL32(x,n) rotate((uint)(x),(uint)(n))
#define ROTR32(x,n) rotate((uint)(x),(uint)(32u-(n)))

// ------------------------------------------------------------------ SHA-1
__constant uint SHA1_IV[5] = {0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};

// One 64-byte block, big-endian load. state updated in place.
inline void sha1_transform(uint st[5], const uchar blk[64]) {
    uint w[80];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = ((uint)blk[i*4]<<24)|((uint)blk[i*4+1]<<16)|((uint)blk[i*4+2]<<8)|((uint)blk[i*4+3]);
    for (int i = 16; i < 80; i++)
        w[i] = ROTL32(w[i-3]^w[i-8]^w[i-14]^w[i-16], 1);

    uint a=st[0],b=st[1],c=st[2],d=st[3],e=st[4];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint f,k;
        if      (i < 20) { f=(b&c)|((~b)&d);        k=0x5A827999u; }
        else if (i < 40) { f=b^c^d;                 k=0x6ED9EBA1u; }
        else if (i < 60) { f=(b&c)|(b&d)|(c&d);     k=0x8F1BBCDCu; }
        else             { f=b^c^d;                 k=0xCA62C1D6u; }
        uint t = ROTL32(a,5)+f+e+k+w[i];
        e=d; d=c; c=ROTL32(b,30); b=a; a=t;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e;
}

// Full SHA-1 over msg[0..len) with len <= SHA_MAX_MSG (private buffer).
// out20 receives 20 big-endian bytes.
inline void sha1_hash(const uchar *msg, uint len, uchar out20[20]) {
    uint st[5];
    #pragma unroll
    for (int i=0;i<5;i++) st[i]=SHA1_IV[i];

    uchar blk[64];
    uint off = 0;
    while (len - off >= 64) {
        #pragma unroll
        for (int i=0;i<64;i++) blk[i]=msg[off+i];
        sha1_transform(st, blk);
        off += 64;
    }
    uint rem = len - off;                 // 0..63
    for (uint i=0;i<rem;i++) blk[i]=msg[off+i];
    blk[rem]=0x80;
    for (uint i=rem+1;i<64;i++) blk[i]=0;
    if (rem >= 56) {                      // no room for length -> extra block
        sha1_transform(st, blk);
        for (int i=0;i<64;i++) blk[i]=0;
    }
    ulong bits = (ulong)len*8;
    for (int i=0;i<8;i++) blk[63-i]=(uchar)(bits>>(8*i));
    sha1_transform(st, blk);

    #pragma unroll
    for (int i=0;i<5;i++){ out20[i*4]=(uchar)(st[i]>>24); out20[i*4+1]=(uchar)(st[i]>>16);
                           out20[i*4+2]=(uchar)(st[i]>>8); out20[i*4+3]=(uchar)st[i]; }
}

// ---------------------------------------------------------------- SHA-256
__constant uint SHA256_IV[8] = {
    0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
__constant uint SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

inline void sha256_transform(uint st[8], const uchar blk[64]) {
    uint w[64];
    #pragma unroll
    for (int i=0;i<16;i++)
        w[i]=((uint)blk[i*4]<<24)|((uint)blk[i*4+1]<<16)|((uint)blk[i*4+2]<<8)|((uint)blk[i*4+3]);
    for (int i=16;i<64;i++){
        uint s0=ROTR32(w[i-15],7)^ROTR32(w[i-15],18)^(w[i-15]>>3);
        uint s1=ROTR32(w[i-2],17)^ROTR32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    #pragma unroll
    for (int i=0;i<64;i++){
        uint S1=ROTR32(e,6)^ROTR32(e,11)^ROTR32(e,25);
        uint ch=(e&f)^((~e)&g);
        uint t1=h+S1+ch+SHA256_K[i]+w[i];
        uint S0=ROTR32(a,2)^ROTR32(a,13)^ROTR32(a,22);
        uint maj=(a&b)^(a&c)^(b&c);
        uint t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    st[0]+=a;st[1]+=b;st[2]+=c;st[3]+=d;st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}

// SHA-256 over a message that may live in global memory (checksum path).
inline void sha256_hash_g(__global const uchar *msg, uint len, uchar out32[32]) {
    uint st[8];
    #pragma unroll
    for (int i=0;i<8;i++) st[i]=SHA256_IV[i];
    uchar blk[64];
    uint off=0;
    while (len-off>=64){ for(int i=0;i<64;i++) blk[i]=msg[off+i]; sha256_transform(st,blk); off+=64; }
    uint rem=len-off;
    for(uint i=0;i<rem;i++) blk[i]=msg[off+i];
    blk[rem]=0x80; for(uint i=rem+1;i<64;i++) blk[i]=0;
    if(rem>=56){ sha256_transform(st,blk); for(int i=0;i<64;i++) blk[i]=0; }
    ulong bits=(ulong)len*8; for(int i=0;i<8;i++) blk[63-i]=(uchar)(bits>>(8*i));
    sha256_transform(st,blk);
    #pragma unroll
    for(int i=0;i<8;i++){ out32[i*4]=(uchar)(st[i]>>24); out32[i*4+1]=(uchar)(st[i]>>16);
                          out32[i*4+2]=(uchar)(st[i]>>8); out32[i*4+3]=(uchar)st[i]; }
}

// SHA-256 over a private buffer (start-key path: SHA256(password bytes)).
inline void sha256_hash_p(const uchar *msg, uint len, uchar out32[32]) {
    uint st[8];
    #pragma unroll
    for (int i=0;i<8;i++) st[i]=SHA256_IV[i];
    uchar blk[64];
    uint off=0;
    while (len-off>=64){ for(int i=0;i<64;i++) blk[i]=msg[off+i]; sha256_transform(st,blk); off+=64; }
    uint rem=len-off;
    for(uint i=0;i<rem;i++) blk[i]=msg[off+i];
    blk[rem]=0x80; for(uint i=rem+1;i<64;i++) blk[i]=0;
    if(rem>=56){ sha256_transform(st,blk); for(int i=0;i<64;i++) blk[i]=0; }
    ulong bits=(ulong)len*8; for(int i=0;i<8;i++) blk[63-i]=(uchar)(bits>>(8*i));
    sha256_transform(st,blk);
    #pragma unroll
    for(int i=0;i<8;i++){ out32[i*4]=(uchar)(st[i]>>24); out32[i*4+1]=(uchar)(st[i]>>16);
                          out32[i*4+2]=(uchar)(st[i]>>8); out32[i*4+3]=(uchar)st[i]; }
}

#endif // HWPX_SHA_CL
