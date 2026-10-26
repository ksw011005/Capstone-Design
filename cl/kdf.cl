// =====================================================================
//  kdf.cl  --  PBKDF2-HMAC-SHA1 (document profile) and the selected-
//              character key derivation (Hancom obfuscation + buggy SHA-1).
//  Requires sha.cl.
// =====================================================================
#ifndef HWPX_KDF_CL
#define HWPX_KDF_CL

inline void sha1_state_to_bytes(const uint st[5], uchar out[20]){
    #pragma unroll
    for(int i=0;i<5;i++){ out[i*4]=(uchar)(st[i]>>24); out[i*4+1]=(uchar)(st[i]>>16);
                          out[i*4+2]=(uchar)(st[i]>>8); out[i*4+3]=(uchar)st[i]; }
}

// Precompute HMAC inner/outer states for key length <= 64 (start key = 32B).
inline void hmac_sha1_init(const uchar *key, uint keylen, uint istate[5], uint ostate[5]){
    uchar ip[64], op[64];
    for(uint i=0;i<64;i++){ uchar kb = (i<keylen)?key[i]:(uchar)0; ip[i]=kb^0x36; op[i]=kb^0x5c; }
    #pragma unroll
    for(int i=0;i<5;i++){ istate[i]=SHA1_IV[i]; ostate[i]=SHA1_IV[i]; }
    sha1_transform(istate, ip);
    sha1_transform(ostate, op);
}

// HMAC-SHA1 for a single short message (msglen <= 55) using precomputed states.
inline void hmac_sha1_msg(const uint istate[5], const uint ostate[5],
                          const uchar *msg, uint msglen, uchar out20[20]){
    uint st[5]; uchar blk[64];
    #pragma unroll
    for(int i=0;i<5;i++) st[i]=istate[i];
    for(uint i=0;i<msglen;i++) blk[i]=msg[i];
    blk[msglen]=0x80; for(uint i=msglen+1;i<64;i++) blk[i]=0;
    ulong bits=(ulong)(64u+msglen)*8u;
    for(int i=0;i<8;i++) blk[63-i]=(uchar)(bits>>(8*i));
    sha1_transform(st, blk);
    uchar inner[20]; sha1_state_to_bytes(st, inner);

    #pragma unroll
    for(int i=0;i<5;i++) st[i]=ostate[i];
    for(int i=0;i<20;i++) blk[i]=inner[i];
    blk[20]=0x80; for(int i=21;i<64;i++) blk[i]=0;
    ulong bits2=(ulong)(64u+20u)*8u;
    for(int i=0;i<8;i++) blk[63-i]=(uchar)(bits2>>(8*i));
    sha1_transform(st, blk);
    sha1_state_to_bytes(st, out20);
}

// PBKDF2-HMAC-SHA1. pass = start key (<=64B). dklen <= 40 here (key_size 16/24/32).
inline void pbkdf2_hmac_sha1(const uchar *pass, uint passlen,
                             const uchar *salt, uint saltlen,
                             uint iters, uchar *out, uint dklen){
    uint istate[5], ostate[5];
    hmac_sha1_init(pass, passlen, istate, ostate);
    uint nblk = (dklen + 19u)/20u;
    for(uint b=1; b<=nblk; b++){
        uchar msg[60];
        for(uint i=0;i<saltlen;i++) msg[i]=salt[i];
        msg[saltlen+0]=(uchar)(b>>24); msg[saltlen+1]=(uchar)(b>>16);
        msg[saltlen+2]=(uchar)(b>>8);  msg[saltlen+3]=(uchar)b;
        uchar u[20], t[20];
        hmac_sha1_msg(istate, ostate, msg, saltlen+4u, u);
        #pragma unroll
        for(int i=0;i<20;i++) t[i]=u[i];
        for(uint it=1; it<iters; it++){
            hmac_sha1_msg(istate, ostate, u, 20u, u);
            #pragma unroll
            for(int i=0;i<20;i++) t[i]^=u[i];
        }
        uint off=(b-1u)*20u;
        for(uint i=0;i<20u && off+i<dklen;i++) out[off+i]=t[i];
    }
}

// ---- selected-character protection key derivation ----
__constant uchar HEXU[16] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};

// SHA1(pw) already computed as 20-byte digest -> 40 uppercase hex chars.
inline void digest_to_hex40(const uchar d[20], uchar hex[40]){
    #pragma unroll
    for(int i=0;i<20;i++){ hex[i*2]=HEXU[(d[i]>>4)&0xF]; hex[i*2+1]=HEXU[d[i]&0xF]; }
}

// Hancom obfuscation of the 40-char ASCII hex, then the "buggy" 2nd SHA-1
// (obfuscated 80B repeated loop_count=2 -> 160B), first 16B = AES-128 key.
inline void derive_partial_key_from_pwdigest(const uchar pwd_sha1[20], uchar key16[16]){
    uchar hex[40];
    digest_to_hex40(pwd_sha1, hex);

    uchar obf[80];
    uchar al;
    for(int i=0;i<40;i++){
        al = (i==0) ? (uchar)0xEC : obf[i*2-1];
        al = (uchar)((al<<1)|(al>>7));          // ROL 1
        uchar code = hex[i];                     // ASCII hex: high byte 0, low byte nonzero
        obf[i*2]   = al;                         // high_byte==0  -> al
        obf[i*2+1] = code;                       // low_byte = char code
    }
    // loop_count = ((80-1)>>6)+1 = 2  => 160 bytes = obf || obf
    uchar buggy[160];
    #pragma unroll
    for(int i=0;i<80;i++){ buggy[i]=obf[i]; buggy[80+i]=obf[i]; }

    uchar d[20];
    sha1_hash(buggy, 160u, d);
    #pragma unroll
    for(int i=0;i<16;i++) key16[i]=d[i];
}

#endif // HWPX_KDF_CL
