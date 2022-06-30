#include <hash_functions/blake2b.hpp>
#include <internal/determine_kernel_config.hpp>

#include <cstring>

#include <tools/usm_smart_ptr.hpp>

using namespace usm_smart_ptr;

static const qword GLOBAL_BLAKE2B_IVS[8]
        = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
           0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
           0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};


static inline void blake2b_init(blake2b_ctx *ctx, const byte *key, dword keylen, dword digestbitlen) {
    //memset(ctx, 0, sizeof(blake2b_ctx));
    memcpy(ctx->buff, key, keylen);
    ctx->keylen = keylen;

    ctx->digestlen = digestbitlen >> 3;
    ctx->pos = 0;
    ctx->t0 = 0;
    ctx->t1 = 0;
    ctx->f0 = 0;
    ctx->chain[0] = GLOBAL_BLAKE2B_IVS[0] ^ (static_cast<qword>(ctx->digestlen) | (ctx->keylen << 8) | 0x1010000);
    ctx->chain[1] = GLOBAL_BLAKE2B_IVS[1];
    ctx->chain[2] = GLOBAL_BLAKE2B_IVS[2];
    ctx->chain[3] = GLOBAL_BLAKE2B_IVS[3];
    ctx->chain[4] = GLOBAL_BLAKE2B_IVS[4];
    ctx->chain[5] = GLOBAL_BLAKE2B_IVS[5];
    ctx->chain[6] = GLOBAL_BLAKE2B_IVS[6];
    ctx->chain[7] = GLOBAL_BLAKE2B_IVS[7];

    ctx->pos = BLAKE2B_BLOCK_LENGTH;
}


static inline qword blake2b_leuint64(const byte *in) {
    qword a;
    memcpy(&a, in, 8);
    return a;
}

static inline qword blake2b_ROTR64(qword a, byte b) { return (a >> b) | (a << (64 - b)); }

//#define blake2b_ROTR64(a, b) ((a) >> (b) | ((a) << (64 - (b))))


struct blake2b_G_arg {
    blake2b_ctx *ctx;
    qword m1, m2;
    dword a, b, c, d;
};

static inline void blake2b_G(blake2b_G_arg arg) {
    qword *state = arg.ctx->state;
    state[arg.a] = state[arg.a] + state[arg.b] + arg.m1;
    state[arg.d] = blake2b_ROTR64(state[arg.d] ^ state[arg.a], 32);
    state[arg.c] = state[arg.c] + state[arg.d];
    state[arg.b] = blake2b_ROTR64(state[arg.b] ^ state[arg.c], 24);
    state[arg.a] = state[arg.a] + state[arg.b] + arg.m2;
    state[arg.d] = blake2b_ROTR64(state[arg.d] ^ state[arg.a], 16);
    state[arg.c] = state[arg.c] + state[arg.d];
    state[arg.b] = blake2b_ROTR64(state[arg.b] ^ state[arg.c], 63);
}

static inline void blake2b_init_state(blake2b_ctx *ctx) {
    static const qword ivs[8]
            = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
               0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
               0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};

    memcpy(ctx->state, ctx->chain, BLAKE2B_CHAIN_LENGTH);
#pragma unroll
    for (dword i = 0; i < 4; i++) {
        ctx->state[BLAKE2B_CHAIN_SIZE + i] = ivs[i];
    }

    ctx->state[12] = ctx->t0 ^ ivs[4];
    ctx->state[13] = ctx->t1 ^ ivs[5];
    ctx->state[14] = ctx->f0 ^ ivs[6];
    ctx->state[15] = ivs[7];
}

static inline void blake2b_compress(blake2b_ctx *ctx, const byte *in, dword inoffset) {
    blake2b_init_state(ctx);

    static const byte sigmas[12][16] =
            {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15},
             {14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3},
             {11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4},
             {7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8},
             {9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13},
             {2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9},
             {12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11},
             {13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10},
             {6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5},
             {10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0},
             {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15},
             {14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3}};

    qword m[16];
#pragma unroll
    for (dword j = 0; j < 16; j++) {
        m[j] = blake2b_leuint64(in + inoffset + (j << 3));
    }

#pragma unroll
    for (auto sigma: sigmas) {
        blake2b_G({ctx, m[sigma[0]], m[sigma[1]], 0, 4, 8, 12});
        blake2b_G({ctx, m[sigma[2]], m[sigma[3]], 1, 5, 9, 13});
        blake2b_G({ctx, m[sigma[4]], m[sigma[5]], 2, 6, 10, 14});
        blake2b_G({ctx, m[sigma[6]], m[sigma[7]], 3, 7, 11, 15});
        blake2b_G({ctx, m[sigma[8]], m[sigma[9]], 0, 5, 10, 15});
        blake2b_G({ctx, m[sigma[10]], m[sigma[11]], 1, 6, 11, 12});
        blake2b_G({ctx, m[sigma[12]], m[sigma[13]], 2, 7, 8, 13});
        blake2b_G({ctx, m[sigma[14]], m[sigma[15]], 3, 4, 9, 14});
    }

#pragma unroll
    for (dword offset = 0; offset < BLAKE2B_CHAIN_SIZE; offset++)
        ctx->chain[offset] = ctx->chain[offset] ^ ctx->state[offset] ^ ctx->state[offset + 8];
}

static inline void blake2b_update(blake2b_ctx *ctx, const byte *in, qword inlen) {
    if (inlen == 0)
        return;

    dword start = 0;
    int64_t in_index, block_index;

    if (ctx->pos) {
        start = BLAKE2B_BLOCK_LENGTH - ctx->pos;
        if (start < inlen) {
            memcpy(ctx->buff + ctx->pos, in, start);
            ctx->t0 += BLAKE2B_BLOCK_LENGTH;
            if (ctx->t0 == 0) ctx->t1++;
            blake2b_compress(ctx, ctx->buff, 0);
            ctx->pos = 0;
            memset(ctx->buff, 0, BLAKE2B_BLOCK_LENGTH);
        } else {
            memcpy(ctx->buff + ctx->pos, in, inlen);//read the whole *in
            ctx->pos += inlen;
            return;
        }
    }

    block_index = (int64_t) inlen - BLAKE2B_BLOCK_LENGTH;
    for (in_index = start; in_index < block_index; in_index += BLAKE2B_BLOCK_LENGTH) {
        ctx->t0 += BLAKE2B_BLOCK_LENGTH;
        if (ctx->t0 == 0)
            ctx->t1++;

        blake2b_compress(ctx, in, in_index);
    }

    memcpy(ctx->buff, in + in_index, inlen - (size_t) in_index);
    ctx->pos += (inlen - (size_t) in_index);
}

static inline void blake2b_final(blake2b_ctx *ctx, byte *out) {
    ctx->f0 = 0xFFFFFFFFFFFFFFFFL;
    ctx->t0 += ctx->pos;
    if (ctx->pos > 0 && ctx->t0 == 0)
        ctx->t1++;

    blake2b_compress(ctx, ctx->buff, 0);
    memset(ctx->buff, 0, BLAKE2B_BLOCK_LENGTH);
    memset(ctx->state, 0, BLAKE2B_STATE_LENGTH);

    int64_t i8;
    for (dword i = 0; i < BLAKE2B_CHAIN_SIZE && ((i8 = i * 8) < ctx->digestlen); i++) {
        byte *BYTEs = (byte *) (&ctx->chain[i]);
        if (i8 < ctx->digestlen - 8)
            memcpy(out + i8, BYTEs, 8);
        else
            memcpy(out + i8, BYTEs, static_cast<size_t>(ctx->digestlen - i8));
    }
}

static inline void kernel_blake2b_hash(const byte *indata, dword inlen, byte *outdata, dword n_batch, dword block_size, dword thread,
                                       const blake2b_ctx *ctx) {

    if (thread >= n_batch) {
        return;
    }
    const byte *in = indata + thread * inlen;
    byte *out = outdata + thread * block_size;
    auto local_ctx = *ctx;
    //if not precomputed CTX, call cuda_blake2b_init() with key
    blake2b_update(&local_ctx, in, inlen);
    blake2b_final(&local_ctx, out);
}


namespace hash::internal {

    usm_shared_ptr<blake2b_ctx, alloc::device> get_blake2b_ctx(sycl::queue &q, const byte *key, dword keylen, dword n_outbit) {
        auto ctxt_device = usm_shared_ptr<blake2b_ctx, alloc::device>(1, q);
        blake2b_ctx ctx = {};
        blake2b_init(&ctx, key, keylen, n_outbit);
        q.memcpy(ctxt_device.raw(), &ctx, sizeof(ctx)).wait();
        return ctxt_device;
    }


    sycl::event
    launch_blake2b_kernel(sycl::queue &item, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit, const byte *,
                          dword, const device_accessible_ptr<blake2b_ctx> ctx) {
        const dword block_size = n_outbit >> 3;
        //  assert(keylen <= 128); // we must define keylen at host
        auto config = get_kernel_sizes(item, n_batch);
        return item.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e);
            cgh.parallel_for<class blake2b_kernel>(
                    sycl::nd_range<1>(sycl::range<1>(config.block) * sycl::range<1>(config.wg_size), sycl::range<1>(config.wg_size)),
                    [=](sycl::nd_item<1> item) {
                        kernel_blake2b_hash(indata, inlen, outdata, n_batch, block_size, item.get_global_linear_id(), ctx);
                    });
        });
    }


    sycl::event
    launch_blake2b_kernel(sycl::queue &item, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit, const byte *key,
                          dword keylen) {
        auto ptr = get_blake2b_ctx(item, key, keylen, n_outbit);
        launch_blake2b_kernel(item, std::move(e), indata, outdata, inlen, n_batch, n_outbit, key, keylen, ptr.get()).wait();
        return sycl::event{};
    }

}
