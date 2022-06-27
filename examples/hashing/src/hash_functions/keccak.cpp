#include "hash_functions/keccak.hpp"
#include "internal/determine_kernel_config.hpp"

using namespace usm_smart_ptr;

struct keccak_ctx_t {
    qword bits_in_queue = 0;
    qword state[KECCAK_STATE_SIZE]{};
    byte q[KECCAK_Q_SIZE]{};
};

static inline qword keccak_leuint64(const void *in) {
    qword a;
    memcpy(&a, in, 8);
    return a;
}


template<qword rate_bits>
static inline void keccak_extract(keccak_ctx_t *ctx) {
    constexpr qword len = rate_bits >> 6;
#pragma unroll len
    for (qword i = 0; i < len; i++) {
        qword a = ctx->state[i];
        memcpy(ctx->q + (i * sizeof(qword)), &a, sizeof(qword));
    }
}

static inline qword keccak_ROTL64(qword a, qword b) {
    return (a << b) | (a >> (64 - b));
}
//#define keccak_ROTL64(a, b) ((a) << (b)) | ((a) >> (64 - (b)))

static inline void keccak_permutations(keccak_ctx_t *ctx) {
    static constexpr std::array<qword, KECCAK_ROUND> consts =
            {0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
             0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
             0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
             0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
             0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
             0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
             0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
             0x8000000000008080, 0x0000000080000001, 0x8000000080008008};

    qword *A = ctx->state;
    qword *a00 = A, *a01 = A + 1, *a02 = A + 2, *a03 = A + 3, *a04 = A + 4;
    qword *a05 = A + 5, *a06 = A + 6, *a07 = A + 7, *a08 = A + 8, *a09 = A + 9;
    qword *a10 = A + 10, *a11 = A + 11, *a12 = A + 12, *a13 = A + 13, *a14 = A + 14;
    qword *a15 = A + 15, *a16 = A + 16, *a17 = A + 17, *a18 = A + 18, *a19 = A + 19;
    qword *a20 = A + 20, *a21 = A + 21, *a22 = A + 22, *a23 = A + 23, *a24 = A + 24;

    for (qword i: consts) {
        /* Theta */
        qword c0 = *a00 ^ *a05 ^ *a10 ^ *a15 ^ *a20;
        qword c1 = *a01 ^ *a06 ^ *a11 ^ *a16 ^ *a21;
        qword c2 = *a02 ^ *a07 ^ *a12 ^ *a17 ^ *a22;
        qword c3 = *a03 ^ *a08 ^ *a13 ^ *a18 ^ *a23;
        qword c4 = *a04 ^ *a09 ^ *a14 ^ *a19 ^ *a24;
        qword d1 = keccak_ROTL64(c1, 1) ^ c4;
        qword d2 = keccak_ROTL64(c2, 1) ^ c0;
        qword d3 = keccak_ROTL64(c3, 1) ^ c1;
        qword d4 = keccak_ROTL64(c4, 1) ^ c2;
        qword d0 = keccak_ROTL64(c0, 1) ^ c3;
        *a00 ^= d1;
        *a05 ^= d1;
        *a10 ^= d1;
        *a15 ^= d1;
        *a20 ^= d1;
        *a01 ^= d2;
        *a06 ^= d2;
        *a11 ^= d2;
        *a16 ^= d2;
        *a21 ^= d2;
        *a02 ^= d3;
        *a07 ^= d3;
        *a12 ^= d3;
        *a17 ^= d3;
        *a22 ^= d3;
        *a03 ^= d4;
        *a08 ^= d4;
        *a13 ^= d4;
        *a18 ^= d4;
        *a23 ^= d4;
        *a04 ^= d0;
        *a09 ^= d0;
        *a14 ^= d0;
        *a19 ^= d0;
        *a24 ^= d0;
        /* Rho pi */
        c1 = keccak_ROTL64(*a01, 1);
        *a01 = keccak_ROTL64(*a06, 44);
        *a06 = keccak_ROTL64(*a09, 20);
        *a09 = keccak_ROTL64(*a22, 61);
        *a22 = keccak_ROTL64(*a14, 39);
        *a14 = keccak_ROTL64(*a20, 18);
        *a20 = keccak_ROTL64(*a02, 62);
        *a02 = keccak_ROTL64(*a12, 43);
        *a12 = keccak_ROTL64(*a13, 25);
        *a13 = keccak_ROTL64(*a19, 8);
        *a19 = keccak_ROTL64(*a23, 56);
        *a23 = keccak_ROTL64(*a15, 41);
        *a15 = keccak_ROTL64(*a04, 27);
        *a04 = keccak_ROTL64(*a24, 14);
        *a24 = keccak_ROTL64(*a21, 2);
        *a21 = keccak_ROTL64(*a08, 55);
        *a08 = keccak_ROTL64(*a16, 45);
        *a16 = keccak_ROTL64(*a05, 36);
        *a05 = keccak_ROTL64(*a03, 28);
        *a03 = keccak_ROTL64(*a18, 21);
        *a18 = keccak_ROTL64(*a17, 15);
        *a17 = keccak_ROTL64(*a11, 10);
        *a11 = keccak_ROTL64(*a07, 6);
        *a07 = keccak_ROTL64(*a10, 3);
        *a10 = c1;

        /* Chi */
        c0 = *a00 ^ (~*a01 & *a02);
        c1 = *a01 ^ (~*a02 & *a03);
        *a02 ^= ~*a03 & *a04;
        *a03 ^= ~*a04 & *a00;
        *a04 ^= ~*a00 & *a01;
        *a00 = c0;
        *a01 = c1;

        c0 = *a05 ^ (~*a06 & *a07);
        c1 = *a06 ^ (~*a07 & *a08);
        *a07 ^= ~*a08 & *a09;
        *a08 ^= ~*a09 & *a05;
        *a09 ^= ~*a05 & *a06;
        *a05 = c0;
        *a06 = c1;

        c0 = *a10 ^ (~*a11 & *a12);
        c1 = *a11 ^ (~*a12 & *a13);
        *a12 ^= ~*a13 & *a14;
        *a13 ^= ~*a14 & *a10;
        *a14 ^= ~*a10 & *a11;
        *a10 = c0;
        *a11 = c1;

        c0 = *a15 ^ (~*a16 & *a17);
        c1 = *a16 ^ (~*a17 & *a18);
        *a17 ^= ~*a18 & *a19;
        *a18 ^= ~*a19 & *a15;
        *a19 ^= ~*a15 & *a16;
        *a15 = c0;
        *a16 = c1;

        c0 = *a20 ^ (~*a21 & *a22);
        c1 = *a21 ^ (~*a22 & *a23);
        *a22 ^= ~*a23 & *a24;
        *a23 ^= ~*a24 & *a20;
        *a24 ^= ~*a20 & *a21;
        *a20 = c0;
        *a21 = c1;

        /* Iota */
        *a00 ^= i;
    }
}


template<qword absorb_round>
static inline void keccak_absorb(keccak_ctx_t *ctx, const byte *in) {
#pragma unroll
    for (uint offset = 0, i = 0; i < absorb_round; ++i) {
        ctx->state[i] ^= keccak_leuint64(in + offset);
        offset += 8;
    }
    keccak_permutations(ctx);
}

template<qword digest_bit_len>
static inline void keccak_pad(keccak_ctx_t *ctx) {
    constexpr dword rate_bits = 1600 - ((digest_bit_len) << 1);
    constexpr dword absorb_round = rate_bits >> 6;

    ctx->q[ctx->bits_in_queue >> 3] |= (1L << (ctx->bits_in_queue & 7));

    if (++(ctx->bits_in_queue) == rate_bits) {
        keccak_absorb<absorb_round>(ctx, ctx->q);
        ctx->bits_in_queue = 0;
    }

    {
        qword full = ctx->bits_in_queue >> 6;
        qword partial = ctx->bits_in_queue & 63;

        qword offset = 0;
        for (dword i = 0; i < full; ++i) {
            ctx->state[i] ^= keccak_leuint64(ctx->q + offset);
            offset += 8;
        }

        if (partial > 0) {
            qword mask = (1L << partial) - 1;
            ctx->state[full] ^= keccak_leuint64(ctx->q + offset) & mask;
        }
    }

    ctx->state[(rate_bits - 1) >> 6] ^= 9223372036854775808ULL;/* 1 << 63 */
    keccak_permutations(ctx);
    keccak_extract<rate_bits>(ctx);
    ctx->bits_in_queue = rate_bits;
}

template<qword digest_bit_len>
static inline void keccak_update(keccak_ctx_t *ctx, const byte *in, qword inlen) {
    constexpr dword rate_bits = 1600 - ((digest_bit_len) << 1);
    constexpr dword absorb_round = rate_bits >> 6;
    constexpr qword rate_BYTEs = rate_bits >> 3;


    qword BYTEs = ctx->bits_in_queue >> 3;
    int64_t count = 0;
    while (count < (int64_t) inlen) {
        if (BYTEs == 0 && count <= ((int64_t) (inlen - rate_BYTEs))) {
            do {
                keccak_absorb<absorb_round>(ctx, in + count);
                count += rate_BYTEs;
            } while (count <= ((int64_t) (inlen - rate_BYTEs)));
        } else {
            qword partial = sycl::min(rate_BYTEs - BYTEs, inlen - (qword) count);
            memcpy(ctx->q + BYTEs, in + count, partial);

            BYTEs += partial;
            count += (int64_t) partial;

            if (BYTEs == rate_BYTEs) {
                keccak_absorb<absorb_round>(ctx, ctx->q);
                BYTEs = 0;
            }
        }
    }
    ctx->bits_in_queue = BYTEs << 3;
}

template<qword digest_bit_len>
static inline void keccak_final(bool is_sha3, keccak_ctx_t *ctx, byte *out) {
    constexpr dword rate_bits = 1600 - ((digest_bit_len) << 1);
    constexpr dword absorb_round = rate_bits >> 6;
    constexpr qword rate_BYTEs = rate_bits >> 3;


    if (is_sha3) {
        int mask = (1 << 2) - 1;
        ctx->q[ctx->bits_in_queue >> 3] = (byte) (0x02 & mask);
        ctx->bits_in_queue += 2;
    }

    keccak_pad<digest_bit_len>(ctx);
    qword i = 0;

    while (i < digest_bit_len) {
        if (ctx->bits_in_queue == 0) {
            keccak_permutations(ctx);
            keccak_extract<rate_bits>(ctx);
            ctx->bits_in_queue = rate_bits;
        }

        qword partial_block = sycl::min(ctx->bits_in_queue, digest_bit_len - i);
        memcpy(out + (i >> 3), ctx->q + (rate_BYTEs - (ctx->bits_in_queue >> 3)), partial_block >> 3);
        ctx->bits_in_queue -= partial_block;
        i += partial_block;
    }
}

template<qword digest_bit_len>
static inline void kernel_keccak_hash(bool is_sha3, const byte *indata, dword inlen, byte *outdata, dword n_batch, dword thread) {
    if (thread >= n_batch) {
        return;
    }
    const byte *in = indata + thread * inlen;
    byte *out = outdata + thread * (digest_bit_len >> 3);
    keccak_ctx_t ctx{};
    keccak_update<digest_bit_len>(&ctx, in, inlen);
    keccak_final<digest_bit_len>(is_sha3, &ctx, out);
}

namespace hash::internal {

    template<dword n_outbit_>
    sycl::event
    launch_keccak_kernel_template(bool is_sha3, sycl::queue &item, sycl::event e, const device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch) {
        auto config = get_kernel_sizes(item, n_batch);
        return item.submit([&](sycl::handler &cgh) {
            cgh.depends_on(std::move(e));
            cgh.parallel_for<keccak_kernel<n_outbit_>>(
                    sycl::nd_range<1>(sycl::range<1>(config.block) * sycl::range<1>(config.wg_size), sycl::range<1>(config.wg_size)),
                    [=](sycl::nd_item<1> item) {
                        kernel_keccak_hash<n_outbit_>(is_sha3, indata, inlen, outdata, n_batch, item.get_global_linear_id());
                    });
        });
    }

    sycl::event
    launch_keccak_kernel(bool is_sha3, sycl::queue &item, sycl::event e, const device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit) {
        if (n_outbit == 128) {
            return launch_keccak_kernel_template<128>(is_sha3, item, std::move(e), indata, outdata, inlen, n_batch);
        } else if (n_outbit == 224) {
            return launch_keccak_kernel_template<224>(is_sha3, item, std::move(e), indata, outdata, inlen, n_batch);
        } else if (n_outbit == 256) {
            return launch_keccak_kernel_template<256>(is_sha3, item, std::move(e), indata, outdata, inlen, n_batch);
        } else if (n_outbit == 288) {
            return launch_keccak_kernel_template<288>(is_sha3, item, std::move(e), indata, outdata, inlen, n_batch);
        } else if (n_outbit == 384) {
            return launch_keccak_kernel_template<384>(is_sha3, item, std::move(e), indata, outdata, inlen, n_batch);
        } else if (n_outbit == 512) {
            return launch_keccak_kernel_template<512>(is_sha3, item, std::move(e), indata, outdata, inlen, n_batch);
        } else {
            abort();
        }
    }

}