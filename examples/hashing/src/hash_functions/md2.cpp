#include <hash_functions/md2.hpp>
#include <internal/determine_kernel_config.hpp>

#include <cstring>
#include <tools/runtime_byte_array.hpp>

using namespace usm_smart_ptr;
using namespace hash;

struct md2_ctx {
    int len = 0;
    runtime_byte_array<16> data{};
    byte state[48]{};
    byte checksum[16]{};
};

/**************************** VARIABLES *****************************/


/*********************** FUNCTION DEFINITIONS ***********************/
template<typename T>
static inline void md2_transform(md2_ctx *ctx, const T &data) {
    constexpr byte consts[256]
            {41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
             19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188, 76,
             130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24, 138,
             23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251, 245, 142,
             187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63, 148, 194, 16,
             137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50, 39, 53, 62,
             204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165, 181, 209, 215,
             94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210, 150, 164, 125, 182,
             118, 252, 107, 226, 156, 116, 4, 241, 69, 157, 112, 89, 100, 113, 135,
             32, 134, 91, 207, 101, 230, 45, 168, 2, 27, 96, 37, 173, 174, 176,
             185, 246, 28, 70, 97, 105, 52, 64, 126, 15, 85, 71, 163, 35, 221,
             81, 175, 58, 195, 92, 249, 206, 186, 197, 234, 38, 44, 83, 13, 110,
             133, 40, 132, 9, 211, 223, 205, 244, 65, 129, 77, 82, 106, 220, 55,
             200, 108, 193, 171, 250, 36, 225, 123, 8, 12, 189, 177, 74, 120, 136,
             149, 139, 227, 99, 232, 109, 233, 203, 213, 254, 59, 0, 29, 57, 242,
             239, 183, 14, 102, 88, 208, 228, 166, 119, 114, 248, 235, 117, 75, 10,
             49, 68, 80, 180, 143, 237, 31, 26, 219, 153, 141, 51, 159, 17, 131,
             20};

#ifdef __NVPTX__
#pragma unroll
#endif
    for (int j = 0; j < 16; ++j) {
        ctx->state[j + 32] = (ctx->state[j + 16] = data[j]) ^ ctx->state[j];
    }

    dword t = 0;

#ifdef __NVPTX__
#pragma unroll
#endif
    for (dword j = 0; j < 18; ++j) {

#ifdef __NVPTX__
#pragma unroll
#endif
        for (unsigned char &k: ctx->state) {
            t = k ^= consts[t];
        }
        t = (t + j) & 0xFF;
    }

    t = ctx->checksum[15];

#ifdef __NVPTX__
#pragma unroll
#endif
    for (int j = 0; j < 16; ++j) {
        t = ctx->checksum[j] ^= consts[data[j] ^ t];
    }
}

static inline void md2_update(md2_ctx *ctx, const byte *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data.write(ctx->len, data[i]);
        ctx->len++;
        if (ctx->len == MD2_BLOCK_SIZE) {
            md2_transform(ctx, ctx->data);
            ctx->len = 0;
        }
    }
}

static inline void md2_final(md2_ctx *ctx, byte *hash) {
    int to_pad = (int) MD2_BLOCK_SIZE - ctx->len;
    if (to_pad > 0) {
#ifdef __NVPTX__
#pragma unroll
#endif
        for (int i = ctx->len; i < MD2_BLOCK_SIZE; ++i) {
            ctx->data.write(i, (byte) to_pad);
        }
    }
    md2_transform(ctx, ctx->data);
    md2_transform(ctx, ctx->checksum);
    memcpy(hash, ctx->state, MD2_BLOCK_SIZE);
}

static inline void kernel_md2_hash(const byte *indata, dword inlen, byte *outdata, dword n_batch, dword thread) {
    if (thread >= n_batch) {
        return;
    }
    const byte *in = indata + thread * inlen;
    byte *out = outdata + thread * MD2_BLOCK_SIZE;
    md2_ctx ctx{};
    md2_update(&ctx, in, inlen);
    md2_final(&ctx, out);
}

namespace hash::internal {

    sycl::event
    launch_md2_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch) {
        auto config = get_kernel_sizes(q, n_batch);
        return q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e);
            cgh.parallel_for<class md2_kernel>(
                    sycl::nd_range<1>(sycl::range<1>(config.block) * sycl::range<1>(config.wg_size), sycl::range<1>(config.wg_size)),
                    [=](sycl::nd_item<1> item) {
                        kernel_md2_hash(indata, inlen, outdata, n_batch, item.get_global_linear_id());
                    });
        });
    }


}

