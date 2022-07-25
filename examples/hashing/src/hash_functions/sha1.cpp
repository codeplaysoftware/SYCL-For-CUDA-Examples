/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's SYCL-For-CUDA-Examples
 *
 *  sha1.cpp
 *
 *  Description:
 *    SHA1 hash function
 **************************************************************************/
#include <hash_functions/sha1.hpp>
#include <internal/determine_kernel_config.hpp>
#include <tools/intrinsics.hpp>

#include <cstring>


using namespace usm_smart_ptr;

struct sha1_ctx {
    byte data[64];
    dword datalen = 0;
    qword bitlen = 0;
    dword state[5]{0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xc3d2e1f0};
    dword k[4]{0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6};

};

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

/*********************** FUNCTION DEFINITIONS ***********************/
void sha1_transform(sha1_ctx *ctx, const byte *data) {
    dword a, b, c, d, e, t, m[80];

#ifdef __NVPTX__
#pragma unroll
#endif
    for (int i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = sbb::upsample(data[j], data[j + 1], data[j + 2], data[j + 3]);
    }


#ifdef __NVPTX__
#pragma unroll
#endif
    for (qword i = 16; i < 80; ++i) {
        m[i] = (m[i - 3] ^ m[i - 8] ^ m[i - 14] ^ m[i - 16]);
        m[i] = (m[i] << 1) | (m[i] >> 31);
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];

#ifdef __NVPTX__
#pragma unroll
#endif
    for (dword i = 0; i < 20; ++i) {
        t = ROTLEFT(a, 5) + ((b & c) ^ (~b & d)) + e + ctx->k[0] + m[i];
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = t;
    }
#ifdef __NVPTX__
#pragma unroll
#endif
    for (dword i = 20; i < 40; ++i) {
        t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[1] + m[i];
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = t;
    }

#ifdef __NVPTX__
#pragma unroll
#endif
    for (dword i = 40; i < 60; ++i) {
        t = ROTLEFT(a, 5) + ((b & c) ^ (b & d) ^ (c & d)) + e + ctx->k[2] + m[i];
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = t;
    }

#ifdef __NVPTX__
#pragma unroll
#endif
    for (dword i = 60; i < 80; ++i) {
        t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[3] + m[i];
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = t;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
}

void sha1_update(sha1_ctx *ctx, const byte *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            sha1_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

void sha1_final(sha1_ctx *ctx, byte *hash) {
    dword i = ctx->datalen;

    // Pad whatever data is left in the buffer.
    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56)
            ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
        sha1_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    sha1_transform(ctx, ctx->data);

    // Since this implementation uses little endian byte ordering and MD uses big endian,
    // reverse all the bytes when copying the final state to the output hash.
    for (i = 0; i < 4; ++i) {
        hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
    }
}

void kernel_sha1_hash(const byte *indata, dword inlen, byte *outdata, dword n_batch, dword thread) {
    if (thread >= n_batch) {
        return;
    }
    const byte *in = indata + thread * inlen;
    byte *out = outdata + thread * SHA1_BLOCK_SIZE;
    sha1_ctx ctx{};
    sha1_update(&ctx, in, inlen);
    sha1_final(&ctx, out);
}


namespace hash::internal {
    sycl::event launch_sha1_kernel(sycl::queue &q, sycl::event e, const device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch) {
        auto config = get_kernel_sizes(q, n_batch);
        return q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e);
            cgh.parallel_for<class sha1_kernel>(
                    sycl::nd_range<1>(sycl::range<1>(config.block) * sycl::range<1>(config.wg_size), sycl::range<1>(config.wg_size)),
                    [=](sycl::nd_item<1> item) {
                        kernel_sha1_hash(indata, inlen, outdata, n_batch, item.get_global_linear_id());
                    });
        });
    }

}
