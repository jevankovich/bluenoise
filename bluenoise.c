#define _XOPEN_SOURCE 600

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <unistd.h>

#include "pcg_basic.h"

float *binomial(size_t len) {

    if (len < 1) {
        return NULL;
    }

    float *out = malloc(sizeof(*out) * len);
    float *back = calloc(sizeof(*back), len + 1);
    float *front = calloc(sizeof(*front), len + 1);

    // Row 1 = [0 1 ]
    back[1] = 1;

    for (size_t row = 2; row <= len; row++) {
        float *tmp;

        for (size_t i = 1; i <= row; i++) {
            front[i] = (back[i] + back[i - 1]) / 2.0f;
        }

        tmp = back;
        back = front;
        front = tmp;
    }

    memcpy(out, &back[1], sizeof(*out) * len);
    return out;
}

void square_kernel(const float *in, float *out, size_t len) {
    for (size_t i = 0; i < len; i++) {
        for (size_t j = 0; j < len; j++) {
            out[i * len + j] = in[i] * in[j];
        }
    }
}

void white_noise(bool *buf, size_t len, uint32_t seed, uint32_t n, uint32_t m) {
    pcg32_srandom(seed, 0);

    for (size_t i = 0; i < len; i++) {
        buf[i] = pcg32_boundedrand(m) < n;
    }
}

void crand(bool *buf, size_t len, uint32_t seed, uint32_t n, uint32_t m) {
    srand(seed);

    for (size_t i = 0; i < len; i++) {
        buf[i] = (rand() % m) < n;
    }
}

void trivial(bool *buf, size_t len, uint32_t seed, uint32_t n, uint32_t m) {
    if (seed != 0) {
        pcg32_srandom(seed, 0);
        seed = pcg32_random();
    }

    for (size_t i = 0; i < len; i++) {
        buf[i] = ((i + seed) % m) < n;
    }
}

static inline void little_endian(uint16_t x, uint8_t *out) {
    out[0] = (x >> 8) & 0xFF;
    out[1] = (x >> 0) & 0xFF;
}

void write_pgm(const char *fname, size_t width, size_t height, uint16_t maxval, const float *data) {
    FILE *f = fopen(fname, "wb");
    size_t pixels = width * height;

    fprintf(f, "P5\n%zu %zu\n%"PRIu16"\n", width, height, maxval);

    if (maxval > 255) {
        uint8_t tmp[2];
        for (size_t i = 0; i < pixels; i++) {
            little_endian((uint16_t)(data[i] * maxval), tmp);
            fwrite(tmp, sizeof(tmp), 1, f);
        }
    } else {
        uint8_t tmp;
        for (size_t i = 0; i < pixels; i++) {
            tmp = (uint8_t)(data[i] * maxval);
            fwrite(&tmp, sizeof(tmp), 1, f);
        }
    }

    fclose(f);
}

void write_pbm(const char *fname, size_t width, size_t height, const bool *data) {
    FILE *f = fopen(fname, "wb");

    fprintf(f, "P4\n%zu %zu\n", width, height);

    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col += 8) {
            uint8_t tmp = 0;
            for (size_t c = 0; (c < 8) && ((col + c) < width); c++) {
                tmp |= (data[row * width + col + c] ? 0 : 1) << (7 - c);
            }

            fwrite(&tmp, sizeof(tmp), 1, f);
        }
    }

    fclose(f);
}

// Maybe make a version of this that's optimized for non-strided vs strided.
// i.e. invert the i and k loops
void conv1d(const float *in, const float *kern, float *out, size_t len, size_t stride, size_t k_len) {
    size_t mid = k_len / 2;

    assert(k_len < len);

    for (size_t i = 0; i < len; i++) {
        float y = 0;

        for (size_t k = 0; k < k_len; k++) {
            size_t ii;
            if (k > i + mid) {
                // i - k + mid < 0
                ii = len + i + mid - k;
            } else if (k + len <= i + mid) {
                // i - k + mid >= len
                ii = i + mid - k - len;
            } else {
                ii = i + mid - k;
            }

            y += kern[k] * in[ii * stride];
        }

        out[i * stride] = y;
    }
}

void conv2d(const bool *in, const float *kern, float *restrict out, float *restrict work, size_t w, size_t h, size_t k_len) {
    // size_t mid = k_len / 2;

    // // The range analysis below depends on these conditions
    // assert(k_len < w);
    // assert(k_len < h);

    for (size_t row = 0; row < h; row++) {
        for (size_t col = 0; col < w; col++) {
            work[row * w + col] = 0;
            out[row * w + col] = in[row * w + col];
        }
    }

    // First convolve along the rows
    for (size_t row = 0; row < h; row++) {
        conv1d(&out[row * w], kern, &work[row * w], w, 1, k_len);
    }

    // Then the columns
    for (size_t col = 0; col < h; col++) {
        conv1d(&work[col], kern, &out[col], h, w, k_len);
    }
}

size_t find_cluster(const bool *mask, const float *conv, size_t len) {
    float cmp = 0.0f;
    size_t ret = 0;

    for (size_t i = 0; i < len; i++) {
        if (conv[i] > cmp && mask[i]) {
            ret = i;
            cmp = conv[i];
        }
    }

    return ret;
}

size_t find_void(const bool *mask, const float *conv, size_t len) {
    float cmp = 1.0f;
    size_t ret = 0;

    for (size_t i = 0; i < len; i++) {
        if (conv[i] < cmp && !mask[i]) {
            ret = i;
            cmp = conv[i];
        }
    }

    return ret;
}

void relax(bool *mask, const float *k, float *restrict conv, float *restrict work, size_t w, size_t h, size_t k_len) {
    // Ideally, instead of doing a full convolution at each update, we could just
    // update the affected part of the image
    size_t ic, iv; // cluster and void index
    size_t iters = 0;

    // Initial convolution
    conv2d(mask, k, conv, work, w, h, k_len);

    do {
        ic = find_cluster(mask, conv, w * h);
        mask[ic] = false;
        conv2d(mask, k, conv, work, w, h, k_len);
        // conv2d_set(mask, k, conv, w, h, k_len, ic, false);

        iv = find_void(mask, conv, w * h);
        mask[iv] = true;
        conv2d(mask, k, conv, work, w, h, k_len);
        // conv2d_set(mask, k, conv, w, h, k_len, iv, true);

        iters++;
    } while (iv != ic);

    printf("%zu iterations to relax\n", iters);
}

float *blue_noise(bool *mask, const float *k, size_t w, size_t h, size_t k_len) {
    // Ideally, instead of doing a full convolution at each update, we could just
    // update the affected part of the image
    // Other idea: Maintain the region around an image so that convolutions cannot wrap
    size_t pixels = w * h;
    size_t rank;

    bool *backup;
    float *values;
    float *conv;
    float *work;

    backup = malloc(sizeof(*backup) * pixels);
    values = malloc(sizeof(*values) * pixels);
    conv = malloc(sizeof(*conv) * pixels);
    work = malloc(sizeof(*work) * pixels);

    // Relax the mask
    relax(mask, k, conv, work, w, h, k_len);

    rank = 0;
    for (size_t i = 0; i < pixels; i++) {
        rank += mask[i];
    }

    // Copy the mask and clear all clusters
    // Assign the value based on the number of remaining 1s in the mask
    memcpy(backup, mask, sizeof(*backup) * pixels);

    size_t i = rank;
    do {
        i--;
        size_t ic = find_cluster(backup, conv, pixels);
        backup[ic] = false;
        conv2d(backup, k, conv, work, w, h, k_len);
        // conv2d_set(backup, k, conv, w, h, k_len, ic, false);
        values[ic] = (float)i / (pixels - 1);
    } while (i != 0);

    // Copy the mask and set all voids
    // Assign the value based on the number of 1s in the mask
    memcpy(backup, mask, sizeof(*backup) * pixels);

    for (size_t i = rank; i < pixels; i++) {
        size_t iv = find_void(backup, conv, pixels);
        backup[iv] = true;
        conv2d(backup, k, conv, work, w, h, k_len);
        // conv2d_set(backup, k, conv, w, h, k_len, iv, true);
        values[iv] = (float)i / (pixels - 1);
    }

    free(backup);
    free(conv);
    free(work);

    return values;
}

int main(int argc, char **argv) {
    size_t width, height, pixels;
    uint32_t fracn = 1, fracd = 10;

    float *kern;
    bool *mask;
    float *image;

    uint32_t seed = 0;
    size_t conv_size = 11;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s width height [fracd | fracn fracd] [conv_size] [seed]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    width = atol(argv[1]);
    height = atol(argv[2]);
    pixels = width * height;

    if (argc > 4) {
        fracn = atol(argv[3]);
        fracd = atol(argv[4]);
    } else if (argc == 4) {
        fracd = atol(argv[3]);
    }

    if (fracn * 2 >= fracd) {
        fprintf(stderr, "Given fraction is greater than 1/2\n");
        exit(EXIT_FAILURE);
    }

    if (argc > 5) {
        conv_size = atol(argv[5]) | 1;
    }
    if (argc > 6) {
        seed = atol(argv[6]);
    }

    // if (sqrt(fracn / fracd) <= 2 / conv_size)
    if (fracd >= fracn * conv_size * conv_size / 4) {
        fprintf(stderr, "Warning: it is expected that each pixel in the mask will have fewer than one neighbor\n");
    }

    kern = binomial(conv_size);

    mask = malloc(sizeof(*mask) * pixels);
    white_noise(mask, pixels, seed, fracn, fracd);

    write_pbm("init_mask.pbm", width, height, mask);

    image = blue_noise(mask, kern, width, height, conv_size);

    write_pbm("relax_mask.pbm", width, height, mask);
    write_pgm("noise.pgm", width, height, UINT16_MAX, image);

    free(kern);
    free(mask);
    free(image);
}
