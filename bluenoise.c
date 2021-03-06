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
#include <limits.h>

#include "pcg_basic.h"

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

typedef struct {
    float *mem;
    size_t w, h; // Width and height
    size_t xs, ys; // x-stride and y-stride
} array;

typedef struct {
    size_t from;
    size_t to;
    size_t by;
} range;

char name_suffix[128];
char initial_mask_name[NAME_MAX + 1];
char mask_name[NAME_MAX + 1];
char noise_name[NAME_MAX + 1];

array new_array(size_t w, size_t h) {
    return (array){
        .mem = calloc(w * h, sizeof(float)),
        .w = w,
        .h = h,
        .xs = 1,
        .ys = w,
    };
}

static inline range from_to(size_t from, size_t to) {
    return (range){.from = from, .to = to, .by = 1};
}

static inline range from_to_by(size_t from, size_t to, size_t by) {
    return (range){.from = from, .to = to, .by = by};
}

void free_array(array arr) {
    free(arr.mem);
}

static inline float *at(array arr, size_t x, size_t y) {
    return &arr.mem[arr.xs * x + arr.ys * y];
}

static inline array slice(array arr, range x, range y) {
    size_t w = x.to - x.from / x.by;
    size_t h = y.to - y.from / y.by;

    assert(x.from <= x.to);
    assert(y.from <= y.to);

    assert(x.from <= arr.w);
    assert(x.to <= arr.w);

    assert(y.from <= arr.h);
    assert(y.to <= arr.h);

    return (array){
        .mem = at(arr, x.from, y.from),
        .w = w,
        .h = h,
        .xs = arr.xs * x.by,
        .ys = arr.ys * y.by,
    };
}

static inline array row(array arr, size_t y) {
    return (array){
        .mem = arr.mem + arr.ys * y,
        .w = arr.w,
        .h = 1,
        .xs = arr.xs,
        .ys = 0,
    };
}

static inline array col(array arr, size_t x) {
    return (array){
        .mem = arr.mem + arr.xs * x,
        .w = arr.h,
        .h = 1,
        .xs = arr.ys,
        .ys = 0,
    };
}

static inline size_t size(array arr) {
    return arr.w * arr.h;
}

static inline void zero_arr(array arr) {
    for (size_t y = 0; y < arr.h; y++) {
        for (size_t x = 0; x < arr.w; x++) {
            *at(arr, x, y) = 0.0f;
        }
    }
}

static inline void copy_arr(array to, array from) {
    assert(to.w <= from.w);
    assert(to.h <= from.h);

    for (size_t y = 0; y < to.h; y++) {
        for (size_t x = 0; x < to.w; x++) {
            *at(to, x, y) = *at(from, x, y);
        }
    }
}

void conv1d(array in, array kern, array out) {
    assert(in.h == 1);
    assert(kern.h == 1);
    assert(out.h == 1);
    assert(size(in) <= size(out) + size(kern) - 1);

    zero_arr(out);
    for (size_t i = 0; i < out.w; i++) {
        for (size_t k = 0; k < kern.w; k++) {
            *at(out, i, 0) += *at(kern, k, 0) * *at(in, i + kern.w - 1 - k, 0);
        }
    }
}

void conv2d(array in, array kern, array out, array work) {
    assert(kern.h == 1);
    assert(in.w == out.w + kern.w - 1);
    assert(in.h == out.h + kern.w - 1);
    assert(work.w == out.w);
    assert(work.h == in.h);

    for (size_t ri = 0; ri < work.h; ri++) {
        conv1d(row(in, ri), kern, row(work, ri));
    }

    for (size_t ci = 0; ci < out.w; ci++) {
        conv1d(col(work, ci), kern, col(out, ci));
    }
}

array repeat2d(array in, size_t dilation) {
    size_t mid = dilation / 2;
    array out = new_array(in.w + dilation - 1, in.h + dilation - 1);

    size_t w_over = in.w * ((out.w + in.w) / in.w);
    size_t h_over = in.h * ((out.h + in.h) / in.h);

    for (size_t y = 0; y < out.h; y++) {
        for (size_t x = 0; x < out.w; x++) {
            *at(out, x, y) = *at(in, (w_over + x - mid) % in.w, (h_over + y - mid) % in.h);
        }
    }

    return out;
}

void set_repeat(array arr, size_t dilation, size_t x, size_t y, float val) {
    size_t mid = dilation / 2;
    size_t w = arr.w - dilation + 1;
    size_t h = arr.h - dilation + 1;

    for (size_t row = (y + mid) % h; row < arr.h; row += h) {
        for (size_t col = (x + mid) % w; col < arr.w; col += w) {
            *at(arr, col, row) = val;
        }
    }
}

void conv2d_set(array in, array kern, array out, float *work, size_t x, size_t y, float val) {
    size_t mid = kern.w / 2;

    assert(in.w == out.w + kern.w - 1);
    assert(in.h == out.h + kern.w - 1);

    set_repeat(in, kern.w, x, y, val);

#ifdef PARANOID
    {
        array work_arr = (array){.mem = work, .w = out.w, .h = in.h, .xs = 1, .ys = out.w};
        conv2d(in, kern, out, work_arr);
    }
#else
    for (size_t row = (y + mid) % out.h; row < in.h; row += out.h) {
        for (size_t col = (x + mid) % out.w; col < in.w; col += out.w) {
            range out_x_range = from_to(col - MIN(kern.w - 1, col), MIN(out.w, col + 1));
            range out_y_range = from_to(row - MIN(kern.w - 1, row), MIN(out.h, row + 1));

            range in_x_range = from_to(out_x_range.from, out_x_range.to + kern.w - 1);
            range in_y_range = from_to(out_y_range.from, out_y_range.to + kern.w - 1);

            array in_slice = slice(in, in_x_range, in_y_range);
            array out_slice = slice(out, out_x_range, out_y_range);
            array work_arr = (array){.mem = work, .w = out_slice.w, .h = in_slice.h, .xs = 1, .ys = out_slice.w};

            conv2d(in_slice, kern, out_slice, work_arr);
        }
    }
#endif
}

array binomial(size_t len) {
    if (len < 1) {
        return (array){
            .mem = NULL,
            .w = 0,
            .h = 0,
            .xs = 0,
            .ys = 0,
        };
    }

    array out = new_array(len, 1);
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

    memcpy(out.mem, &back[1], sizeof(*out.mem) * len);
    free(back);
    free(front);
    return out;
}

void permute(float *arr, size_t len, uint32_t seed) {
    pcg32_srandom(seed, 0);

    // Fisher-Yates Shuffle
    for (size_t i = 0; i < len - 1; i++) {
        float tmp;
        size_t rand = pcg32_boundedrand(len - i) + i;
        tmp = arr[i];
        arr[i] = arr[rand];
        arr[rand] = tmp;
    }
}

static inline void little_endian(uint16_t x, uint8_t *out) {
    out[0] = (x >> 8) & 0xFF;
    out[1] = (x >> 0) & 0xFF;
}

void write_pgm(const char *fname, array data, uint16_t maxval) {
    FILE *f = fopen(fname, "wb");

    fprintf(f, "P5\n%zu %zu\n%"PRIu16"\n", data.w, data.h, maxval);

    if (maxval > 255) {
        uint8_t tmp[2];
        for (size_t y = 0; y < data.h; y++) {
            for (size_t x = 0; x < data.w; x++) {
                little_endian((uint16_t)(*at(data, x, y) * maxval), tmp);
                fwrite(tmp, sizeof(tmp), 1, f);
            }
        }
    } else {
        uint8_t tmp;
        for (size_t y = 0; y < data.h; y++) {
            for (size_t x = 0; x < data.w; x++) {
                tmp = (uint8_t)(*at(data, x, y) * maxval);
                fwrite(&tmp, sizeof(tmp), 1, f);
            }
        }
    }

    fclose(f);
}

void write_pbm(const char *fname, array data) {
    FILE *f = fopen(fname, "wb");

    fprintf(f, "P4\n%zu %zu\n", data.w, data.h);

    for (size_t row = 0; row < data.h; row++) {
        for (size_t col = 0; col < data.w; col += 8) {
            uint8_t tmp = 0;
            for (size_t c = 0; (c < 8) && ((col + c) < data.w); c++) {
                tmp |= (*at(data, col + c, row) > 0.5f ? 0 : 1) << (7 - c);
            }

            fwrite(&tmp, sizeof(tmp), 1, f);
        }
    }

    fclose(f);
}

void find_cluster(array mask, array conv, size_t *x, size_t *y) {
    assert(mask.w == conv.w);
    assert(mask.h == conv.h);

    float cmp = -INFINITY;

    *x = 0;
    *y = 0;

    for (size_t row = 0; row < conv.h; row++) {
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val > cmp && *at(mask, col, row) == 1.0f) {
                cmp = val;
                *x = col;
                *y = row;
            }
        }
    }
}

void find_void(array mask, array conv, size_t *x, size_t *y) {
    assert(mask.w == conv.w);
    assert(mask.h == conv.h);

    float cmp = INFINITY;

    *x = 0;
    *y = 0;

    for (size_t row = 0; row < conv.h; row++) {
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val < cmp && *at(mask, col, row) == 0.0f) {
                cmp = val;
                *x = col;
                *y = row;
            }
        }
    }
}

void scan_clusters(array mask, array conv, size_t *xs) {
    for (size_t row = 0; row < conv.h; row++) {
        float cmp = -INFINITY;
        xs[row] = 0;

        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val > cmp && *at(mask, col, row) == 1.0f) {
                cmp = val;
                xs[row] = col;
            }
        }
    }
}

void scan_voids(array mask, array conv, size_t *xs) {
    for (size_t row = 0; row < conv.h; row++) {
        float cmp = INFINITY;
        xs[row] = 0;

        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val < cmp && *at(mask, col, row) == 0.0f) {
                cmp = val;
                xs[row] = col;
            }
        }
    }
}

void clear_cluster(array mask, array kern, array conv, array work, size_t *x, size_t *y, size_t *cxs, size_t *vxs) {
    size_t mid = kern.w / 2;
    float cmp = -INFINITY;

    *x = 0;
    *y = 0;

#ifdef PARANOID
    for (size_t row = 0; row < conv.h; row++) {
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val > cmp && *at(mask, col + mid, row + mid) == 1.0f) {
                cmp = val;
                *x = col;
                *y = row;
            }
        }
    }

    conv2d_set(mask, kern, conv, work.mem, *x, *y, 0);
#else
    for (size_t row = 0; row < conv.h; row++) {
        size_t col = cxs[row];
        float val = *at(conv, col, row);

        if (val > cmp && *at(mask, col + mid, row + mid) == 1.0f) {
            cmp = val;
            *x = col;
            *y = row;
        }
    }

    conv2d_set(mask, kern, conv, work.mem, *x, *y, 0);

    for (size_t k = 0; k < kern.w; k++) {
        size_t row = (conv.h + *y + k - mid) % conv.h;
        
        cmp = -INFINITY;
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val > cmp && *at(mask, col + mid, row + mid) == 1.0f) {
                cmp = val;
                cxs[row] = col;
            }
        }

        cmp = INFINITY;
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val < cmp && *at(mask, col + mid, row + mid) == 0.0f) {
                cmp = val;
                vxs[row] = col;
            }
        }
    }
#endif
}

void fill_void(array mask, array kern, array conv, array work, size_t *x, size_t *y, size_t *cxs, size_t *vxs) {
    size_t mid = kern.w / 2;
    float cmp = INFINITY;

    *x = 0;
    *y = 0;

#ifdef PARANOID
    for (size_t row = 0; row < conv.h; row++) {
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val < cmp && *at(mask, col + mid, row + mid) == 0.0f) {
                cmp = val;
                *x = col;
                *y = row;
            }
        }
    }

    conv2d_set(mask, kern, conv, work.mem, *x, *y, 1);
#else
    for (size_t row = 0; row < conv.h; row++) {
        size_t col = vxs[row];
        float val = *at(conv, col, row);

        if (val < cmp && *at(mask, col + mid, row + mid) == 0.0f) {
            cmp = val;
            *x = col;
            *y = row;
        }
    }

    conv2d_set(mask, kern, conv, work.mem, *x, *y, 1);

    for (size_t k = 0; k < kern.w; k++) {
        size_t row = (conv.h + *y + k - mid) % conv.h;
        
        cmp = -INFINITY;
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val > cmp && *at(mask, col + mid, row + mid) == 1.0f) {
                cmp = val;
                cxs[row] = col;
            }
        }

        cmp = INFINITY;
        for (size_t col = 0; col < conv.w; col++) {
            float val = *at(conv, col, row);
            if (val < cmp && *at(mask, col + mid, row + mid) == 0.0f) {
                cmp = val;
                vxs[row] = col;
            }
        }
    }
#endif
}

void relax(array mask, array kern, array conv, array work, size_t *cxs, size_t *vxs) {
    size_t cx, cy;
    size_t vx, vy;

    size_t mid = kern.w / 2;

    array inset = slice(mask, from_to(mid, mid + conv.w), from_to(mid, mid + conv.h));

    conv2d(mask, kern, conv, work);
    scan_clusters(inset, conv, cxs);
    scan_voids(inset, conv, vxs);

    do {
        clear_cluster(mask, kern, conv, work, &cx, &cy, cxs, vxs);

        fill_void(mask, kern, conv, work, &vx, &vy, cxs, vxs);
    } while (cx != vx || cy != vy);
}

array blue_noise(array mask, array kern) {
    size_t w = mask.w;
    size_t h = mask.h;
    size_t pixels = w * h;
    size_t mid = kern.w / 2;

    size_t rank = 0;
    for (size_t row = 0; row < mask.h; row++) {
        for (size_t col = 0; col < mask.w; col++) {
            if (*at(mask, col, row)) {
                rank++;
            }
        }
    }

    mask = repeat2d(mask, kern.w);

    array backup = new_array(mask.w, mask.h);
    array inset_mask = slice(backup, from_to(mid, mid + w), from_to(mid, mid + h));

    array values = new_array(w, h);
    array conv = new_array(w, h);
    array work = new_array(w, mask.h);

    size_t *cxs = malloc(sizeof(*cxs) * h);
    size_t *vxs = malloc(sizeof(*vxs) * h);

    relax(mask, kern, conv, work, cxs, vxs);
    copy_arr(backup, mask);
    write_pbm(mask_name, inset_mask);

    for (size_t i = rank; i != 0; i--) {
        size_t cx, cy;
        clear_cluster(backup, kern, conv, work, &cx, &cy, cxs, vxs);
        *at(values, cx, cy) = (float)(i - 1) / (pixels - 1);
    }

    conv2d(mask, kern, conv, work);
    copy_arr(backup, mask);
    scan_clusters(inset_mask, conv, cxs);
    scan_voids(inset_mask, conv, vxs);

    for (size_t i = rank; i < pixels; i++) {
        size_t vx, vy;
        fill_void(mask, kern, conv, work, &vx, &vy, cxs, vxs);
        *at(values, vx, vy) = (float)i / (pixels - 1);
    }

    free_array(mask);
    free_array(backup);
    free_array(conv);
    free_array(work);

    free(cxs);
    free(vxs);

    return values;
}

void usage(FILE *f, char *prog) {
    fprintf(f, "Usage: %s [options] width height\n\n", prog);
    fprintf(f, "-n numerator\tThe numerator used to determine the fraction of pixels set\n"
               "\t\tin the initial mask. (Default: 1)\n");
    fprintf(f, "-d denominator\tThe denominator used to determine the fraction of pixels set\n"
               "\t\tin the initial mask. (Default: 10)\n");
    fprintf(f, "-c kernel size\tThe size of the kernel to use in filtering. The original\n"
               "\t\tpaper recommends a kernel with std-deviation of 1.5.\n"
               "\t\tThe std-deviation of the kernel is sqrt(n/4). (Default: 9)\n");
    fprintf(f, "-s seed\t\tThe seed to use for the RNG when generating the initial mask\n"
               "\t\t(Default: 0)\n");
}

int main(int argc, char **argv) {
    int c;
    char *end;
    long value;

    size_t width, height;
    uint32_t fracn = 1, fracd = 10;

    array kern;
    array mask;
    array image;

    uint32_t seed = 0;
    size_t conv_size = 9; // The paper recommends sigma = 1.5. 9 gets us to 1.5

    while ((c = getopt(argc, argv, "hn:d:c:s:")) != -1) {
        switch(c) {
        case 'n':
            value = strtol(optarg, &end, 0);
            if (*end) {
                fprintf(stderr, "Could not parse -n argument\n");
                continue;
            }
            if (value <= 0) {
                fprintf(stderr, "The argument to -n must be positive\n");
                continue;
            }
            fracn = value;
            break;
        case 'd':
            value = strtol(optarg, &end, 0);
            if (*end) {
                fprintf(stderr, "Could not parse -%c argument\n", c);
                continue;
            }
            if (value <= 0) {
                fprintf(stderr, "The argument to -%c must be positive\n", c);
                continue;
            }
            fracd = value;
            break;
        case 'c':
            value = strtol(optarg, &end, 0);
            if (*end) {
                fprintf(stderr, "Could not parse -%c argument\n", c);
                continue;
            }
            if (value <= 0) {
                fprintf(stderr, "The argument to -%c must be positive\n", c);
                continue;
            }
            conv_size = atol(optarg);
            break;
        case 's':
            value = strtol(optarg, &end, 0);
            if (*end) {
                fprintf(stderr, "Could not parse -%c argument\n", c);
                continue;
            }
            if (value < 0) {
                fprintf(stderr, "The argument to -%c must be non-negative\n", c);
                continue;
            }
            seed = atol(optarg);
            break;
        case 'h':
            usage(stdout, argv[0]);
            return 0;
        case '?':
            usage(stderr, argv[0]);
            return 1;
        default:
            abort();
        }
    }

    if (argc < optind + 1) {
        usage(stderr, argv[0]);
        return 1;
    }

    {
        value = strtol(argv[optind], &end, 0);
        if (*end) {
            fprintf(stderr, "Could not parse width\n");
            return 1;
        }
        if (value <= 0) {
            fprintf(stderr, "The width must be positive\n");
            return 1;
        }
        width = value;
    }

    {
        value = strtol(argv[optind], &end, 0);
        if (*end) {
            fprintf(stderr, "Could not parse height\n");
            return 1;
        }
        if (value <= 0) {
            fprintf(stderr, "The height must be positive\n");
            return 1;
        }
        height = value;
    }

    if (fracn * 2 >= fracd) {
        fprintf(stderr, "Given fraction is greater than 1/2\n");
        exit(EXIT_FAILURE);
    }

    // if (sqrt(fracn / fracd) <= 2 / conv_size)
    if (fracd >= fracn * conv_size * conv_size / 4) {
        fprintf(stderr, "Warning: it is expected that each pixel in the mask will have fewer than one neighbor\n");
    }

    snprintf(name_suffix, sizeof(name_suffix), "%zux%zu_n%"PRIu32"_d%"PRIu32"_s%"PRIu32, width, height, fracn, fracd, seed);
    snprintf(initial_mask_name, sizeof(initial_mask_name), "initial_mask_%s.pbm", name_suffix);
    snprintf(mask_name, sizeof(mask_name), "mask_%s.pbm", name_suffix);
    snprintf(noise_name, sizeof(noise_name), "noise_%s.pgm", name_suffix);

    kern = binomial(conv_size);

    mask = new_array(width, height);
    for (size_t i = 0; i < width * height; i++) {
        mask.mem[i] = (i * fracd) < (width * height * fracn);
    }

    permute(mask.mem, width * height, seed);

    write_pbm(initial_mask_name, mask);

    image = blue_noise(mask, kern);

    write_pgm(noise_name, image, UINT16_MAX);

    free_array(kern);
    free_array(mask);
    free_array(image);
}
