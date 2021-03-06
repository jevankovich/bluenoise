OPT_FLAGS := -O2
CFLAGS = ${OPT_FLAGS} -Wall -Wextra -std=c99 -pedantic
SRCS := bluenoise.c pcg_basic.c
TARGET := bluenoise

${TARGET}: ${SRCS}
	${CC} ${CFLAGS} $^ -o $@

.PHONY: perf
perf: OPT_FLAGS += -fno-omit-frame-pointer
perf: ${TARGET}

.PHONY: fast
fast: OPT_FLAGS := -O3 -march=native -mtune=native
fast: ${TARGET}

.PHONY: debug
debug: OPT_FLAGS := -Og -ggdb3
debug: ${TARGET}

.PHONY: clean
clean:
	rm -f ${TARGET}