
all:
	~/AFLplusplus/afl-clang-fast -O2 ./foo.c

fuzz: all
	AFL_SKIP_CPUFREQ=1 ~/AFLplusplus/afl-fuzz -p fast -d -i inputs -o outputs ./a.out @@

internal: all
	./a.out internal

