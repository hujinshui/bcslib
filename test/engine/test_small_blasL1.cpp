/*
 * @file test_small_blasL1.cpp
 *
 * Unit testing of small matrix BLAS (Level 1)
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/engine/small_blasL1.h>
#include <cstdio>

using namespace bcs;

template<typename T>
void L1_init_vec(int n, T *v, T b)
{
	for (int i = 0; i < n; ++i) v[i] = i & 1 ? T(i+b) : T(-(i+b));
}

template<typename T>
void L1_prn_vec(const char *name, int n, const T *x, const int inc = 1)
{
	std::printf("%s: ", name);
	for (int i = 0; i < n; ++i) std::printf("%6g ", x[i * inc]);
	std::printf("\n");
}



/************************************************
 *
 *  asum
 *
 ************************************************/

template<typename T>
bool asum_check(int n, const T *x, int incx, const T r)
{
	T s(0);
	for (int i = 0; i < n; ++i) s += std::fabs(x[i * incx]);

	if (s == r)
	{
		return true;
	}
	else
	{
		std::printf("asum-dump [n = %d, incx = %d]:\n", n, incx);
		L1_prn_vec("  x", n, x, incx);
		std::printf(" r = %g  (s = %g)\n\n", r, s);

		return false;
	}
}


template<typename T, int N>
void asum_test()
{
	T x[4 * N];

	L1_init_vec(4 * N, x, T(1));

	T r1 = engine::small_asum<T, N>::eval(x, 1);
	bool pass_1 = asum_check(N, x, 1, r1);
	ASSERT_TRUE( pass_1 );

	T rx = engine::small_asum<T, N>::eval(x, 3);
	bool pass_x = asum_check(N, x, 3, rx);
	ASSERT_TRUE( pass_x );

}


TEST( SmallBlasL1, Asum1d )
{
	asum_test<double, 1>();
}

TEST( SmallBlasL1, Asum2d )
{
	asum_test<double, 2>();
}

TEST( SmallBlasL1, Asum3d )
{
	asum_test<double, 3>();
}

TEST( SmallBlasL1, Asum4d )
{
	asum_test<double, 4>();
}

TEST( SmallBlasL1, Asum6d )
{
	asum_test<double, 6>();
}



/************************************************
 *
 *  dot
 *
 ************************************************/

template<typename T>
bool dot_check(int n, const T *x, int incx, const T *y, int incy, const T r)
{
	T s(0);
	for (int i = 0; i < n; ++i) s += x[i * incx] * y[i * incy];

	if (s == r)
	{
		return true;
	}
	else
	{
		std::printf("dot-dump [n = %d, incx = %d, incy = %d]:\n", n, incx, incy);
		L1_prn_vec("  x", n, x, incx);
		L1_prn_vec("  y", n, y, incy);
		std::printf(" r = %g  (s = %g)\n\n", r, s);

		return false;
	}
}


template<typename T, int N>
void dot_test()
{
	T x[4 * N];
	T y[4 * N];

	L1_init_vec(4 * N, x, T(1));
	L1_init_vec(4 * N, y, T(2));

	T r11 = engine::small_dot<T, N>::eval(x, 1, y, 1);
	bool pass_11 = dot_check(N, x, 1, y, 1, r11);
	ASSERT_TRUE( pass_11 );

	T r1x = engine::small_dot<T, N>::eval(x, 1, y, 3);
	bool pass_1x = dot_check(N, x, 1, y, 3, r1x);
	ASSERT_TRUE( pass_1x );

	T rx1 = engine::small_dot<T, N>::eval(x, 2, y, 1);
	bool pass_x1 = dot_check(N, x, 2, y, 1, rx1);
	ASSERT_TRUE( pass_x1 );

	T rxx = engine::small_dot<T, N>::eval(x, 2, y, 3);
	bool pass_xx = dot_check(N, x, 2, y, 3, rxx);
	ASSERT_TRUE( pass_xx );
}


TEST( SmallBlasL1, Dot1d )
{
	dot_test<double, 1>();
}

TEST( SmallBlasL1, Dot2d )
{
	dot_test<double, 2>();
}

TEST( SmallBlasL1, Dot3d )
{
	dot_test<double, 3>();
}

TEST( SmallBlasL1, Dot4d )
{
	dot_test<double, 4>();
}

TEST( SmallBlasL1, Dot6d )
{
	dot_test<double, 6>();
}


/************************************************
 *
 *  nrm2
 *
 ************************************************/

template<typename T>
bool nrm2_check(int n, const T *x, int incx, const T r)
{
	T s(0);
	for (int i = 0; i < n; ++i) s += (x[i * incx]) * (x[i * incx]);
	s = std::sqrt(s);

	if (s == r)
	{
		return true;
	}
	else
	{
		std::printf("nrm2-dump [n = %d, incx = %d]:\n", n, incx);
		L1_prn_vec("  x", n, x, incx);
		std::printf(" r = %g  (s = %g)\n\n", r, s);

		return false;
	}
}


template<typename T, int N>
void nrm2_test()
{
	T x[4 * N];

	L1_init_vec(4 * N, x, T(1));

	T r1 = engine::small_nrm2<T, N>::eval(x, 1);
	bool pass_1 = nrm2_check(N, x, 1, r1);
	ASSERT_TRUE( pass_1 );

	T rx = engine::small_nrm2<T, N>::eval(x, 3);
	bool pass_x = nrm2_check(N, x, 3, rx);
	ASSERT_TRUE( pass_x );

}


TEST( SmallBlasL1, Nrm21d )
{
	nrm2_test<double, 1>();
}

TEST( SmallBlasL1, Nrm22d )
{
	nrm2_test<double, 2>();
}

TEST( SmallBlasL1, Nrm23d )
{
	nrm2_test<double, 3>();
}

TEST( SmallBlasL1, Nrm24d )
{
	nrm2_test<double, 4>();
}

TEST( SmallBlasL1, Nrm26d )
{
	nrm2_test<double, 6>();
}



/************************************************
 *
 *  axpy
 *
 ************************************************/

template<typename T>
bool axpy_check(int n, const T alpha, const T *x, int incx, const T *y0, const T *y, int incy)
{
	bool pass = true;
	for (int i = 0; i < n; ++i)
	{
		if (y[i * incy] != y0[i * incy] + alpha * x[i * incx]) pass = false;
	}

	if (!pass)
	{
		std::printf("axpy-dump [n = %d, incx = %d, incy = %d, alpha = %g]:\n", n, incx, incy, alpha);
		L1_prn_vec("  x ", n, x, incx);
		L1_prn_vec("  y0", n, y0, incy);
		L1_prn_vec("  y ", n, y, incy);
	}

	return pass;
}


template<typename T, int N>
void axpy_test()
{
	T x[4 * N];
	T y0[4 * N];
	T y[4 * N];

	L1_init_vec(4 * N, x, T(1));
	L1_init_vec(4 * N, y0, T(2));

	T a1(1);
	T a2(2);

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a1, x, 1, y, 1);
	bool pass_111 = axpy_check(N, a1, x, 1, y0, y, 1);
	ASSERT_TRUE( pass_111 );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a2, x, 1, y, 1);
	bool pass_11a = axpy_check(N, a2, x, 1, y0, y, 1);
	ASSERT_TRUE( pass_11a );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a1, x, 1, y, 3);
	bool pass_1x1 = axpy_check(N, a1, x, 1, y0, y, 3);
	ASSERT_TRUE( pass_1x1 );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a2, x, 1, y, 3);
	bool pass_1xa = axpy_check(N, a2, x, 1, y0, y, 3);
	ASSERT_TRUE( pass_1xa );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a1, x, 2, y, 1);
	bool pass_x11 = axpy_check(N, a1, x, 2, y0, y, 1);
	ASSERT_TRUE( pass_x11 );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a2, x, 2, y, 1);
	bool pass_x1a = axpy_check(N, a2, x, 2, y0, y, 1);
	ASSERT_TRUE( pass_x1a );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a1, x, 2, y, 3);
	bool pass_xx1 = axpy_check(N, a1, x, 2, y0, y, 3);
	ASSERT_TRUE( pass_xx1 );

	L1_init_vec(4 * N, y, T(2));
	engine::small_axpy<T, N>::eval(a2, x, 2, y, 3);
	bool pass_xxa = axpy_check(N, a2, x, 2, y0, y, 3);
	ASSERT_TRUE( pass_xxa );
}


TEST( SmallBlasL1, Axpy1d )
{
	axpy_test<double, 1>();
}

TEST( SmallBlasL1, Axpy2d )
{
	axpy_test<double, 2>();
}

TEST( SmallBlasL1, Axpy3d )
{
	axpy_test<double, 3>();
}

TEST( SmallBlasL1, Axpy4d )
{
	axpy_test<double, 4>();
}

TEST( SmallBlasL1, Axpy6d )
{
	axpy_test<double, 6>();
}


/************************************************
 *
 *  rot test
 *
 ************************************************/

template<typename T>
bool rot_check(int n, const T c, const T s,
		const T *x0, const T *x, int incx,
		const T *y0, const T *y, int incy)
{
	bool pass = true;
	for (int i = 0; i < n; ++i)
	{
		if (x[i * incx] != c * x0[i * incx] + s * y0[i * incy]) pass = false;
		if (y[i * incy] != c * y0[i * incy] - s * x0[i * incx]) pass = false;
	}

	if (!pass)
	{
		std::printf("rot-dump [n = %d, incx = %d, incy = %d, c = %g, s = %g]:\n", n, incx, incy, c, s);
		L1_prn_vec("  x0", n, x0, incx);
		L1_prn_vec("  y0", n, y0, incy);
		L1_prn_vec("  x ", n, x, incx);
		L1_prn_vec("  y ", n, y, incy);
	}

	return pass;
}

template<typename T, int N>
void rot_test()
{
	T x0[4 * N];
	T y0[4 * N];

	T x[4 * N];
	T y[4 * N];

	L1_init_vec(4 * N, x0, T(1));
	L1_init_vec(4 * N, y0, T(2));

	T c(2);
	T s(3);

	L1_init_vec(4 * N, x, T(1));
	L1_init_vec(4 * N, y, T(2));
	engine::small_rot<T, N>::eval(c, s, x, 1, y, 1);
	bool pass_11 = rot_check(N, c, s, x0, x, 1, y0, y, 1);
	ASSERT_TRUE( pass_11 );

	L1_init_vec(4 * N, x, T(1));
	L1_init_vec(4 * N, y, T(2));
	engine::small_rot<T, N>::eval(c, s, x, 1, y, 3);
	bool pass_1x = rot_check(N, c, s, x0, x, 1, y0, y, 3);
	ASSERT_TRUE( pass_1x );

	L1_init_vec(4 * N, x, T(1));
	L1_init_vec(4 * N, y, T(2));
	engine::small_rot<T, N>::eval(c, s, x, 2, y, 1);
	bool pass_x1 = rot_check(N, c, s, x0, x, 2, y0, y, 1);
	ASSERT_TRUE( pass_x1 );

	L1_init_vec(4 * N, x, T(1));
	L1_init_vec(4 * N, y, T(2));
	engine::small_rot<T, N>::eval(c, s, x, 2, y, 3);
	bool pass_xx = rot_check(N, c, s, x0, x, 2, y0, y, 3);
	ASSERT_TRUE( pass_xx );
}

TEST( SmallBlasL1, Rot1d )
{
	rot_test<double, 1>();
}

TEST( SmallBlasL1, Rot2d )
{
	rot_test<double, 2>();
}

TEST( SmallBlasL1, Rot3d )
{
	rot_test<double, 3>();
}

TEST( SmallBlasL1, Rot4d )
{
	rot_test<double, 4>();
}

TEST( SmallBlasL1, Rot6d )
{
	rot_test<double, 6>();
}



