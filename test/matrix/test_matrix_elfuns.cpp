/**
 * @file test_matrix_elfuns.h
 *
 * Unit testing for Elementary functions on matrices
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

TEST( MatrixElFuns, Sqrt )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::sqrt(a[i]);
	col_f64 c = sqrt(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Pow )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::sqr(a[i]);
	col_f64 c = pow(a, 2.0);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Exp )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::exp(a[i]);
	col_f64 c = exp(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Log )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::log(a[i]);
	col_f64 c = log(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Log10 )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::log10(a[i]);
	col_f64 c = log10(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Floor )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1) * 3.6;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::floor(a[i]);
	col_f64 c = floor(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}

TEST( MatrixElFuns, Ceil )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1) * 3.6;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::ceil(a[i]);
	col_f64 c = ceil(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Sin )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::sin(a[i]);
	col_f64 c = sin(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Cos )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::cos(a[i]);
	col_f64 c = cos(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Tan )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::tan(a[i]);
	col_f64 c = tan(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Asin )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::asin(a[i]);
	col_f64 c = asin(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Acos )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::acos(a[i]);
	col_f64 c = acos(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Atan )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::atan(a[i]);
	col_f64 c = atan(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Atan2 )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = double(i + 1) * 1.6 + 2.5;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::atan2(a[i], b[i]);
	col_f64 c = atan2(a, b);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Sinh )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::sinh(a[i]);
	col_f64 c = sinh(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Cosh )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::cosh(a[i]);
	col_f64 c = cosh(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}


TEST( MatrixElFuns, Tanh )
{
	const int len = 16;
	const double tol = 1.0e-14;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = double(i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = math::tanh(a[i]);
	col_f64 c = tanh(a);

	ASSERT_TRUE( is_approx(c, c0, tol) );
}





