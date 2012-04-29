/**
 * @file test_matrix_arithmetic.cpp
 *
 * Unit testing of Matrix arithmetic
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


/************************************************
 *
 *  Plus
 *
 ************************************************/

TEST( MatrixArith, PlusMatMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] + b[i];
	col_f64 c = a + b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, PlusMatSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] + b;
	col_f64 c = a + b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, PlusScaMat )
{
	const int len = 16;

	const double a = 2.0;
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a + b[i];
	col_f64 c = a + b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplacePlusMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] + b[i];
	col_f64 c(a);
	c += b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplacePlusSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] + b;
	col_f64 c(a);
	c += b;

	ASSERT_TRUE( is_equal(c, c0) );
}


/************************************************
 *
 *  Minus
 *
 ************************************************/

TEST( MatrixArith, MinusMatMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] - b[i];
	col_f64 c = a - b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, MinusMatSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] - b;
	col_f64 c = a - b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, MinusScaMat )
{
	const int len = 16;

	const double a = 2.0;
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a - b[i];
	col_f64 c = a - b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplaceMinusMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] - b[i];
	col_f64 c(a);
	c -= b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplaceMinusSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] - b;
	col_f64 c(a);
	c -= b;

	ASSERT_TRUE( is_equal(c, c0) );
}


/************************************************
 *
 *  Times
 *
 ************************************************/

TEST( MatrixArith, TimesMatMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] * b[i];
	col_f64 c = a * b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, TimesMatSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] * b;
	col_f64 c = a * b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, TimesScaMat )
{
	const int len = 16;

	const double a = 2.0;
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a * b[i];
	col_f64 c = a * b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplaceTimesMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = 2 * (100. - i);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] * b[i];
	col_f64 c(a);
	c *= b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplaceTimesSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] * b;
	col_f64 c(a);
	c *= b;

	ASSERT_TRUE( is_equal(c, c0) );
}


/************************************************
 *
 *  Divides
 *
 ************************************************/

TEST( MatrixArith, DividesMatMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = (i % 2 == 0 ? 2.0 : 4.0);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] / b[i];
	col_f64 c = a / b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, DividesMatSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] / b;
	col_f64 c = a / b;

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, DividesScaMat )
{
	const int len = 16;

	const double a = 2.0;
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = (i % 2 == 0 ? 2.0 : 4.0);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a / b[i];
	col_f64 c = a / b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplaceDividesMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	col_f64 b(len); for (index_t i = 0; i < len; ++i) b[i] = (i % 2 == 0 ? 2.0 : 4.0);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] / b[i];
	col_f64 c(a);
	c /= b;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, InplaceDividesSca )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);
	const double b = 2.0;

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] / b;
	col_f64 c(a);
	c /= b;

	ASSERT_TRUE( is_equal(c, c0) );
}


/************************************************
 *
 *  Other Arithmetic functions
 *
 ************************************************/

TEST( MatrixArith, NegateMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i+1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = -a[i];
	col_f64 c = -a;

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, RcpMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i % 2 == 0 ? 2.0 : 4.0);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = 1 / a[i];
	col_f64 c = rcp(a);

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixArith, SqrMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i + 1);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] * a[i];
	col_f64 c = sqr(a);

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( MatrixArith, CubeMat )
{
	const int len = 16;

	col_f64 a(len); for (index_t i = 0; i < len; ++i) a[i] = (i % 2 == 0 ? 2.0 : 4.0);

	col_f64 c0(len); for (index_t i = 0; i < len; ++i) c0[i] = a[i] * a[i] * a[i];
	col_f64 c = cube(a);

	ASSERT_TRUE( is_equal(c, c0) );
}






