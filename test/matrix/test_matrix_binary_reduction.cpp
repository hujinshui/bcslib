/**
 * @file test_matrix_binary_reduction.cpp
 *
 *  Unit testing for binary reduction
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

template<class VecFun, class LMat, class RMat>
static double evaluate_reduction_value(VecFun vecfun,
		const IMatrixXpr<LMat, double>& a,
		const IMatrixXpr<RMat, double>& b)
{
	dense_matrix<double> da(a);
	dense_matrix<double> db(b);

	return vecfun(da.nelems(), da.ptr_data(), db.ptr_data());
}

struct vec_dot_ftor
{
	double operator() (index_t n, const double *a, const double *b) const
	{
		double s = 0;
		for (index_t i = 0; i < n; ++i) s += a[i] * b[i];
		return s;
	}
};

template<class LMat, class RMat>
void test_dot(const IMatrixXpr<LMat, double>& a, const IMatrixXpr<RMat, double>& b)
{
	double v0 = evaluate_reduction_value(vec_dot_ftor(), a, b);
	ASSERT_EQ(v0, dot(a, b));
}


template<int CTRows, int CTCols>
void test_dot_on_densemat(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> a(m, n);
	dense_matrix<double, CTRows, CTCols> b(m, n);

	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i)
		{
			a(i, j) = double((i+1) + j * m);
			b(i, j) = double((i+2) + j * m);
		}

	test_dot(a, b);
}


template<int CTRows, int CTCols>
void test_dot_on_refex_mat(const index_t m, const index_t n)
{
	index_t ldim = m + 3;

	dense_matrix<double> a(ldim, n);
	dense_matrix<double> b(ldim, n);

	for (index_t i = 0; i < a.nelems(); ++i)
	{
		a[i] = double(i + 2);
		b[i] = a[i] + 1;
	}

	ref_matrix_ex<double, CTRows, CTCols> ar(a.ptr_data(), m, n, ldim);
	ref_matrix_ex<double, CTRows, CTCols> br(b.ptr_data(), m, n, ldim);

	test_dot(ar, br);
}



TEST( BinaryMatrixReduction, DotProdDenseMatDD )
{
	test_dot_on_densemat<DynamicDim, DynamicDim>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMatDS )
{
	test_dot_on_densemat<DynamicDim, 6>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMatD2 )
{
	test_dot_on_densemat<DynamicDim, 2>(5, 2);
}

TEST( BinaryMatrixReduction, DotProdDenseMatD1 )
{
	test_dot_on_densemat<DynamicDim, 1>(5, 1);
}

TEST( BinaryMatrixReduction, DotProdDenseMatSD )
{
	test_dot_on_densemat<5, DynamicDim>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMatSS )
{
	test_dot_on_densemat<5, 6>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMatS2 )
{
	test_dot_on_densemat<5, 2>(5, 2);
}

TEST( BinaryMatrixReduction, DotProdDenseMatS1 )
{
	test_dot_on_densemat<5, 1>(5, 1);
}

TEST( BinaryMatrixReduction, DotProdDenseMat2D )
{
	test_dot_on_densemat<2, DynamicDim>(2, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMat2S )
{
	test_dot_on_densemat<2, 6>(2, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMat22 )
{
	test_dot_on_densemat<2, 2>(2, 2);
}

TEST( BinaryMatrixReduction, DotProdDenseMat21 )
{
	test_dot_on_densemat<2, 1>(2, 1);
}

TEST( BinaryMatrixReduction, DotProdDenseMat1D )
{
	test_dot_on_densemat<1, DynamicDim>(1, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMat1S )
{
	test_dot_on_densemat<1, 6>(1, 6);
}

TEST( BinaryMatrixReduction, DotProdDenseMat12 )
{
	test_dot_on_densemat<1, 2>(1, 2);
}

TEST( BinaryMatrixReduction, DotProdDenseMat11 )
{
	test_dot_on_densemat<1, 1>(1, 1);
}


TEST( BinaryMatrixReduction, DotProdRefExMatDD )
{
	test_dot_on_refex_mat<DynamicDim, DynamicDim>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMatDS )
{
	test_dot_on_refex_mat<DynamicDim, 6>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMatD2 )
{
	test_dot_on_refex_mat<DynamicDim, 2>(5, 2);
}

TEST( BinaryMatrixReduction, DotProdRefExMatD1 )
{
	test_dot_on_refex_mat<DynamicDim, 1>(5, 1);
}

TEST( BinaryMatrixReduction, DotProdRefExMatSD )
{
	test_dot_on_refex_mat<5, DynamicDim>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMatSS )
{
	test_dot_on_refex_mat<5, 6>(5, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMatS2 )
{
	test_dot_on_refex_mat<5, 2>(5, 2);
}

TEST( BinaryMatrixReduction, DotProdRefExMatS1 )
{
	test_dot_on_refex_mat<5, 1>(5, 1);
}

TEST( BinaryMatrixReduction, DotProdRefExMat2D )
{
	test_dot_on_refex_mat<2, DynamicDim>(2, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMat2S )
{
	test_dot_on_refex_mat<2, 6>(2, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMat22 )
{
	test_dot_on_refex_mat<2, 2>(2, 2);
}

TEST( BinaryMatrixReduction, DotProdRefExMat21 )
{
	test_dot_on_refex_mat<2, 1>(2, 1);
}

TEST( BinaryMatrixReduction, DotProdRefExMat1D )
{
	test_dot_on_refex_mat<1, DynamicDim>(1, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMat1S )
{
	test_dot_on_refex_mat<1, 6>(1, 6);
}

TEST( BinaryMatrixReduction, DotProdRefExMat12 )
{
	test_dot_on_refex_mat<1, 2>(1, 2);
}

TEST( BinaryMatrixReduction, DotProdRefExMat11 )
{
	test_dot_on_refex_mat<1, 1>(1, 1);
}



TEST( BinaryMatrixReduction, MatL1NormDiff )
{
	mat_f64 A(4, 5);
	mat_f64 B(4, 5);

	for (index_t i = 0; i < A.nelems(); ++i)
	{
		A[i] = double(i+1);
		B[i] = 12.0 - A[i] * 2.0;
	}

	double v0 = 426;
	ASSERT_EQ(v0, L1norm_diff(A, B));
}


TEST( BinaryMatrixReduction, MatSqL2NormDiff )
{
	mat_f64 A(4, 5);
	mat_f64 B(4, 5);

	for (index_t i = 0; i < A.nelems(); ++i)
	{
		A[i] = double(i+1);
		B[i] = 12.0 - A[i] * 2.0;
	}

	double v0 = 13590;
	ASSERT_EQ(v0, sqL2norm_diff(A, B));
}

TEST( BinaryMatrixReduction, MatL2NormDiff )
{
	mat_f64 A(4, 5);
	mat_f64 B(4, 5);

	for (index_t i = 0; i < A.nelems(); ++i)
	{
		A[i] = double(i+1);
		B[i] = 12.0 - A[i] * 2.0;
	}

	double v0 = math::sqrt(13590.0);
	ASSERT_EQ(v0, L2norm_diff(A, B));
}


TEST( BinaryMatrixReduction, MatLinfNormDiff )
{
	mat_f64 A(4, 5);
	mat_f64 B(4, 5);

	for (index_t i = 0; i < A.nelems(); ++i)
	{
		A[i] = double(i+1);
		B[i] = 12.0 - A[i] * 2.0;
	}

	double v0 = 48;
	ASSERT_EQ(v0, Linfnorm_diff(A, B));
}
