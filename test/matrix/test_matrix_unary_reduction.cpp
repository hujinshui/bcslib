/**
 * @file test_matrix_reduction.cpp
 *
 * Unit testing of unary matrix reduction
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


template<class VecFun, class Mat>
static double evaluate_reduction_value(VecFun vecfun, const IMatrixXpr<Mat, double>& mat)
{
	dense_matrix<double> dmat(mat);
	return vecfun(dmat.nelems(), dmat.ptr_data());
}

struct vec_sum_ftor
{
	double operator() (index_t n, const double *a) const
	{
		double s = 0;
		for (index_t i = 0; i < n; ++i) s += a[i];
		return s;
	}
};

template<class Mat>
void test_sum(const IMatrixXpr<Mat, double>& mat)
{
	double v0 = evaluate_reduction_value(vec_sum_ftor(), mat);
	ASSERT_EQ(v0, sum(mat));
}


template<int CTRows, int CTCols>
void test_sum_on_densemat(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> mat(m, n);

	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i) mat(i, j) = double((i+1) + j * m);

	test_sum(mat);
}

template<int CTRows, int CTCols>
void test_sum_on_refex_mat(const index_t m, const index_t n)
{
	index_t ldim = m + 3;
	dense_matrix<double> mat(ldim, n);

	for (int i = 0; i < mat.nelems(); ++i) mat[i] = (i+2);

	ref_matrix_ex<double, CTRows, CTCols> rmat(mat.ptr_data(), m, n, ldim);
	test_sum(rmat);
}


TEST( UnaryMatrixReduction, SumDenseMatDD )
{
	test_sum_on_densemat<DynamicDim, DynamicDim>(5, 6);
}

TEST( UnaryMatrixReduction, SumDenseMatDS )
{
	test_sum_on_densemat<DynamicDim, 6>(5, 6);
}

TEST( UnaryMatrixReduction, SumDenseMatD2 )
{
	test_sum_on_densemat<DynamicDim, 2>(5, 2);
}

TEST( UnaryMatrixReduction, SumDenseMatD1 )
{
	test_sum_on_densemat<DynamicDim, 1>(5, 1);
}

TEST( UnaryMatrixReduction, SumDenseMatSD )
{
	test_sum_on_densemat<5, DynamicDim>(5, 6);
}

TEST( UnaryMatrixReduction, SumDenseMatSS )
{
	test_sum_on_densemat<5, 6>(5, 6);
}

TEST( UnaryMatrixReduction, SumDenseMatS2 )
{
	test_sum_on_densemat<5, 2>(5, 2);
}

TEST( UnaryMatrixReduction, SumDenseMatS1 )
{
	test_sum_on_densemat<5, 1>(5, 1);
}

TEST( UnaryMatrixReduction, SumDenseMat2D )
{
	test_sum_on_densemat<2, DynamicDim>(2, 6);
}

TEST( UnaryMatrixReduction, SumDenseMat2S )
{
	test_sum_on_densemat<2, 6>(2, 6);
}

TEST( UnaryMatrixReduction, SumDenseMat22 )
{
	test_sum_on_densemat<2, 2>(2, 2);
}

TEST( UnaryMatrixReduction, SumDenseMat21 )
{
	test_sum_on_densemat<2, 1>(2, 1);
}

TEST( UnaryMatrixReduction, SumDenseMat1D )
{
	test_sum_on_densemat<1, DynamicDim>(1, 6);
}

TEST( UnaryMatrixReduction, SumDenseMat1S )
{
	test_sum_on_densemat<1, 6>(1, 6);
}

TEST( UnaryMatrixReduction, SumDenseMat12 )
{
	test_sum_on_densemat<1, 2>(1, 2);
}

TEST( UnaryMatrixReduction, SumDenseMat11 )
{
	test_sum_on_densemat<1, 1>(1, 1);
}



TEST( UnaryMatrixReduction, SumRefExMatDD )
{
	test_sum_on_refex_mat<DynamicDim, DynamicDim>(5, 6);
}

TEST( UnaryMatrixReduction, SumRefExMatDS )
{
	test_sum_on_refex_mat<DynamicDim, 6>(5, 6);
}

TEST( UnaryMatrixReduction, SumRefExMatD2 )
{
	test_sum_on_refex_mat<DynamicDim, 2>(5, 2);
}

TEST( UnaryMatrixReduction, SumRefExMatD1 )
{
	test_sum_on_refex_mat<DynamicDim, 1>(5, 1);
}

TEST( UnaryMatrixReduction, SumRefExMatSD )
{
	test_sum_on_refex_mat<5, DynamicDim>(5, 6);
}

TEST( UnaryMatrixReduction, SumRefExMatSS )
{
	test_sum_on_refex_mat<5, 6>(5, 6);
}

TEST( UnaryMatrixReduction, SumRefExMatS2 )
{
	test_sum_on_refex_mat<5, 2>(5, 2);
}

TEST( UnaryMatrixReduction, SumRefExMatS1 )
{
	test_sum_on_refex_mat<5, 1>(5, 1);
}

TEST( UnaryMatrixReduction, SumRefExMat2D )
{
	test_sum_on_refex_mat<2, DynamicDim>(2, 6);
}

TEST( UnaryMatrixReduction, SumRefExMat2S )
{
	test_sum_on_refex_mat<2, 6>(2, 6);
}

TEST( UnaryMatrixReduction, SumRefExMat22 )
{
	test_sum_on_refex_mat<2, 2>(2, 2);
}

TEST( UnaryMatrixReduction, SumRefExMat21 )
{
	test_sum_on_refex_mat<2, 1>(2, 1);
}

TEST( UnaryMatrixReduction, SumRefExMat1D )
{
	test_sum_on_refex_mat<1, DynamicDim>(1, 6);
}

TEST( UnaryMatrixReduction, SumRefExMat1S )
{
	test_sum_on_refex_mat<1, 6>(1, 6);
}

TEST( UnaryMatrixReduction, SumRefExMat12 )
{
	test_sum_on_refex_mat<1, 2>(1, 2);
}

TEST( UnaryMatrixReduction, SumRefExMat11 )
{
	test_sum_on_refex_mat<1, 1>(1, 1);
}


struct vec_mean_ftor
{
	double operator() (index_t n, const double *a) const
	{
		double s = 0;
		for (index_t i = 0; i < n; ++i) s += a[i];
		return s / double(n);
	}
};

template<class Mat>
void test_mean(const IMatrixXpr<Mat, double>& mat)
{
	double v0 = evaluate_reduction_value(vec_mean_ftor(), mat);
	ASSERT_EQ(v0, mean(mat));
}


template<int CTRows, int CTCols>
void test_mean_on_densemat(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> mat(m, n);

	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i) mat(i, j) = double((i+1) + j * m);

	test_mean(mat);
}

TEST( UnaryMatrixReduction, MeanDenseMatDD )
{
	test_mean_on_densemat<DynamicDim, DynamicDim>(5, 6);
}

TEST( UnaryMatrixReduction, MeanDenseMatDS )
{
	test_mean_on_densemat<DynamicDim, 6>(5, 6);
}

TEST( UnaryMatrixReduction, MeanDenseMatD1 )
{
	test_mean_on_densemat<DynamicDim, 1>(5, 1);
}

TEST( UnaryMatrixReduction, MeanDenseMatSD )
{
	test_mean_on_densemat<5, DynamicDim>(5, 6);
}

TEST( UnaryMatrixReduction, MeanDenseMatSS )
{
	test_mean_on_densemat<5, 6>(5, 6);
}

TEST( UnaryMatrixReduction, MeanDenseMatS1 )
{
	test_mean_on_densemat<5, 1>(5, 1);
}

TEST( UnaryMatrixReduction, MeanDenseMat1D )
{
	test_mean_on_densemat<1, DynamicDim>(1, 6);
}

TEST( UnaryMatrixReduction, MeanDenseMat1S )
{
	test_mean_on_densemat<1, 6>(1, 6);
}

TEST( UnaryMatrixReduction, MeanDenseMat11 )
{
	test_mean_on_densemat<1, 1>(1, 1);
}



TEST( UnaryMatrixReduction, MatMaxVal )
{
	mat_f64 A(4, 5);
	for (index_t i = 0; i < A.nelems(); ++i) A[i] = double(i+1);

	double v0 = 20.0;
	ASSERT_EQ(v0, max_val(A));
}

TEST( UnaryMatrixReduction, MatMinVal )
{
	mat_f64 A(4, 5);
	for (index_t i = 0; i < A.nelems(); ++i) A[i] = double(i+1);

	double v0 = 1.0;
	ASSERT_EQ(v0, min_val(A));
}

TEST( UnaryMatrixReduction, MatL1Norm )
{
	mat_f64 A(4, 5);
	for (index_t i = 0; i < A.nelems(); ++i)
		A[i] = double(i+1) * (i % 2 == 0 ? 1.0 : -1.0);

	double v0 = 210;
	ASSERT_EQ(v0, L1norm(A));
}

TEST( UnaryMatrixReduction, MatSqL2Norm )
{
	mat_f64 A(4, 5);
	for (index_t i = 0; i < A.nelems(); ++i)
		A[i] = double(i+1) * (i % 2 == 0 ? 1.0 : -1.0);

	double v0 = 2870;
	ASSERT_EQ(v0, sqL2norm(A));
}

TEST( UnaryMatrixReduction, MatL2Norm )
{
	mat_f64 A(4, 5);
	for (index_t i = 0; i < A.nelems(); ++i)
		A[i] = double(i+1) * (i % 2 == 0 ? 1.0 : -1.0);

	double v0 = math::sqrt(2870.0);
	ASSERT_NEAR(v0, L2norm(A), 1.0e-14);
}


TEST( UnaryMatrixReduction, MatLinfNorm )
{
	mat_f64 A(4, 5);
	for (index_t i = 0; i < A.nelems(); ++i)
		A[i] = double(i+1) * (i % 2 == 0 ? 1.0 : -1.0);

	double v0 = 20;
	ASSERT_EQ(v0, Linfnorm(A));
}







