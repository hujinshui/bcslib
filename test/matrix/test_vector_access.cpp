/**
 * @file test_vector_access.cpp
 *
 * Test vector accessors
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

template<class Mat>
bool verify_linear_reader(const Mat& A)
{
	typedef typename matrix_traits<Mat>::value_type T;

	dense_matrix<T> Amat(A);
	typename vec_reader<Mat>::type reader(A);

	index_t N = A.nelems();

	for (index_t i = 0; i < N; ++i)
	{
		if (Amat[i] != reader.get(i)) return false;
	}

	return true;
}


template<int CTRows, int CTCols>
void test_linear_reader_on_dense(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	ASSERT_TRUE( verify_linear_reader(A) );
}


TEST( LinearReader, DenseDD )
{
	test_linear_reader_on_dense<DynamicDim, DynamicDim>(5, 6);
}

TEST( LinearReader, DenseDS )
{
	test_linear_reader_on_dense<DynamicDim, 6>(5, 6);
}

TEST( LinearReader, DenseD1 )
{
	test_linear_reader_on_dense<DynamicDim, 1>(5, 1);
}

TEST( LinearReader, DenseSD )
{
	test_linear_reader_on_dense<5, DynamicDim>(5, 6);
}

TEST( LinearReader, DenseSS )
{
	test_linear_reader_on_dense<5, 6>(5, 6);
}

TEST( LinearReader, DenseS1 )
{
	test_linear_reader_on_dense<5, 1>(5, 1);
}

TEST( LinearReader, Dense1D )
{
	test_linear_reader_on_dense<1, DynamicDim>(1, 6);
}

TEST( LinearReader, Dense1S )
{
	test_linear_reader_on_dense<1, 6>(1, 6);
}

TEST( LinearReader, Dense11 )
{
	test_linear_reader_on_dense<1, 1>(1, 1);
}


template<int CTRows, int CTCols>
void test_linear_reader_on_refex(const index_t m, const index_t n)
{
	const index_t ldim = m + 3;

	dense_matrix<double> A0(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) A0[i] = double(i+2);

	ref_matrix_ex<double, CTRows, CTCols> A(A0.ptr_data(), m, n, ldim);
	ASSERT_TRUE( verify_linear_reader(A) );
}


TEST( LinearReader, RefExDD )
{
	test_linear_reader_on_refex<DynamicDim, DynamicDim>(5, 6);
}

TEST( LinearReader, RefExDS )
{
	test_linear_reader_on_refex<DynamicDim, 6>(5, 6);
}

TEST( LinearReader, RefExD1 )
{
	test_linear_reader_on_refex<DynamicDim, 1>(5, 1);
}

TEST( LinearReader, RefExSD )
{
	test_linear_reader_on_refex<5, DynamicDim>(5, 6);
}

TEST( LinearReader, RefExSS )
{
	test_linear_reader_on_refex<5, 6>(5, 6);
}

TEST( LinearReader, RefExS1 )
{
	test_linear_reader_on_refex<5, 1>(5, 1);
}

TEST( LinearReader, RefEx1D )
{
	test_linear_reader_on_refex<1, DynamicDim>(1, 6);
}

TEST( LinearReader, RefEx1S )
{
	test_linear_reader_on_refex<1, 6>(1, 6);
}

TEST( LinearReader, RefEx11 )
{
	test_linear_reader_on_refex<1, 1>(1, 1);
}



template<class Mat>
bool verify_colwise_reader(const Mat& A)
{
	typedef typename matrix_traits<Mat>::value_type T;
	typedef typename colwise_reader_bank<Mat>::type bank_t;
	typedef typename bank_t::reader_type reader_t;

	dense_matrix<T> Amat(A);
	bank_t bank(A);

	index_t m = A.nrows();
	index_t n = A.ncolumns();

	for (index_t j = 0; j < n; ++j)
	{
		reader_t in(bank, j);
		for (index_t i = 0; i < m; ++i)
		{
			if (in.get(i) != A(i, j)) return false;
		}
	}

	return true;
}


template<int CTRows, int CTCols>
void test_colwise_reader_on_dense(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	ASSERT_TRUE( verify_colwise_reader(A) );
}


TEST( ColwiseReader, DenseDD )
{
	test_colwise_reader_on_dense<DynamicDim, DynamicDim>(5, 6);
}

TEST( ColwiseReader, DenseDS )
{
	test_colwise_reader_on_dense<DynamicDim, 6>(5, 6);
}

TEST( ColwiseReader, DenseD1 )
{
	test_colwise_reader_on_dense<DynamicDim, 1>(5, 1);
}

TEST( ColwiseReader, DenseSD )
{
	test_colwise_reader_on_dense<5, DynamicDim>(5, 6);
}

TEST( ColwiseReader, DenseSS )
{
	test_colwise_reader_on_dense<5, 6>(5, 6);
}

TEST( ColwiseReader, DenseS1 )
{
	test_colwise_reader_on_dense<5, 1>(5, 1);
}

TEST( ColwiseReader, Dense1D )
{
	test_colwise_reader_on_dense<1, DynamicDim>(1, 6);
}

TEST( ColwiseReader, Dense1S )
{
	test_colwise_reader_on_dense<1, 6>(1, 6);
}

TEST( ColwiseReader, Dense11 )
{
	test_colwise_reader_on_dense<1, 1>(1, 1);
}

template<int CTRows, int CTCols>
void test_colwise_reader_on_refex(const index_t m, const index_t n)
{
	const index_t ldim = m + 3;

	dense_matrix<double> A0(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) A0[i] = double(i+2);

	ref_matrix_ex<double, CTRows, CTCols> A(A0.ptr_data(), m, n, ldim);
	ASSERT_TRUE( verify_colwise_reader(A) );
}


TEST( ColwiseReader, RefExDD )
{
	test_colwise_reader_on_refex<DynamicDim, DynamicDim>(5, 6);
}

TEST( ColwiseReader, RefExDS )
{
	test_colwise_reader_on_refex<DynamicDim, 6>(5, 6);
}

TEST( ColwiseReader, RefExD1 )
{
	test_colwise_reader_on_refex<DynamicDim, 1>(5, 1);
}

TEST( ColwiseReader, RefExSD )
{
	test_colwise_reader_on_refex<5, DynamicDim>(5, 6);
}

TEST( ColwiseReader, RefExSS )
{
	test_colwise_reader_on_refex<5, 6>(5, 6);
}

TEST( ColwiseReader, RefExS1 )
{
	test_colwise_reader_on_refex<5, 1>(5, 1);
}

TEST( ColwiseReader, RefEx1D )
{
	test_colwise_reader_on_refex<1, DynamicDim>(1, 6);
}

TEST( ColwiseReader, RefEx1S )
{
	test_colwise_reader_on_refex<1, 6>(1, 6);
}

TEST( ColwiseReader, RefEx11 )
{
	test_colwise_reader_on_refex<1, 1>(1, 1);
}




