/**
 * @file test_ref_matrix.cpp
 *
 * Unit Testing of Ref/CRef Matrices
 *
 * @author Dahua Lin
 */

#include "bcs_test_basics.h"
#include <bcslib/matrix/matrix.h>
#include <bcslib/base/type_traits.h>

using namespace bcs;
using namespace bcs::test;

// explicit template for syntax check

template class bcs::RefMatrix<double, DynamicDim, DynamicDim>;
template class bcs::RefMatrix<double, DynamicDim, 1>;
template class bcs::RefMatrix<double, 1, DynamicDim>;
template class bcs::RefMatrix<double, 2, 2>;

template class bcs::CRefMatrix<double, DynamicDim, DynamicDim>;
template class bcs::CRefMatrix<double, DynamicDim, 1>;
template class bcs::CRefMatrix<double, 1, DynamicDim>;
template class bcs::CRefMatrix<double, 2, 2>;

template class bcs::RefCol<double, DynamicDim>;
template class bcs::RefCol<double, 3>;
template class bcs::RefRow<double, DynamicDim>;
template class bcs::RefRow<double, 3>;

template class bcs::CRefCol<double, DynamicDim>;
template class bcs::CRefCol<double, 3>;
template class bcs::CRefRow<double, DynamicDim>;
template class bcs::CRefRow<double, 3>;

template class bcs::StepVector<double, VertDir, DynamicDim>;
template class bcs::StepVector<double, HorzDir, DynamicDim>;
template class bcs::StepVector<double, VertDir, 3>;
template class bcs::StepVector<double, HorzDir, 3>;

template class bcs::CStepVector<double, VertDir, DynamicDim>;
template class bcs::CStepVector<double, HorzDir, DynamicDim>;
template class bcs::CStepVector<double, VertDir, 3>;
template class bcs::CStepVector<double, HorzDir, 3>;

// auxiliary functions

template<class Derived, typename T>
static bool verify_dense_matrix(const IDenseMatrix<Derived, T>& A, index_t m, index_t n)
{
	if (!(A.nrows() == m)) return false;
	if (!(A.ncolumns() == n)) return false;
	if (!(A.nelems() == m * n)) return false;
	if (!(A.size() == size_t(m * n))) return false;
	if (!(A.is_empty() == (m == 0 || n == 0))) return false;
	if (!(A.is_vector() == (m == 1 || n == 1))) return false;
	if (!(A.lead_dim() == m)) return false;

	if (A.is_empty())
	{
		if (A.ptr_base() != BCS_NULL) return false;
	}
	else
	{
		if (A.ptr_base() == BCS_NULL) return false;
		const T *p = A.ptr_base();

		for (index_t i = 0; i < m; ++i)
		{
			for (index_t j = 0; j < n; ++j)
			{
				if (A.elem(i, j) != p[i + j * m]) return false;
				if (A(i, j) != p[i + j * m]) return false;
				if (A[i + j * m] != p[i + j * m]) return false;
			}
		}
	}

	DenseMatrixCapture<Derived, T> cap(A.derived());
	if ( &(cap.get()) != &(A.derived()) ) return false;

	return true;
}


template<typename T, typename Dir, int Dim>
static bool verify_stepvec(const CStepVector<T, Dir, Dim>& A, index_t len, index_t step)
{
	index_t m = 0;
	index_t n = 0;

	if (bcs::is_same<Dir, VertDir>::value) { m = len; n = 1; }
	else if (bcs::is_same<Dir, HorzDir>::value) { m = 1; n = len; }
	else throw std::runtime_error("Dir is neither Vert nor Horz");

	if (!(A.nrows() == m)) return false;
	if (!(A.ncolumns() == n)) return false;
	if (!(A.nelems() == len)) return false;
	if (!(A.size() == size_t(len))) return false;
	if (!(A.step() == step)) return false;
	if (!(A.is_empty() == (len == 0))) return false;
	if (!(A.is_vector())) return false;

	if (len > 0)
	{
		const T *p = &(A[0]);

		if (bcs::is_same<Dir, VertDir>::value)
		{
			for (index_t i = 0; i < len; ++i)
			{
				if (A[i] != p[i * step]) return false;
				if (A(i, 0) != p[i * step]) return false;
				if (A.elem(i, 0) != p[i * step]) return false;
			}
		}
		else
		{
			for (index_t i = 0; i < len; ++i)
			{
				if (A[i] != p[i * step]) return false;
				if (A(0, i) != p[i * step]) return false;
				if (A.elem(0, i) != p[i * step]) return false;
			}
		}
	}

	return true;
}

template<typename T, typename Dir, int Dim>
static bool verify_stepvec(const StepVector<T, Dir, Dim>& A, index_t len, index_t step)
{
	index_t m = 0;
	index_t n = 0;

	if (bcs::is_same<Dir, VertDir>::value) { m = len; n = 1; }
	else if (bcs::is_same<Dir, HorzDir>::value) { m = 1; n = len; }
	else throw std::runtime_error("Dir is neither Vert nor Horz");

	if (!(A.nrows() == m)) return false;
	if (!(A.ncolumns() == n)) return false;
	if (!(A.nelems() == len)) return false;
	if (!(A.size() == size_t(len))) return false;
	if (!(A.step() == step)) return false;
	if (!(A.is_empty() == (len == 0))) return false;
	if (!(A.is_vector())) return false;

	if (len > 0)
	{
		const T *p = &(A[0]);

		if (bcs::is_same<Dir, VertDir>::value)
		{
			for (index_t i = 0; i < len; ++i)
			{
				if (A[i] != p[i * step]) return false;
				if (A(i, 0) != p[i * step]) return false;
				if (A.elem(i, 0) != p[i * step]) return false;
			}
		}
		else
		{
			for (index_t i = 0; i < len; ++i)
			{
				if (A[i] != p[i * step]) return false;
				if (A(0, i) != p[i * step]) return false;
				if (A.elem(0, i) != p[i * step]) return false;
			}
		}
	}

	return true;
}





// test cases

TEST( RefClasses, RefMatrix)
{
	const index_t m = 2;
	const index_t n = 3;

	double src[m * n] = {1, 2, 3, 4, 5, 6};
	double dst[m * n] = {0, 0, 0, 0, 0, 0};

	CRefMatrix<double> S(src, m, n);
	RefMatrix<double> D(dst, m, n);

	ASSERT_TRUE( verify_dense_matrix(S, m, n) );
	ASSERT_TRUE( verify_dense_matrix(D, m, n) );

	ASSERT_TRUE( S.ptr_base() == src );
	ASSERT_TRUE( D.ptr_base() == dst );

	D.fill(7.0);
	ASSERT_TRUE( elems_equal(6, dst, 7.0) );

	D.copy_from(src);
	ASSERT_TRUE( elems_equal(6, dst, src) );

	CRefMatrix<double> S2(D);

	ASSERT_TRUE( verify_dense_matrix(S2, m, n) );
	ASSERT_TRUE( S2.ptr_base() == D.ptr_base() );

	double tar[m * n] = {10, 20, 30, 40, 50, 60};
	RefMatrix<double> S3(tar, m, n);

	ASSERT_TRUE( verify_dense_matrix(S3, m, n) );
	ASSERT_TRUE( S3.ptr_base() == tar );

	S3 = S2;

	ASSERT_TRUE( S3.ptr_base() == tar );
	ASSERT_TRUE( S2.ptr_base() == D.ptr_base() );
	ASSERT_FALSE( S3.ptr_base() == S2.ptr_base() );

	ASSERT_TRUE( is_equal(S2, S3) );
	ASSERT_TRUE( elems_equal(6, dst, tar) );
}


TEST( RefClasses, RefVector )
{
	const index_t len = 6;

	double src[len] = {1, 2, 3, 4, 5, 6};
	double dst[len] = {-1, -1, -1, -1, -1};

	CRefRow<double> crow(src, len);
	RefRow<double> row(dst, len);

	ASSERT_TRUE( verify_dense_matrix(crow, 1, len) );
	ASSERT_TRUE( verify_dense_matrix(row, 1, len) );

	ASSERT_TRUE( crow.ptr_base() == src );
	ASSERT_TRUE( row.ptr_base() == dst );

	row.zero();

	ASSERT_TRUE( elems_equal(len, dst, 0.0) );

	row.copy_from(src);

	ASSERT_TRUE( elems_equal(len, dst, src) );
	ASSERT_TRUE( is_equal(row, crow) );

	CRefCol<double> ccol(src, len);
	RefCol<double> col(dst, len);

	ASSERT_TRUE( verify_dense_matrix(ccol, len, 1) );
	ASSERT_TRUE( verify_dense_matrix(col, len, 1) );

	ASSERT_TRUE( is_equal(ccol, col) );

	col.fill(2.0);

	ASSERT_TRUE( elems_equal(len, dst, 2.0) );
}


TEST( RefClasses, StepColumn )
{
	const index_t len = 5;
	const index_t step = 2;
	const index_t ulen = 9;

	double src[ulen] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	double dst[ulen] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

	CStepVector<double, VertDir> ccol(src, len, step);
	StepVector<double, VertDir> col(dst, len, step);

	ASSERT_TRUE( verify_stepvec(ccol, len, step) );
	ASSERT_TRUE( verify_stepvec(col, len, step) );

	col.fill(0.0);

	double r1[ulen] = {0, 8, 0, 6, 0, 4, 0, 2, 0};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r1) );

	col.copy_from(src);

	double r2[ulen] = {1, 8, 2, 6, 3, 4, 4, 2, 5};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r2) );

	zero_elems(size_t(ulen), dst);
	col = RefRow<double>(src, len);
	col = RefCol<double>(src, len);

	double r3[ulen] = {1, 0, 2, 0, 3, 0, 4, 0, 5};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r3) );

	zero_elems(size_t(ulen), dst);
	col = ccol;

	double r4[ulen] = {1, 0, 3, 0, 5, 0, 7, 0, 9};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r4) );

}


TEST( RefClasses, StepRow )
{
	const index_t len = 5;
	const index_t step = 2;
	const index_t ulen = 9;

	double src[ulen] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	double dst[ulen] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

	CStepVector<double, HorzDir> ccol(src, len, step);
	StepVector<double, HorzDir> col(dst, len, step);

	ASSERT_TRUE( verify_stepvec(ccol, len, step) );
	ASSERT_TRUE( verify_stepvec(col, len, step) );

	col.fill(0.0);

	double r1[ulen] = {0, 8, 0, 6, 0, 4, 0, 2, 0};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r1) );

	col.copy_from(src);

	double r2[ulen] = {1, 8, 2, 6, 3, 4, 4, 2, 5};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r2) );

	zero_elems(size_t(ulen), dst);
	col = RefRow<double>(src, len);
	col = RefCol<double>(src, len);

	double r3[ulen] = {1, 0, 2, 0, 3, 0, 4, 0, 5};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r3) );

	zero_elems(size_t(ulen), dst);
	col = ccol;

	double r4[ulen] = {1, 0, 3, 0, 5, 0, 7, 0, 9};
	ASSERT_TRUE( elems_equal(size_t(ulen), dst, r4) );

}



