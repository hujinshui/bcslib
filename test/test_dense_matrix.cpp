/**
 * @file test_dense_matrix.cpp
 *
 * Unit testing for dense matrices
 *
 * @author Dahua Lin
 */

#include "bcs_test_basics.h"

#include <bcslib/matrix/dense_matrix.h>

using namespace bcs;
using namespace bcs::test;

// explicit template for syntax check

template class bcs::RefMatrix<double, DynamicDim, DynamicDim>;
template class bcs::RefMatrix<double, DynamicDim, 1>;
template class bcs::RefMatrix<double, 1, DynamicDim>;
template class bcs::RefMatrix<double, 2, 2>;

template class bcs::DenseMatrix<double, DynamicDim, DynamicDim>;
template class bcs::DenseMatrix<double, DynamicDim, 1>;
template class bcs::DenseMatrix<double, 1, DynamicDim>;
template class bcs::DenseMatrix<double, 2, 2>;

template class bcs::DenseRow<double, DynamicDim>;
template class bcs::DenseRow<double, 3>;
template class bcs::DenseCol<double, DynamicDim>;
template class bcs::DenseCol<double, 3>;


template<class Derived, typename T>
bool verify_dense_matrix(const IDenseMatrix<Derived, T>& A, index_t m, index_t n)
{
	if (!(A.nrows() == m)) return false;
	if (!(A.ncolumns() == n)) return false;
	if (!(A.nelems() == m * n)) return false;
	if (!(A.size() == size_t(m * n))) return false;
	if (!(A.is_empty() == (m == 0 || n == 0))) return false;
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

	return true;
}


TEST( MatrixBasics, RefMatrix)
{
	const index_t m = 2;
	const index_t n = 3;

	double src[m * n] = {1, 2, 3, 4, 5, 6};
	double dst[m * n] = {0, 0, 0, 0, 0, 0};

	bcs_cref_mat(double, S, src, m, n);
	bcs_ref_mat(double, D, dst, m, n);

	ASSERT_TRUE( verify_dense_matrix(S, m, n) );
	ASSERT_TRUE( verify_dense_matrix(D, m, n) );

	ASSERT_TRUE( S.ptr_base() == src );
	ASSERT_TRUE( D.ptr_base() == dst );

	D.fill(7.0);
	ASSERT_TRUE( elems_equal(6, dst, 7.0) );

	D.copy_from(src);
	ASSERT_TRUE( elems_equal(6, dst, src) );
}


TEST( MatrixBasics, RefVector )
{
	const index_t len = 6;

	double src[len] = {1, 2, 3, 4, 5, 6};
	double dst[len] = {0, 0, 0, 0, 0, 0};

	bcs_cref_col(double, S, src, len);
	bcs_ref_col(double, D, dst, len);

	ASSERT_TRUE( verify_dense_matrix(S, len, 1) );
	ASSERT_TRUE( verify_dense_matrix(D, len, 1) );

	ASSERT_TRUE( S.ptr_base() == src );
	ASSERT_TRUE( D.ptr_base() == dst );

	D.fill(7.0);
	ASSERT_TRUE( elems_equal(6, dst, 7.0) );

	D.copy_from(src);
	ASSERT_TRUE( elems_equal(6, dst, src) );

	D.zero();
	ASSERT_TRUE( elems_equal(6, dst, 0.0) );

	bcs_cref_row(double, Sr, src, len);
	bcs_ref_row(double, Dr, dst, len);

	ASSERT_TRUE( verify_dense_matrix(Sr, 1, len) );
	ASSERT_TRUE( verify_dense_matrix(Dr, 1, len) );

	ASSERT_TRUE( Sr.ptr_base() == src );
	ASSERT_TRUE( Dr.ptr_base() == dst );

	Dr.fill(7.0);
	ASSERT_TRUE( elems_equal(6, dst, 7.0) );

	Dr.copy_from(src);
	ASSERT_TRUE( elems_equal(6, dst, src) );
}



TEST( MatrixBasics, Matrix_DynRowDim_DynColDim )
{
	mat_f64 A0;

	ASSERT_TRUE( verify_dense_matrix(A0, (index_t)0, (index_t)0) );

	index_t m1 = 2, n1 = 3;
	mat_f64 A1(m1, n1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	mat_f64 A1a(m1, n1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	mat_f64 A2(m1, n1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, m1, n1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_same_size(A1, A2) );
	ASSERT_FALSE( is_same_size(A1, A0) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	mat_f64 A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_f64, double>& A2_cref = A2;
	mat_f64 A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, Matrix_DynRowDim_StaColDym )
{
	typedef DenseMatrix<double, DynamicDim, 3> mat_t;

	mat_t A0;

	ASSERT_TRUE( verify_dense_matrix(A0, (index_t)0, (index_t)3) );

	index_t m1 = 2, n1 = 3;
	mat_t A1(m1, n1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	mat_t A1a(m1, n1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	mat_t A2(m1, n1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, m1, n1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_same_size(A1, A2) );
	ASSERT_FALSE( is_same_size(A1, A0) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	mat_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_t, double>& A2_cref = A2;
	mat_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, Matrix_StaRowDim_DynColDym )
{
	typedef DenseMatrix<double, 2, DynamicDim> mat_t;

	mat_t A0;

	ASSERT_TRUE( verify_dense_matrix(A0, (index_t)2, (index_t)0) );

	index_t m1 = 2, n1 = 3;
	mat_t A1(m1, n1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	mat_t A1a(m1, n1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	mat_t A2(m1, n1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, m1, n1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_same_size(A1, A2) );
	ASSERT_FALSE( is_same_size(A1, A0) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	mat_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_t, double>& A2_cref = A2;
	mat_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, Matrix_StaRowDim_StaColDym )
{
	typedef mat23_f64 mat_t;

	mat_t A0;

	ASSERT_EQ( 2, A0.nrows() );
	ASSERT_EQ( 3, A0.ncolumns() );

	index_t m1 = 2, n1 = 3;
	mat_t A1(m1, n1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	mat_t A1a(m1, n1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, m1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	mat_t A2(m1, n1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, m1, n1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	mat_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_t, double>& A2_cref = A2;
	mat_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, m1, n1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );

	mat_f64 B = A2;
	ASSERT_TRUE( verify_dense_matrix(B, m1, n1) );
	ASSERT_TRUE( is_equal(B, A2) );
}



TEST( MatrixBasics, ColVec_DynDim )
{
	typedef DenseMatrix<double, DynamicDim, 1> mat_bt;
	typedef col_f64 vec_t;

	vec_t A0;

	ASSERT_TRUE( verify_dense_matrix(A0, 0, 1) );

	index_t m1 = 6;
	vec_t A1(m1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, m1, 1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	vec_t A1a(m1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, m1, 1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	vec_t A2(m1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, m1, 1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_same_size(A1, A2) );
	ASSERT_FALSE( is_same_size(A1, A0) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	vec_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, m1, 1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_bt, double>& A2_cref = A2;
	vec_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, m1, 1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, ColVec_StaDim )
{
	typedef DenseMatrix<double, 6, 1> mat_bt;
	typedef DenseCol<double, 6> vec_t;

	vec_t A0;

	ASSERT_EQ(6, A0.nrows());
	ASSERT_EQ(1, A0.ncolumns());

	index_t m1 = 6;
	vec_t A1(m1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, m1, 1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	vec_t A1a(m1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, m1, 1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	vec_t A2(m1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, m1, 1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	vec_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, m1, 1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_bt, double>& A2_cref = A2;
	vec_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, m1, 1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, RowVec_DynDim )
{
	typedef DenseMatrix<double, 1, DynamicDim> mat_bt;
	typedef row_f64 vec_t;

	vec_t A0;

	ASSERT_TRUE( verify_dense_matrix(A0, 1, 0) );

	index_t n1 = 6;
	vec_t A1(n1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, 1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	vec_t A1a(n1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, 1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	vec_t A2(n1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, 1, n1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_same_size(A1, A2) );
	ASSERT_FALSE( is_same_size(A1, A0) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	vec_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, 1, n1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_bt, double>& A2_cref = A2;
	vec_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, 1, n1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, RowVec_StaDim )
{
	typedef DenseMatrix<double, 1, 6> mat_bt;
	typedef DenseRow<double, 6> vec_t;

	vec_t A0;

	ASSERT_EQ(1, A0.nrows());
	ASSERT_EQ(6, A0.ncolumns());

	index_t n1 = 6;
	vec_t A1(n1);
	A1.fill(2.4);

	const double r1[6] = {2.4, 2.4, 2.4, 2.4, 2.4, 2.4};

	ASSERT_TRUE( verify_dense_matrix(A1, 1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1.ptr_base(), r1) );

	vec_t A1a(n1, 2.4);

	ASSERT_TRUE( verify_dense_matrix(A1a, 1, n1) );
	ASSERT_TRUE( elems_equal(A1.size(), A1a.ptr_base(), r1) );

	const double r2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};

	vec_t A2(n1, r2);

	ASSERT_TRUE( verify_dense_matrix(A2, 1, n1) );
	ASSERT_TRUE( elems_equal(A2.size(), A2.ptr_base(), r2) );

	ASSERT_TRUE( is_equal(A1, A1a) );
	ASSERT_FALSE( is_equal(A1, A2) );
	ASSERT_FALSE( is_equal(A1, A0) );

	vec_t A2c(A2);

	ASSERT_TRUE( verify_dense_matrix(A2c, 1, n1) );
	ASSERT_TRUE( is_equal(A2, A2c) );
	ASSERT_FALSE( A2.ptr_base() == A2c.ptr_base() );

	const IMatrixBase<mat_bt, double>& A2_cref = A2;
	vec_t A2e(A2_cref);

	ASSERT_TRUE( verify_dense_matrix(A2e, 1, n1) );
	ASSERT_TRUE( is_equal(A2, A2e) );
	ASSERT_FALSE( A2.ptr_base() == A2e.ptr_base() );
}


TEST( MatrixBasics, Resize )
{
	mat_f64 A(2, 3);
	ASSERT_EQ(2, A.nrows());
	ASSERT_EQ(3, A.ncolumns());

	const double *pa = A.ptr_base();
	A.resize(3, 2);
	ASSERT_EQ(3, A.nrows());
	ASSERT_EQ(2, A.ncolumns());
	ASSERT_TRUE( pa == A.ptr_base() );

	A.resize(5, 4);
	ASSERT_EQ(5, A.nrows());
	ASSERT_EQ(4, A.ncolumns());
	ASSERT_FALSE( pa == A.ptr_base() );

	A.resize(0, 0);
	ASSERT_EQ(0, A.nrows());
	ASSERT_EQ(0, A.ncolumns());
	ASSERT_TRUE( A.ptr_base() == BCS_NULL );


	col_f64 B(5);
	ASSERT_EQ(5, B.nrows());
	ASSERT_EQ(1, B.ncolumns());

	const double *pb = B.ptr_base();
	B.resize(5);
	ASSERT_EQ(5, B.nrows());
	ASSERT_EQ(1, B.ncolumns());
	ASSERT_TRUE( pb == B.ptr_base() );

	B.resize(3);
	ASSERT_EQ(3, B.nrows());
	ASSERT_EQ(1, B.ncolumns());

	B.resize(0);
	ASSERT_EQ(0, B.nrows());
	ASSERT_EQ(1, B.ncolumns());
	ASSERT_TRUE( B.ptr_base() == BCS_NULL );


	row_f64 C(5);
	ASSERT_EQ(1, C.nrows());
	ASSERT_EQ(5, C.ncolumns());

	const double *pc = C.ptr_base();
	C.resize(5);
	ASSERT_EQ(1, C.nrows());
	ASSERT_EQ(5, C.ncolumns());
	ASSERT_TRUE( pc == C.ptr_base() );

	C.resize(3);
	ASSERT_EQ(1, C.nrows());
	ASSERT_EQ(3, C.ncolumns());

	C.resize(0);
	ASSERT_EQ(1, C.nrows());
	ASSERT_EQ(0, C.ncolumns());
	ASSERT_TRUE( C.ptr_base() == BCS_NULL );
}







