/**
 * @file test_matrix_ewise.cpp
 *
 * Unit testing for element-wise expression evaluation
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


/************************************************
 *
 *  A simple but full-fledged class for testing
 *
 ************************************************/

class MyConstMat;

template<>
struct matrix_traits<MyConstMat>
{
	static const int num_dimensions = 2;
	static const int compile_time_num_rows = DynamicDim;
	static const int compile_time_num_cols = DynamicDim;

	static const bool is_linear_indexable = false;
	static const bool is_continuous = false;
	static const bool is_sparse = false;
	static const bool is_readonly = true;

	typedef double value_type;
	typedef index_t index_type;
};

class MyConstMat : public IMatrixXpr<MyConstMat, double>
{
public:
	BCS_MAT_TRAITS_DEFS(double)

	MyConstMat(index_type m, index_type n, const double& v)
	: m_nrows(m), m_ncols(n), m_scalar(v) { }

	BCS_ENSURE_INLINE index_type nelems() const
	{
		return m_nrows * m_ncols;
	}

	BCS_ENSURE_INLINE size_type size() const
	{
		return static_cast<size_type>(nelems());
	}

	BCS_ENSURE_INLINE index_type nrows() const
	{
		return m_nrows;
	}

	BCS_ENSURE_INLINE index_type ncolumns() const
	{
		return m_ncols;
	}

	BCS_ENSURE_INLINE
	double scalar() const { return m_scalar; }

private:
	index_type m_nrows;
	index_type m_ncols;
	double m_scalar;
};


template<>
struct expr_optimizer<MyConstMat>
{
	typedef MyConstMat result_expr_type;

	BCS_ENSURE_INLINE
	static const result_expr_type& optimize(const MyConstMat& e)
	{
		return e;
	}
};


template<>
struct expr_evaluator<MyConstMat>
{
	template<class DMap>
	BCS_ENSURE_INLINE
	static void evaluate(const MyConstMat& e, IRegularMatrix<DMap, double>& dst)
	{
		fill(dst, e.scalar());
	}
};




/************************************************
 *
 *  Test cases
 *
 ************************************************/


TEST( MatrixEWise, UnaryforDenseMat )
{
	const index_t m = 3;
	const index_t n = 4;

	mat_f64 A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = (i+1);

	mat_f64 C0(m, n);
	for (index_t i = 0; i < m * n; ++i) C0[i] = (i+1) * (i+1);

	mat_f64 C = sqr(A);

	ASSERT_EQ(m, C.nrows());
	ASSERT_EQ(n, C.ncolumns());

	ASSERT_TRUE( is_equal(C, C0) );

	mat_f64 C1;
	C1 = sqr(A);

	ASSERT_EQ(m, C1.nrows());
	ASSERT_EQ(n, C1.ncolumns());

	ASSERT_TRUE( is_equal(C1, C0) );

	mat_f64 C2(m, n, 0.0);
	const double *p2 = C2.ptr_data();
	C2 = sqr(A);

	ASSERT_EQ(m, C2.nrows());
	ASSERT_EQ(n, C2.ncolumns());
	ASSERT_TRUE( C2.ptr_data() == p2 );
	ASSERT_TRUE( is_equal(C2, C0) );
}


TEST( MatrixEWise, UnaryforNewType )
{
	const index_t m = 3;
	const index_t n = 4;

	double cval = 3;
	MyConstMat A(m, n, cval);

	mat_f64 B(A);
	ASSERT_EQ(m, B.nrows());
	ASSERT_EQ(n, B.ncolumns());
	ASSERT_TRUE( elems_equal(m * n, B.ptr_data(), cval) );

	mat_f64 C0(m, n);
	for (index_t i = 0; i < m * n; ++i) C0[i] = cval * cval;

	mat_f64 C = sqr(A);

	ASSERT_EQ(m, C.nrows());
	ASSERT_EQ(n, C.ncolumns());

	ASSERT_TRUE( is_equal(C, C0) );

	mat_f64 C1;
	C1 = sqr(A);

	ASSERT_EQ(m, C1.nrows());
	ASSERT_EQ(n, C1.ncolumns());

	ASSERT_TRUE( is_equal(C1, C0) );

	mat_f64 C2(m, n, 0.0);
	const double *p2 = C2.ptr_data();
	C2 = sqr(A);

	ASSERT_EQ(m, C2.nrows());
	ASSERT_EQ(n, C2.ncolumns());
	ASSERT_TRUE( C2.ptr_data() == p2 );
	ASSERT_TRUE( is_equal(C2, C0) );
}


TEST( MatrixEWise, BinaryforDenseMatWithDenseMat )
{
	const index_t m = 3;
	const index_t n = 4;

	mat_f64 A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = (i+1);

	mat_f64 B(m, n);
	for (index_t i = 0; i < m * n; ++i) B[i] = (i+1) * 2;

	mat_f64 C0(m, n);
	for (index_t i = 0; i < m * n; ++i) C0[i] = (i+1) * 3;

	mat_f64 C = A + B;

	ASSERT_EQ(m, C.nrows());
	ASSERT_EQ(n, C.ncolumns());

	ASSERT_TRUE( is_equal(C, C0) );

	mat_f64 C1;
	C1 = A + B;

	ASSERT_EQ(m, C1.nrows());
	ASSERT_EQ(n, C1.ncolumns());

	ASSERT_TRUE( is_equal(C1, C0) );

	mat_f64 C2(m, n, 0.0);
	const double *p2 = C2.ptr_data();
	C2 = A + B;

	ASSERT_EQ(m, C2.nrows());
	ASSERT_EQ(n, C2.ncolumns());
	ASSERT_TRUE( C2.ptr_data() == p2 );
	ASSERT_TRUE( is_equal(C2, C0) );
}


TEST( MatrixEWise, BinaryforDenseMatWithNewMat )
{
	const index_t m = 3;
	const index_t n = 4;

	mat_f64 A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = (i+1);

	const double cval = 3.0;
	MyConstMat B(m, n, cval);

	mat_f64 C0(m, n);
	for (index_t i = 0; i < m * n; ++i) C0[i] = (i+1) + cval;

	mat_f64 C = A + B;

	ASSERT_EQ(m, C.nrows());
	ASSERT_EQ(n, C.ncolumns());

	ASSERT_TRUE( is_equal(C, C0) );

	mat_f64 C1;
	C1 = A + B;

	ASSERT_EQ(m, C1.nrows());
	ASSERT_EQ(n, C1.ncolumns());

	ASSERT_TRUE( is_equal(C1, C0) );

	mat_f64 C2(m, n, 0.0);
	const double *p2 = C2.ptr_data();
	C2 = A + B;

	ASSERT_EQ(m, C2.nrows());
	ASSERT_EQ(n, C2.ncolumns());
	ASSERT_TRUE( C2.ptr_data() == p2 );
	ASSERT_TRUE( is_equal(C2, C0) );
}


TEST( MatrixEWise, BinaryforNewMatWithDenseMat )
{
	const index_t m = 3;
	const index_t n = 4;

	const double cval = 3.0;
	MyConstMat A(m, n, cval);

	mat_f64 B(m, n);
	for (index_t i = 0; i < m * n; ++i) B[i] = (i+1) * 2;

	mat_f64 C0(m, n);
	for (index_t i = 0; i < m * n; ++i) C0[i] = (i+1) * 2 + cval;

	mat_f64 C = A + B;

	ASSERT_EQ(m, C.nrows());
	ASSERT_EQ(n, C.ncolumns());

	ASSERT_TRUE( is_equal(C, C0) );

	mat_f64 C1;
	C1 = A + B;

	ASSERT_EQ(m, C1.nrows());
	ASSERT_EQ(n, C1.ncolumns());

	ASSERT_TRUE( is_equal(C1, C0) );

	mat_f64 C2(m, n, 0.0);
	const double *p2 = C2.ptr_data();
	C2 = A + B;

	ASSERT_EQ(m, C2.nrows());
	ASSERT_EQ(n, C2.ncolumns());
	ASSERT_TRUE( C2.ptr_data() == p2 );
	ASSERT_TRUE( is_equal(C2, C0) );
}


TEST( MatrixEWise, BinaryforNewMatWithNewMat )
{
	const index_t m = 3;
	const index_t n = 4;

	const double cval = 3.0;
	MyConstMat A(m, n, cval);
	MyConstMat B(m, n, cval * 2);

	mat_f64 C0(m, n);
	for (index_t i = 0; i < m * n; ++i) C0[i] = cval * 3;

	mat_f64 C = A + B;

	ASSERT_EQ(m, C.nrows());
	ASSERT_EQ(n, C.ncolumns());

	ASSERT_TRUE( is_equal(C, C0) );

	mat_f64 C1;
	C1 = A + B;

	ASSERT_EQ(m, C1.nrows());
	ASSERT_EQ(n, C1.ncolumns());
	ASSERT_TRUE( is_equal(C1, C0) );

	mat_f64 C2(m, n, 0.0);
	const double *p2 = C2.ptr_data();
	C2 = A + B;

	ASSERT_EQ(m, C2.nrows());
	ASSERT_EQ(n, C2.ncolumns());
	ASSERT_TRUE( C2.ptr_data() == p2 );
	ASSERT_TRUE( is_equal(C2, C0) );
}







