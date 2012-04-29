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


static inline double complex_formula1(double x, double y, double z, double w)
{
	using bcs::math::sqr;

	double left = sqr(sqr(x + y));
	double right = sqr(sqr(z)) + sqr(sqr(w));

	return left + right;
}


TEST( MatrixEWise, ComplexFormulaToCol )
{
	const index_t n = 32;

	const double yv = 2;
	const double zv = 3;

	col_f64 x(n); for (index_t i = 0; i < n; ++i) x[i] = i + 1;
	col_f64 w(n); for (index_t i = 0; i < n; ++i) w[i] = (2*i + 1);

	MyConstMat y(n, 1, yv);
	MyConstMat z(n, 1, zv);

	col_f64 r0(n);
	for (index_t i = 0; i < n; ++i) r0[i] = complex_formula1(x[i], yv, zv, w[i]);

	col_f64 r = sqr(sqr(x + y)) + (sqr(sqr(z)) + sqr(sqr(w)));

	ASSERT_EQ(n, r.nrows());
	ASSERT_EQ(1, r.ncolumns());
	ASSERT_TRUE( is_equal(r, r0) );

	col_f64 r1;
	r1 = sqr(sqr(x + y)) + (sqr(sqr(z)) + sqr(sqr(w)));

	ASSERT_EQ(n, r1.nrows());
	ASSERT_EQ(1, r1.ncolumns());
	ASSERT_TRUE( is_equal(r1, r0) );
}


TEST( MatrixEWise, ComplexFormulaToRow )
{
	const index_t n = 32;

	const double yv = 2;
	const double zv = 3;

	row_f64 x(n); for (index_t i = 0; i < n; ++i) x[i] = i + 1;
	row_f64 w(n); for (index_t i = 0; i < n; ++i) w[i] = (2*i + 1);

	MyConstMat y(1, n, yv);
	MyConstMat z(1, n, zv);

	row_f64 r0(n);
	for (index_t i = 0; i < n; ++i) r0[i] = complex_formula1(x[i], yv, zv, w[i]);

	row_f64 r = sqr(sqr(x + y)) + (sqr(sqr(z)) + sqr(sqr(w)));

	ASSERT_EQ(1, r.nrows());
	ASSERT_EQ(n, r.ncolumns());
	ASSERT_TRUE( is_equal(r, r0) );
}


TEST( MatrixEWise, ComplexFormulaToRefCol )
{
	const index_t n = 32;

	const double yv = 2;
	const double zv = 3;

	col_f64 x(n); for (index_t i = 0; i < n; ++i) x[i] = i + 1;
	col_f64 w(n); for (index_t i = 0; i < n; ++i) w[i] = (2*i + 1);

	MyConstMat y(n, 1, yv);
	MyConstMat z(n, 1, zv);

	col_f64 r0(n);
	for (index_t i = 0; i < n; ++i) r0[i] = complex_formula1(x[i], yv, zv, w[i]);

	scoped_block<double> r_blk(n, 0.0);
	ref_col<double> r(r_blk.ptr_begin(), n);
	r = sqr(sqr(x + y)) + (sqr(sqr(z)) + sqr(sqr(w)));

	ASSERT_EQ(n, r.nrows());
	ASSERT_EQ(1, r.ncolumns());
	ASSERT_TRUE( is_equal(r, r0) );

}


TEST( MatrixEWise, ComplexFormulaToRefRow )
{
	const index_t n = 32;

	const double yv = 2;
	const double zv = 3;

	row_f64 x(n); for (index_t i = 0; i < n; ++i) x[i] = i + 1;
	row_f64 w(n); for (index_t i = 0; i < n; ++i) w[i] = (2*i + 1);

	MyConstMat y(1, n, yv);
	MyConstMat z(1, n, zv);

	row_f64 r0(n);
	for (index_t i = 0; i < n; ++i) r0[i] = complex_formula1(x[i], yv, zv, w[i]);

	scoped_block<double> r_blk(n, 0.0);
	ref_row<double> r(r_blk.ptr_begin(), n);
	r = sqr(sqr(x + y)) + (sqr(sqr(z)) + sqr(sqr(w)));

	ASSERT_EQ(1, r.nrows());
	ASSERT_EQ(n, r.ncolumns());
	ASSERT_TRUE( is_equal(r, r0) );

}


TEST( MatrixEWise, ComplexFormulaToRefMatEx)
{
	const index_t m = 32;
	const index_t n = 8;
	const index_t ldim = 48;

	const double yv = 2;
	const double zv = 3;

	mat_f64 x(m, n); for (index_t i = 0; i < m * n; ++i) x[i] = i + 1;
	mat_f64 w(m, n); for (index_t i = 0; i < m * n; ++i) w[i] = (2*i + 1);

	MyConstMat y(m, n, yv);
	MyConstMat z(m, n, zv);

	mat_f64 r0(m, n);
	for (index_t i = 0; i < m * n; ++i) r0[i] = complex_formula1(x[i], yv, zv, w[i]);

	scoped_block<double> r_blk(ldim * n, 0.0);
	ref_matrix_ex<double> r(r_blk.ptr_begin(), m, n, ldim);
	r = sqr(sqr(x + y)) + (sqr(sqr(z)) + sqr(sqr(w)));

	ASSERT_EQ(m, r.nrows());
	ASSERT_EQ(n, r.ncolumns());
	ASSERT_TRUE( is_equal(r, r0) );

}










