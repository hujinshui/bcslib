/**
 * @file test_array2d.cpp
 *
 * The Unit Testing for array2d
 * 
 * @author Dahua Lin
 */


#include "bcs_test_basics.h"
#include <bcslib/array/array2d.h>

using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::caview2d_ex<double, row_major_t,    step_ind, step_ind>;
template class bcs::caview2d_ex<double, column_major_t, step_ind, step_ind>;
template class bcs::aview2d_ex<double,  row_major_t,    step_ind, step_ind>;
template class bcs::aview2d_ex<double,  column_major_t, step_ind, step_ind>;

template class bcs::caview2d_ex<double, row_major_t,    id_ind, step_ind>;
template class bcs::caview2d_ex<double, column_major_t, id_ind, step_ind>;
template class bcs::aview2d_ex<double,  row_major_t,    id_ind, step_ind>;
template class bcs::aview2d_ex<double,  column_major_t, id_ind, step_ind>;

template class bcs::caview2d<double, row_major_t>;
template class bcs::caview2d<double, column_major_t>;

template class bcs::aview2d<double, row_major_t>;
template class bcs::aview2d<double, column_major_t>;

template class bcs::array2d<double, row_major_t>;
template class bcs::array2d<double, column_major_t>;


/************************************************
 *
 *  Inheritance testing
 *
 ************************************************/

// caview2d_ex

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview_base<bcs::caview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::caview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview2d_base<bcs::caview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::caview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_caview2d_base<bcs::caview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::caview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

// aview2d_ex

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview_base<bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::aview_base<bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview2d_base<bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::aview2d_base<bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_caview2d_base<bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_aview2d_base<bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >,
		bcs::aview2d_ex<double, row_major_t, id_ind, step_ind> >::value) );

// caview2d

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview_base<bcs::caview2d<double, row_major_t> >,
		bcs::caview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_caview_base<bcs::caview2d<double, row_major_t> >,
		bcs::caview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview2d_base<bcs::caview2d<double, row_major_t> >,
		bcs::caview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_caview2d_base<bcs::caview2d<double, row_major_t> >,
		bcs::caview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::block_caview2d_base<bcs::caview2d<double, row_major_t> >,
		bcs::caview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_caview2d_base<bcs::caview2d<double, row_major_t> >,
		bcs::caview2d<double, row_major_t> >::value) );

// aview2d

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::aview_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_caview_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_aview_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::aview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_caview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_aview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::block_caview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::block_aview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_caview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_aview2d_base<bcs::aview2d<double, row_major_t> >,
		bcs::aview2d<double, row_major_t> >::value) );

// array2d

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::aview_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_caview_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_aview_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::caview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::aview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_caview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::dense_aview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::block_caview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::block_aview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_caview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );

BCS_STATIC_ASSERT( (is_base_of<
		bcs::continuous_aview2d_base<bcs::array2d<double, row_major_t> >,
		bcs::array2d<double, row_major_t> >::value) );



template<typename T>
void dummy_a2(const T& ) { }

inline void do_on_view2(const caview2d<double, column_major_t>& v) { }

void syntax_check_arr2()
{
	typedef bcs::array2d<double, column_major_t> dt;

	dt a(3, 5);
	const dt& ca = a;

	caview_base<dt>& r1 = a;
	const dt& d1 = r1.derived();
	dummy_a2(d1);

	aview_base<dt>& r2 = a;
	const dt& d2 = r2.derived();
	dummy_a2(d2);

	continuous_caview_base<dt>& r3 = a;
	const dt& d3 = r3.derived();
	dummy_a2(d3);

	continuous_aview_base<dt>& r4 = a;
	const dt& d4 = r4.derived();
	dummy_a2(d4);

	caview2d_base<dt>& r5 = a;
	const dt& d5 = r5.derived();
	dummy_a2(d5);

	aview2d_base<dt>& r6 = a;
	const dt& d6 = r6.derived();
	dummy_a2(d6);

	dense_caview2d_base<dt>& r7 = a;
	const dt& d7 = r7.derived();
	dummy_a2(d7);

	dense_aview2d_base<dt>& r8 = a;
	const dt& d8 = r8.derived();
	dummy_a2(d8);

	block_caview2d_base<dt>& r9 = a;
	const dt& d9 = r9.derived();
	dummy_a2(d9);

	block_aview2d_base<dt>& r10 = a;
	const dt& d10 = r10.derived();
	dummy_a2(d10);

	continuous_caview2d_base<dt>& r11 = a;
	const dt& d11 = r11.derived();
	dummy_a2(d11);

	continuous_aview2d_base<dt>& r12 = a;
	const dt& d12 = r12.derived();
	dummy_a2(d12);

	dummy_a2(a.ndims());
	dummy_a2(a.dim0());
	dummy_a2(a.dim1());
	dummy_a2(a.nrows());
	dummy_a2(a.ncolumns());
	dummy_a2(a.size());
	dummy_a2(a.nelems());
	dummy_a2(a.shape());
	dummy_a2(a.pbase());
	dummy_a2(a(0, 0));
	dummy_a2(a[0]);
	dummy_a2(ca(0, 0));
	dummy_a2(ca[0]);
	dummy_a2(begin(a));
	dummy_a2(end(a));
	dummy_a2(begin(ca));
	dummy_a2(end(ca));

	do_on_view2(a);
}


/************************************************
 *
 *  Auxiliary functions
 *
 ************************************************/

template<class Derived>
inline index_t sub2ind(const caview2d_base<Derived>& a, index_t i, index_t j)
{
	index_t m = a.nrows();
	index_t n = a.ncolumns();

	return is_same<typename Derived::layout_order, row_major_t>::value ?
			(i * n + j) : (i + j * m);
}

template<class Derived>
bool array_integrity_test(const bcs::caview2d_base<Derived>& a)
{
	index_t m = a.dim0();
	index_t n = a.dim1();

	if (a.ndims() != 2) return false;
	if (a.nrows() != m) return false;
	if (a.ncolumns() != n) return false;
	if (a.nelems() != m * n) return false;
	if (a.size() != (size_t)(m * n)) return false;
	if (a.shape() != arr_shape(m, n)) return false;
	if (a.is_empty() != (a.nelems() == 0)) return false;

	return true;
}

template<class Derived>
bool cont_array_integrity_test(const bcs::continuous_caview2d_base<Derived>& a)
{
	if (!array_integrity_test(a)) return false;

	index_t m = a.dim0();
	index_t n = a.dim1();
	if (!a.is_empty())
	{
		if (a.pbase() != &(a[0])) return false;
	}

	typename Derived::flatten_cview_type fview = a.flatten();

	if (fview.pbase() != a.pbase()) return false;
	if (fview.nelems() != m * n) return false;

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			index_t idx = sub2ind(a, i, j);

			if (&(a(i, j)) != &(a[idx])) return false;
			if (a(i, j) != a[idx]) return false;
			if (a[idx] != fview[idx]) return false;
		}
	}

	return true;
}


template<class Derived>
bool elemwise_operation_test(bcs::dense_aview2d_base<Derived>& a)
{
	typedef typename Derived::value_type T;

	index_t m = a.nrows();
	index_t n = a.ncolumns();
	block<T> blk(a.size());
	const T *b = blk.pbase();

	// export
	export_to(a, blk.pbase());

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (b[sub2ind(a, i, j)] != a(i, j)) return false;
		}
	}

	// fill
	T v = T(123);
	fill(a, v);

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (a(i, j) != v) return false;
		}
	}

	// import
	import_from(a, blk.pbase());

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (a(i, j) != b[sub2ind(a, i, j)]) return false;
		}
	}

	return true;
}


/**********************************************************
 *
 *  Test Cases
 *
 **********************************************************/

TEST( Array2D, Aview2DRowMajor )
{
	double src[6] = {3, 4, 5, 1, 2, 7};
	index_t m = 2;
	index_t n = 3;

	aview2d<double, row_major_t> a1(src, m, n);

	ASSERT_EQ(a1.dim0(), m);
	ASSERT_EQ(a1.dim1(), n);
	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	aview2d<double, row_major_t> a2(a1);

	ASSERT_EQ(a2.dim0(), m);
	ASSERT_EQ(a2.dim1(), n);
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal(a2, src, m * n) );

	ASSERT_EQ( a1.pbase(), a2.pbase() );
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal(a2, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a2) );
}


TEST( Array2D, Aview2DColumnMajor )
{
	double src[6] = {3, 4, 5, 1, 2, 7};
	index_t m = 2;
	index_t n = 3;

	aview2d<double, column_major_t> a1(src, m, n);

	ASSERT_EQ(a1.dim0(), m);
	ASSERT_EQ(a1.dim1(), n);
	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	aview2d<double, column_major_t> a2(a1);

	ASSERT_EQ(a2.dim0(), m);
	ASSERT_EQ(a2.dim1(), n);
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal(a2, src, m * n) );

	ASSERT_EQ( a1.pbase(), a2.pbase() );
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal(a2, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a2) );
}


TEST( Array2D, Array2DRowMajor  )
{
	double src[24];
	for (int i = 0; i < 24; ++i) src[i] = i+1;
	index_t k = 0;

	index_t m = 2;
	index_t n = 3;

	double v2 = 7.0;

	// row major

	array2d<double, row_major_t> a0(0, 0);

	ASSERT_EQ( a0.dim0(), 0 );
	ASSERT_EQ( a0.dim1(), 0 );
	ASSERT_TRUE( a0.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a0) );

	array2d<double, row_major_t> a1(m, n);
	k = 0;
	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			a1(i, j) = src[k++];
		}
	}

	ASSERT_EQ( a1.dim0(), m );
	ASSERT_EQ( a1.dim1(), n );
	ASSERT_TRUE( a1.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	array2d<double, row_major_t> a2(m, n, v2);

	ASSERT_EQ( a2.dim0(), m );
	ASSERT_EQ( a2.dim1(), n );
	ASSERT_TRUE( a2.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal_scalar(a2, v2, m * n) );

	array2d<double, row_major_t> a3(m, n, src);

	ASSERT_EQ( a3.dim0(), m );
	ASSERT_EQ( a3.dim1(), n );
	ASSERT_TRUE( a3.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a3) );

	array2d<double, row_major_t> a4(a3);

	ASSERT_EQ(a4.dim0(), m);
	ASSERT_EQ(a4.dim1(), n);
	ASSERT_FALSE( a3.is_unique() );
	ASSERT_FALSE( a4.is_unique() );
	ASSERT_EQ( a4.pbase(), a3.pbase() );

	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );

	ASSERT_TRUE( cont_array_integrity_test(a4) );
	ASSERT_TRUE( array_equal(a4, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a4) );

	a4.make_unique();

	ASSERT_NE( a4.pbase(), a3.pbase() );
	ASSERT_TRUE( a3.is_unique() );
	ASSERT_TRUE( a4.is_unique() );

	ASSERT_EQ(a3.dim0(), m);
	ASSERT_EQ(a3.dim1(), n);
	ASSERT_EQ(a4.dim0(), m);
	ASSERT_EQ(a4.dim1(), n);

	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );

	ASSERT_TRUE( cont_array_integrity_test(a4) );
	ASSERT_TRUE( array_equal(a4, src, m * n) );

	array2d<double, row_major_t> a5 = a1.deep_copy();

	ASSERT_NE( a1.pbase(), a5.pbase() );
	ASSERT_TRUE( a1.is_unique() );
	ASSERT_TRUE( a5.is_unique() );

	ASSERT_EQ(a1.dim0(), m);
	ASSERT_EQ(a1.dim1(), n);
	ASSERT_EQ(a5.dim0(), m);
	ASSERT_EQ(a5.dim1(), n);

	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );

	ASSERT_TRUE( cont_array_integrity_test(a5) );
	ASSERT_TRUE( array_equal(a5, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a5) );
}


TEST( Array2D, Array2DColumnMajor  )
{
	double src[24];
	for (int i = 0; i < 24; ++i) src[i] = i+1;
	index_t k = 0;

	index_t m = 2;
	index_t n = 3;

	double v2 = 7.0;

	// row major

	array2d<double, column_major_t> a0(0, 0);

	ASSERT_EQ( a0.dim0(), 0 );
	ASSERT_EQ( a0.dim1(), 0 );
	ASSERT_TRUE( a0.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a0) );

	array2d<double, column_major_t> a1(m, n);
	k = 0;
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			a1(i, j) = src[k++];
		}
	}

	ASSERT_EQ( a1.dim0(), m );
	ASSERT_EQ( a1.dim1(), n );
	ASSERT_TRUE( a1.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	array2d<double, column_major_t> a2(m, n, v2);

	ASSERT_EQ( a2.dim0(), m );
	ASSERT_EQ( a2.dim1(), n );
	ASSERT_TRUE( a2.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal_scalar(a2, v2, m * n) );

	array2d<double, column_major_t> a3(m, n, src);

	ASSERT_EQ( a3.dim0(), m );
	ASSERT_EQ( a3.dim1(), n );
	ASSERT_TRUE( a3.is_unique() );
	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a3) );

	array2d<double, column_major_t> a4(a3);

	ASSERT_EQ(a4.dim0(), m);
	ASSERT_EQ(a4.dim1(), n);
	ASSERT_FALSE( a3.is_unique() );
	ASSERT_FALSE( a4.is_unique() );
	ASSERT_EQ( a4.pbase(), a3.pbase() );

	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );

	ASSERT_TRUE( cont_array_integrity_test(a4) );
	ASSERT_TRUE( array_equal(a4, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a4) );

	a4.make_unique();

	ASSERT_NE( a4.pbase(), a3.pbase() );
	ASSERT_TRUE( a3.is_unique() );
	ASSERT_TRUE( a4.is_unique() );

	ASSERT_EQ(a3.dim0(), m);
	ASSERT_EQ(a3.dim1(), n);
	ASSERT_EQ(a4.dim0(), m);
	ASSERT_EQ(a4.dim1(), n);

	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );

	ASSERT_TRUE( cont_array_integrity_test(a4) );
	ASSERT_TRUE( array_equal(a4, src, m * n) );

	array2d<double, column_major_t> a5 = a1.deep_copy();

	ASSERT_NE( a1.pbase(), a5.pbase() );
	ASSERT_TRUE( a1.is_unique() );
	ASSERT_TRUE( a5.is_unique() );

	ASSERT_EQ(a1.dim0(), m);
	ASSERT_EQ(a1.dim1(), n);
	ASSERT_EQ(a5.dim0(), m);
	ASSERT_EQ(a5.dim1(), n);

	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );

	ASSERT_TRUE( cont_array_integrity_test(a5) );
	ASSERT_TRUE( array_equal(a5, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a5) );
}



TEST( Array2D, Aview2DExRowMajor )
{
	double src[36];
	for (int i = 0; i < 36; ++i) src[i] = i+1;

	row_extent bext = get_extent(5, 6, row_major_t());
	ASSERT_EQ( bext.value, 6 );

	// id x id

	aview2d_ex<double, row_major_t, id_ind, id_ind> a0(src, bext, id_ind(2), id_ind(3));
	double r0[] = {1, 2, 3, 7, 8, 9};

	ASSERT_EQ(a0.dim0(), 2);
	ASSERT_EQ(a0.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a0) );
	ASSERT_TRUE( array2d_equal(a0, make_caview2d_rm(r0, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a0) );

	// id x step

	aview2d_ex<double, row_major_t, id_ind, step_ind> a1(src, bext, id_ind(2), step_ind(3, 2));
	double r1[] = {1, 3, 5, 7, 9, 11};

	ASSERT_EQ(a1.dim0(), 2);
	ASSERT_EQ(a1.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a1) );
	ASSERT_TRUE( array2d_equal(a1, make_caview2d_rm(r1, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	// step x step

	aview2d_ex<double, row_major_t, step_ind, step_ind> a2(src, bext, step_ind(2, 2), step_ind(3, 2));
	double r2[] = {1, 3, 5, 13, 15, 17};

	ASSERT_EQ(a2.dim0(), 2);
	ASSERT_EQ(a2.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a2) );
	ASSERT_TRUE( array2d_equal(a2, make_caview2d_rm(r2, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a2) );

	// step x rep

	aview2d_ex<double, row_major_t, step_ind, rep_ind> a3(src, bext, step_ind(2, 2), rep_ind(3));
	double r3[] = {1, 1, 1, 13, 13, 13};

	ASSERT_EQ(a3.dim0(), 2);
	ASSERT_EQ(a3.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a3) );
	ASSERT_TRUE( array2d_equal(a3, make_caview2d_rm(r3, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a3) );

	// rep x rep

	aview2d_ex<double, row_major_t, rep_ind, rep_ind> a4(src, bext, rep_ind(2), rep_ind(3));
	double r4[] = {1, 1, 1, 1, 1, 1};

	ASSERT_EQ(a4.dim0(), 2);
	ASSERT_EQ(a4.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a4) );
	ASSERT_TRUE( array2d_equal(a4, make_caview2d_rm(r4, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a4) );
}


TEST( Array2D, Aview2DExColumnMajor )
{
	double src[36];
	for (int i = 0; i < 36; ++i) src[i] = i+1;

	column_extent bext = get_extent(5, 6, column_major_t());
	ASSERT_EQ( bext.value, 5 );

	// id x id

	aview2d_ex<double, column_major_t, id_ind, id_ind> a0(src, bext, id_ind(2), id_ind(3));
	double r0[] = {1, 2, 6, 7, 11, 12};

	ASSERT_EQ(a0.dim0(), 2);
	ASSERT_EQ(a0.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a0) );
	ASSERT_TRUE( array2d_equal(a0, make_caview2d_cm(r0, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a0) );

	// id x step

	aview2d_ex<double, column_major_t, id_ind, step_ind> a1(src, bext, id_ind(2), step_ind(3, 2));
	double r1[] = {1, 2, 11, 12, 21, 22};

	ASSERT_EQ(a1.dim0(), 2);
	ASSERT_EQ(a1.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a1) );
	ASSERT_TRUE( array2d_equal(a1, make_caview2d_cm(r1, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	// step x step

	aview2d_ex<double, column_major_t, step_ind, step_ind> a2(src, bext, step_ind(2, 2), step_ind(3, 2));
	double r2[] = {1, 3, 11, 13, 21, 23};

	ASSERT_EQ(a2.dim0(), 2);
	ASSERT_EQ(a2.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a2) );
	ASSERT_TRUE( array2d_equal(a2, make_caview2d_cm(r2, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a2) );

	// step x rep

	aview2d_ex<double, column_major_t, step_ind, rep_ind> a3(src, bext, step_ind(2, 2), rep_ind(3));
	double r3[] = {1, 3, 1, 3, 1, 3};

	ASSERT_EQ(a3.dim0(), 2);
	ASSERT_EQ(a3.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a3) );
	ASSERT_TRUE( array2d_equal(a3, make_caview2d_cm(r3, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a3) );

	// rep x rep

	aview2d_ex<double, column_major_t, rep_ind, rep_ind> a4(src, bext, rep_ind(2), rep_ind(3));
	double r4[] = {1, 1, 1, 1, 1, 1};

	ASSERT_EQ(a4.dim0(), 2);
	ASSERT_EQ(a4.dim1(), 3);
	ASSERT_TRUE( array_integrity_test(a4) );
	ASSERT_TRUE( array2d_equal(a4, make_caview2d_cm(r4, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a4) );
}


TEST( Array2D, Array2DClone )
{
	double src[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	aview2d<double, row_major_t> view1_rm(src, 3, 5);
	array2d<double, row_major_t> a1_rm = clone_array(view1_rm);

	ASSERT_TRUE( array_integrity_test(a1_rm) );
	ASSERT_TRUE( a1_rm.pbase() != view1_rm.pbase() );
	ASSERT_TRUE( array2d_equal(a1_rm, view1_rm) );

	aview2d_ex<double, row_major_t, step_ind, step_ind> view2_rm = make_aview2d_ex_rm(src, 4, 5, step_ind(2, 2), step_ind(3, 2));
	array2d<double, row_major_t> a2_rm = clone_array(view2_rm);

	double a2_rm_s[6] = {1, 3, 5, 11, 13, 15};
	ASSERT_TRUE( array_integrity_test(a2_rm) );
	ASSERT_TRUE( array2d_equal(a2_rm, make_caview2d_rm(a2_rm_s, 2, 3) ));

	aview2d<double, column_major_t> view1_cm(src, 3, 5);
	array2d<double, column_major_t> a1_cm = clone_array(view1_cm);

	ASSERT_TRUE( array_integrity_test(a1_cm) );
	ASSERT_TRUE( a1_cm.pbase() != view1_cm.pbase() );
	ASSERT_TRUE( array2d_equal(a1_cm, view1_cm) );

	aview2d_ex<double, column_major_t, step_ind, step_ind> view2_cm = make_aview2d_ex_cm(src, 4, 5, step_ind(2, 2), step_ind(3, 2));
	array2d<double, column_major_t> a2_cm = clone_array(view2_cm);

	double a2_cm_s[6] = {1, 3, 9, 11, 17, 19};
	ASSERT_TRUE( array_integrity_test(a2_cm) );
	ASSERT_TRUE( array2d_equal(a2_cm, make_caview2d_cm(a2_cm_s, 2, 3) ));
}


TEST( Array2D, Slices )
{
	double src[60];
	for (int i = 0; i < 60; ++i) src[i] = i+1;

	array2d<double, row_major_t> Arm( aview2d_ex<double, row_major_t, id_ind, id_ind>(
			src, get_extent(7, 8, row_major_t()), id_ind(5), id_ind(6)) );
	array2d<double, column_major_t> Acm( aview2d_ex<double, column_major_t, id_ind, id_ind>(
			src, get_extent(7, 8, column_major_t()), id_ind(5), id_ind(6)) );

	// row

	double r0_rm[6] = {9, 10, 11, 12, 13, 14};
	double r0_cm[6] = {2, 9, 16, 23, 30, 37};

	ASSERT_TRUE( array1d_equal( Arm.row(1), make_caview1d(r0_rm, 6) ) );
	ASSERT_TRUE( array1d_equal( Acm.row(1), make_caview1d(r0_cm, 6) ) );

	// row range

	double r1_rm[4] = {10, 11, 12, 13};
	double r1_cm[4] = {9, 16, 23, 30};

	ASSERT_TRUE( array1d_equal( Arm.row(1, rgn_n(1, 4)), make_caview1d(r1_rm, 4) ) );
	ASSERT_TRUE( array1d_equal( Acm.row(1, rgn_n(1, 4)), make_caview1d(r1_cm, 4) ) );

	double r2_rm[3] = {9, 11, 13};
	double r2_cm[3] = {2, 16, 30};

	ASSERT_TRUE( array1d_equal( Arm.row(1, rgn_n(0, 3, 2)), make_caview1d(r2_rm, 3) ) );
	ASSERT_TRUE( array1d_equal( Acm.row(1, rgn_n(0, 3, 2)), make_caview1d(r2_cm, 3) ) );

	double r3_rm[3] = {11, 11, 11};
	double r3_cm[3] = {16, 16, 16};

	ASSERT_TRUE( array1d_equal( Arm.row(1, rep(2, 3)), make_caview1d(r3_rm, 3) ) );
	ASSERT_TRUE( array1d_equal( Acm.row(1, rep(2, 3)), make_caview1d(r3_cm, 3) ) );

	// column

	double c0_rm[5] = {2, 10, 18, 26, 34};
	double c0_cm[5] = {8, 9, 10, 11, 12};

	ASSERT_TRUE( array1d_equal( Arm.column(1), make_caview1d(c0_rm, 5) ) );
	ASSERT_TRUE( array1d_equal( Acm.column(1), make_caview1d(c0_cm, 5) ) );

	// column range

	double c1_rm[4] = {10, 18, 26, 34};
	double c1_cm[4] = {9, 10, 11, 12};

	ASSERT_TRUE( array1d_equal( Arm.column(1, rgn_n(1, 4)), make_caview1d(c1_rm, 4) ) );
	ASSERT_TRUE( array1d_equal( Acm.column(1, rgn_n(1, 4)), make_caview1d(c1_cm, 4) ) );

	double c2_rm[3] = {2, 18, 34};
	double c2_cm[3] = {8, 10, 12};

	ASSERT_TRUE( array1d_equal( Arm.column(1, rgn_n(0, 3, 2)), make_caview1d(c2_rm, 3) ) );
	ASSERT_TRUE( array1d_equal( Acm.column(1, rgn_n(0, 3, 2)), make_caview1d(c2_cm, 3) ) );

	double c3_rm[3] = {18, 18, 18};
	double c3_cm[3] = {10, 10, 10};

	ASSERT_TRUE( array1d_equal( Arm.column(1, rep(2, 3)), make_caview1d(c3_rm, 3) ) );
	ASSERT_TRUE( array1d_equal( Acm.column(1, rep(2, 3)), make_caview1d(c3_cm, 3) ) );

}


TEST( Array2D, SubViews )
{
	double src0[60];
	for (int i = 0; i < 60; ++i) src0[i] = i+1;

	array2d<double, row_major_t> a0_rm( aview2d_ex<double, row_major_t, id_ind, id_ind>(
			src0, get_extent(7, 8, row_major_t()), id_ind(5), id_ind(6)) );
	array2d<double, column_major_t> a0_cm( aview2d_ex<double, column_major_t, id_ind, id_ind>(
			src0, get_extent(7, 8, column_major_t()), id_ind(5), id_ind(6)) );

	// (whole, whole)

	double a0_rm_s0[] = {
			1, 2, 3, 4, 5, 6,
			9, 10, 11, 12, 13, 14,
			17, 18, 19, 20, 21, 22,
			25, 26, 27, 28, 29, 30,
			33, 34, 35, 36, 37, 38
	};

	double a0_cm_s0[] = {
			1, 2, 3, 4, 5,
			8, 9, 10, 11, 12,
			15, 16, 17, 18, 19,
			22, 23, 24, 25, 26,
			29, 30, 31, 32, 33,
			36, 37, 38, 39, 40
	};

	ASSERT_TRUE( array2d_equal( a0_rm.V(whole(), whole()),  make_caview2d_rm(a0_rm_s0, 5, 6) ));
	ASSERT_TRUE( array2d_equal( a0_cm.V(whole(), whole()),  make_caview2d_cm(a0_cm_s0, 5, 6) ));

	// (range, whole)

	double a0_rm_s1[] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22};
	double a0_cm_s1[] = {1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30, 31, 36, 37, 38};

	ASSERT_TRUE( array2d_equal( a0_rm.V(rgn(0, 3), whole()),  make_caview2d_rm(a0_rm_s1, 3, 6) ));
	ASSERT_TRUE( array2d_equal( a0_cm.V(rgn(0, 3), whole()),  make_caview2d_cm(a0_cm_s1, 3, 6) ));

	// (whole, range)

	double a0_rm_s2[] = {2, 3, 4, 10, 11, 12, 18, 19, 20, 26, 27, 28, 34, 35, 36};
	double a0_cm_s2[] = {8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};

	ASSERT_TRUE( array2d_equal( a0_rm.V(whole(), rgn(1, 4)),  make_caview2d_rm(a0_rm_s2, 5, 3) ));
	ASSERT_TRUE( array2d_equal( a0_cm.V(whole(), rgn(1, 4)),  make_caview2d_cm(a0_cm_s2, 5, 3) ));

	// (range, range)

	double a0_rm_s3[] = {18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37};
	double a0_cm_s3[] = {10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33};

	ASSERT_TRUE( array2d_equal( a0_rm.V(rgn(2, 5), rgn(1, 5)),  make_caview2d_rm(a0_rm_s3, 3, 4) ));
	ASSERT_TRUE( array2d_equal( a0_cm.V(rgn(2, 5), rgn(1, 5)),  make_caview2d_cm(a0_cm_s3, 3, 4) ));

	// (range, step_range)

	double a0_rm_s4[] = {18, 20, 22, 26, 28, 30, 34, 36, 38};
	double a0_cm_s4[] = {10, 11, 12, 24, 25, 26, 38, 39, 40};

	ASSERT_TRUE( array2d_equal(a0_rm.V(rgn(2, 5), rgn(1, 6, 2)),  make_caview2d_rm(a0_rm_s4, 3, 3) ));
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(2, 5), rgn(1, 6, 2)),  make_caview2d_cm(a0_cm_s4, 3, 3) ));

	// (step_range, step_range)

	double a0_rm_s5[] = {2, 4, 6, 18, 20, 22, 34, 36, 38};
	double a0_cm_s5[] = {8, 10, 12, 22, 24, 26, 36, 38, 40};

	ASSERT_TRUE( array2d_equal(a0_rm.V(rgn(0, 5, 2), rgn(1, 6, 2)), make_caview2d_rm(a0_rm_s5, 3, 3) ));
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(0, 5, 2), rgn(1, 6, 2)), make_caview2d_cm(a0_cm_s5, 3, 3) ));

	// (step_range, rev_whole)

	double a0_rm_s6[] = {6, 5, 4, 3, 2, 1, 22, 21, 20, 19, 18, 17, 38, 37, 36, 35, 34, 33};
	double a0_cm_s6[] = {36, 38, 40, 29, 31, 33, 22, 24, 26, 15, 17, 19, 8, 10, 12, 1, 3, 5};

	ASSERT_TRUE( array2d_equal(a0_rm.V(rgn(0, 5, 2), rev_whole()), make_caview2d_rm(a0_rm_s6, 3, 6) ));
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(0, 5, 2), rev_whole()), make_caview2d_cm(a0_cm_s6, 3, 6) ));
}



TEST( Array2D, ViewCopy )
{
	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;

	double a0_buf[N] = {2, 4, 5, 7, 8, 3};
	double a1_buf[N];
	double e2_buf[4 * N];
	double e3_buf[4 * N];
	double a4_buf[N];

	set_zeros_to_elements(a1_buf, N);
	set_zeros_to_elements(e2_buf, 4 * N);
	set_zeros_to_elements(e3_buf, 4 * N);
	set_zeros_to_elements(a4_buf, N);

	// row major

	aview2d<double, row_major_t> a0_rm(a0_buf, m, n);
	aview2d<double, row_major_t> a1_rm(a1_buf, m, n);
	aview2d_ex<double, row_major_t, step_ind, step_ind> e2_rm = make_aview2d_ex_rm(e2_buf, 2*m, 2*n, step_ind(m, 2), step_ind(n, 2));
	aview2d_ex<double, row_major_t, step_ind, step_ind> e3_rm = make_aview2d_ex_rm(e3_buf, 2*m, 2*n, step_ind(m, 2), step_ind(n, 2));
	aview2d<double, row_major_t> a4_rm(a4_buf, m, n);

	copy(a0_rm, a1_rm);
	ASSERT_TRUE( array2d_equal(a1_rm, make_caview2d_rm(a0_buf, m, n) ));

	copy(a1_rm, e2_rm);
	ASSERT_TRUE( array2d_equal(e2_rm, make_caview2d_rm(a0_buf, m, n) ));

	copy(e2_rm, e3_rm);
	ASSERT_TRUE( array2d_equal(e3_rm, make_caview2d_rm(a0_buf, m, n) ));

	copy(e3_rm, a4_rm);
	ASSERT_TRUE( array2d_equal(a4_rm, make_caview2d_rm(a0_buf, m, n) ));

	set_zeros_to_elements(a1_buf, N);
	set_zeros_to_elements(e2_buf, 4 * N);
	set_zeros_to_elements(e3_buf, 4 * N);
	set_zeros_to_elements(a4_buf, N);

	// column_major

	aview2d<double, column_major_t> a0_cm(a0_buf, m, n);
	aview2d<double, column_major_t> a1_cm(a1_buf, m, n);
	aview2d_ex<double, column_major_t, step_ind, step_ind> e2_cm = make_aview2d_ex_cm(e2_buf, 2*m, 2*n, step_ind(m, 2), step_ind(n, 2));
	aview2d_ex<double, column_major_t, step_ind, step_ind> e3_cm = make_aview2d_ex_cm(e3_buf, 2*m, 2*n, step_ind(m, 2), step_ind(n, 2));
	aview2d<double, column_major_t> a4_cm(a4_buf, m, n);

	copy(a0_cm, a1_cm);
	ASSERT_TRUE( array2d_equal(a1_cm, make_caview2d_cm(a0_buf, m, n) ));

	copy(a1_cm, e2_cm);
	ASSERT_TRUE( array2d_equal(e2_cm, make_caview2d_cm(a0_buf, m, n) ));

	copy(e2_cm, e3_cm);
	ASSERT_TRUE( array2d_equal(e3_cm, make_caview2d_cm(a0_buf, m, n) ));

	copy(e3_cm, a4_cm);
	ASSERT_TRUE( array2d_equal(a4_cm, make_caview2d_cm(a0_buf, m, n) ));
}


TEST( Array2D, SubArraySelection )
{
	const size_t m0 = 6;
	const size_t n0 = 6;
	double src[m0 * n0];
	for (size_t i = 0; i < m0 * n0; ++i) src[i] = (double)(i + 1);

	caview2d<double, row_major_t>    Arm = make_aview2d_rm(src, m0, n0);
	caview2d<double, column_major_t> Acm = make_aview2d_cm(src, m0, n0);

	// select_elems

	const index_t sn0 = 6;
	index_t Is[sn0] = {1, 3, 4, 5, 2, 0};
	index_t Js[sn0] = {2, 2, 4, 5, 0, 0};
	caview1d<index_t> vIs(Is, sn0);
	caview1d<index_t> vJs(Js, sn0);

	array1d<double> s0_rm = select_elems(Arm, vIs, vJs);
	array1d<double> s0_cm = select_elems(Acm, vIs, vJs);

	double s0_rm_r[sn0] = {9, 21, 29, 36, 13, 1};
	double s0_cm_r[sn0] = {14, 16, 29, 36, 3, 1};

	ASSERT_TRUE( array1d_equal(s0_rm, make_caview1d(s0_rm_r, sn0) ));
	ASSERT_TRUE( array1d_equal(s0_cm, make_caview1d(s0_cm_r, sn0) ));

	// select_rows

	const index_t sn1 = 3;
	index_t rs[sn1] = {1, 4, 5};

	caview1d<index_t> rows(rs, sn1);

	array2d<double, row_major_t>    s1_rm = select_rows(Arm, rows);
	array2d<double, column_major_t> s1_cm = select_rows(Acm, rows);

	double s1_rm_r[sn1 * n0] = {7, 8, 9, 10, 11, 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
	double s1_cm_r[sn1 * n0] = {2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23, 24, 26, 29, 30, 32, 35, 36};

	ASSERT_TRUE( array2d_equal(s1_rm, make_caview2d_rm(s1_rm_r, sn1, n0) ));
	ASSERT_TRUE( array2d_equal(s1_cm, make_caview2d_cm(s1_cm_r, sn1, n0) ));

	// select_columns

	const index_t sn2 = 3;
	index_t cs[sn2] = {2, 3, 5};

	caview1d<index_t> cols(cs, sn2);

	array2d<double, row_major_t>    s2_rm = select_columns(Arm, cols);
	array2d<double, column_major_t> s2_cm = select_columns(Acm, cols);

	double s2_rm_r[m0 * sn2] = {3, 4, 6, 9, 10, 12, 15, 16, 18, 21, 22, 24, 27, 28, 30, 33, 34, 36};
	double s2_cm_r[m0 * sn2] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36};

	ASSERT_TRUE( array2d_equal(s2_rm, make_caview2d_rm(s2_rm_r, m0, sn2) ));
	ASSERT_TRUE( array2d_equal(s2_cm, make_caview2d_cm(s2_cm_r, m0, sn2) ));

	// select_rows_and_cols

	const index_t sm3 = 2;  index_t rs3[sm3] = {2, 4};
	const index_t sn3 = 3;  index_t cs3[sn3] = {1, 3, 5};

	caview1d<index_t> rows3(rs3, sm3);
	caview1d<index_t> cols3(cs3, sn3);

	array2d<double, row_major_t>    s3_rm = select_rows_cols(Arm, rows3, cols3);
	array2d<double, column_major_t> s3_cm = select_rows_cols(Acm, rows3, cols3);

	double s3_rm_r[sm3 * sn3] = {14, 16, 18, 26, 28, 30};
	double s3_cm_r[sm3 * sn3] = {9, 11, 21, 23, 33, 35};

	ASSERT_TRUE( array2d_equal(s3_rm, make_caview2d_rm(s3_rm_r, sm3, sn3) ));
	ASSERT_TRUE( array2d_equal(s3_cm, make_caview2d_cm(s3_cm_r, sm3, sn3) ));

}

