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


template class bcs::caview2d_ex<double, step_ind, step_ind>;
template class bcs::aview2d_ex<double,  step_ind, step_ind>;

template class bcs::caview2d_ex<double, id_ind, step_ind>;
template class bcs::aview2d_ex<double,  id_ind, step_ind>;

template class bcs::caview2d_block<double>;
template class bcs::aview2d_block<double>;

template class bcs::caview2d<double>;
template class bcs::aview2d<double>;

template class bcs::array2d<double>;


/************************************************
 *
 *  Inheritance testing
 *
 ************************************************/

// caview2d_ex


#ifdef BCS_USE_STATIC_ASSERT

static_assert( (is_base_of<
		bcs::IConstAView2DBase<bcs::caview2d_ex<double, id_ind, step_ind>, double>,
		bcs::caview2d_ex<double, id_ind, step_ind> >::value),
		"caview2d_ex base-class assertion failure" );

static_assert( (is_base_of<
		bcs::IConstRegularAView2D<bcs::caview2d_ex<double, id_ind, step_ind>, double>,
		bcs::caview2d_ex<double, id_ind, step_ind> >::value),
		"caview2d_ex base-class assertion failure" );

// aview2d_ex

static_assert( (is_base_of<
		bcs::caview2d_ex<double, id_ind, step_ind>,
		bcs::aview2d_ex<double, id_ind, step_ind> >::value),
		"aview2d_ex base-class assertion failure");

static_assert( (is_base_of<
		bcs::IAView2DBase<bcs::aview2d_ex<double, id_ind, step_ind>, double>,
		bcs::aview2d_ex<double, id_ind, step_ind> >::value),
		"aview2d_ex base-class assertion failure");

static_assert( (is_base_of<
		bcs::IRegularAView2D<bcs::aview2d_ex<double, id_ind, step_ind>, double>,
		bcs::aview2d_ex<double, id_ind, step_ind> >::value),
		"aview2d_ex base-class assertion failure" );


// caview2d_block

static_assert( (is_base_of<
		bcs::IConstAView2DBase<bcs::caview2d_block<double>, double>,
		bcs::caview2d_block<double> >::value),
		"cview2d_block base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstRegularAView2D<bcs::caview2d_block<double>, double>,
		bcs::caview2d_block<double> >::value),
		"cview2d_block base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstBlockAView2D<bcs::caview2d_block<double>, double>,
		bcs::caview2d_block<double> >::value),
		"cview2d_block base-class assertion failure");


// aview2d_block

static_assert( (is_base_of<
		bcs::caview2d_block<double>,
		bcs::aview2d_block<double> >::value),
		"aview2d_block base-class assertion failure");

static_assert( (is_base_of<
		bcs::IAView2DBase<bcs::aview2d_block<double>, double>,
		bcs::aview2d_block<double> >::value),
		"aview2d_block base-class assertion failure");

static_assert( (is_base_of<
		bcs::IRegularAView2D<bcs::aview2d_block<double>, double>,
		bcs::aview2d_block<double> >::value),
		"aview2d_block base-class assertion failure");

static_assert( (is_base_of<
		bcs::IBlockAView2D<bcs::aview2d_block<double>, double>,
		bcs::aview2d_block<double> >::value),
		"aview2d_block base-class assertion failure");


// caview2d

static_assert( (is_base_of<
		bcs::IConstAView2DBase<bcs::caview2d<double>, double>,
		bcs::caview2d<double> >::value),
		"caview2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstRegularAView2D<bcs::caview2d<double>, double>,
		bcs::caview2d<double> >::value),
		"caview2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstBlockAView2D<bcs::caview2d<double>, double>,
		bcs::caview2d<double> >::value),
		"caview2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstContinuousAView2D<bcs::caview2d<double>, double>,
		bcs::caview2d<double> >::value),
		"caview2d base-class assertion failure");

// aview2d

static_assert( (is_base_of<
		bcs::IAView2DBase<bcs::aview2d<double>, double>,
		bcs::aview2d<double> >::value),
		"aview2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IRegularAView2D<bcs::aview2d<double>, double>,
		bcs::aview2d<double> >::value),
		"aview2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IBlockAView2D<bcs::aview2d<double>, double>,
		bcs::aview2d<double> >::value),
		"aview2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IContinuousAView2D<bcs::aview2d<double>, double>,
		bcs::aview2d<double> >::value),
		"aview2d base-class assertion failure");

// array2d

static_assert( (is_base_of<
		bcs::IConstAView2DBase<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstRegularAView2D<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstBlockAView2D<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IConstContinuousAView2D<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IAView2DBase<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IRegularAView2D<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IBlockAView2D<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

static_assert( (is_base_of<
		bcs::IContinuousAView2D<bcs::array2d<double>, double>,
		bcs::array2d<double> >::value),
		"array2d base-class assertion failure");

#endif



// syntax checking for subviews

template<class Derived> void only_accept_blk_views(const IConstBlockAView2D<Derived, double>& a) { }

template<class Derived> void only_accept_cont_views(const IConstContinuousAView2D<Derived, double>& a) { }

void syntax_check_arr2_subview()
{
	caview2d_block<double> cb_cm(BCS_NULL, 0, 0, 0);
	aview2d_block<double> bv_cm(BCS_NULL, 0, 0, 0);

	only_accept_blk_views(cb_cm);
	only_accept_blk_views(bv_cm);

	only_accept_blk_views(cb_cm.V(whole(), whole()));
	only_accept_blk_views(cb_cm.V(rgn(0, 0), whole()));
	only_accept_blk_views(cb_cm.V(whole(), rgn(0, 0)));
	only_accept_blk_views(cb_cm.V(rgn(0, 0), rgn(0, 0)));

	only_accept_blk_views(bv_cm.V(whole(), whole()));
	only_accept_blk_views(bv_cm.V(rgn(0, 0), whole()));
	only_accept_blk_views(bv_cm.V(whole(), rgn(0, 0)));
	only_accept_blk_views(bv_cm.V(rgn(0, 0), rgn(0, 0)));

	caview2d<double> cv_cm(BCS_NULL, 0, 0);
	aview2d<double> av_cm(BCS_NULL, 0, 0);
	array2d<double> arr_cm(0, 0);

	only_accept_cont_views(cv_cm);
	only_accept_cont_views(av_cm);
	only_accept_cont_views(arr_cm);

	only_accept_cont_views(cv_cm.V(whole(), whole()));
	only_accept_cont_views(av_cm.V(whole(), whole()));
	only_accept_cont_views(arr_cm.V(whole(), whole()));

	only_accept_cont_views(cv_cm.V(whole(), rgn(0, 0)));
	only_accept_cont_views(av_cm.V(whole(), rgn(0, 0)));
	only_accept_cont_views(arr_cm.V(whole(), rgn(0, 0)));
}



/************************************************
 *
 *  Auxiliary functions
 *
 ************************************************/


template<class Derived, typename T>
inline index_t _sub2ind(const IConstAView2DBase<Derived, T>& a, index_t i, index_t j)
{
	return i + j * a.nrows();
}


template<class Derived, typename T>
bool array_integrity_test(const bcs::IConstRegularAView2D<Derived, T>& a)
{
	index_t m = a.nrows();
	index_t n = a.ncolumns();

	if (a.ndims() != 2) return false;
	if (a.nelems() != m * n) return false;
	if (a.size() != (size_t)(m * n)) return false;
	if (a.shape() != arr_shape(m, n)) return false;
	if (a.is_empty() != (a.nelems() == 0)) return false;

	return true;
}

template<class Derived, typename T>
bool cont_array_integrity_test(const bcs::IConstContinuousAView2D<Derived, T>& a)
{
	if (!array_integrity_test(a)) return false;

	index_t m = a.nrows();
	index_t n = a.ncolumns();
	if (!a.is_empty())
	{
		if (a.pbase() != &(a[0])) return false;
	}

	caview1d<T> fview = a.flatten();

	if (fview.pbase() != a.pbase()) return false;
	if (fview.nelems() != m * n) return false;

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			index_t idx = _sub2ind(a, i, j);

			if (&(a(i, j)) != &(a[idx])) return false;
			if (a(i, j) != a[idx]) return false;
			if (a[idx] != fview[idx]) return false;
		}
	}

	return true;
}


template<class Derived, typename T>
bool elemwise_operation_test(bcs::IRegularAView2D<Derived, T>& a)
{
	index_t m = a.nrows();
	index_t n = a.ncolumns();
	block<T> blk(a.nelems());
	const T *b = blk.pbase();

	Derived& ad = a.derived();

	// export
	export_to(ad, blk.pbase());

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (b[_sub2ind(ad, i, j)] != a(i, j)) return false;
		}
	}

	// fill
	T v = T(123);
	fill(ad, v);

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (a(i, j) != v) return false;
		}
	}

	// import
	import_from(ad, blk.pbase());

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (a(i, j) != b[_sub2ind(ad, i, j)]) return false;
		}
	}

	return true;
}


/**********************************************************
 *
 *  Test Cases
 *
 **********************************************************/


TEST( Array2D, Aview2D )
{
	double src[6] = {3, 4, 5, 1, 2, 7};
	index_t m = 2;
	index_t n = 3;

	aview2d<double> a1(src, m, n);

	ASSERT_EQ(a1.nrows(), m);
	ASSERT_EQ(a1.ncolumns(), n);
	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	aview2d<double> a2(a1);

	ASSERT_EQ(a2.nrows(), m);
	ASSERT_EQ(a2.ncolumns(), n);
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal(a2, src, m * n) );

	ASSERT_EQ( a1.pbase(), a2.pbase() );
}


TEST( Array2D, Aview2DBlock )
{
	double src[12] = {3, 4, 5, 1, 2, 7, 8, 9, 0, 3, 7, 6};
	index_t ldim = 3;
	index_t m = 2;
	index_t n = 3;

	double ref[6] = {3, 4, 1, 2, 8, 9};

	aview2d_block<double> a1(src, ldim, m, n);

	ASSERT_EQ(a1.nrows(), m);
	ASSERT_EQ(a1.ncolumns(), n);
	ASSERT_EQ(a1.lead_dim(), 3);
	ASSERT_TRUE( array_integrity_test(a1) );
	ASSERT_TRUE( array2d_equal(a1, make_aview2d(ref, m, n)) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	aview2d_block<double> a2(a1);

	ASSERT_EQ(a2.nrows(), m);
	ASSERT_EQ(a2.ncolumns(), n);
	ASSERT_EQ(a1.lead_dim(), 3);
	ASSERT_TRUE( array_integrity_test(a2) );
	ASSERT_TRUE( array2d_equal(a2, make_aview2d(ref, m, n)) );

	ASSERT_EQ( a1.pbase(), a2.pbase() );
}


TEST( Array2D, Array2D  )
{
	double src[24];
	for (int i = 0; i < 24; ++i) src[i] = i+1;
	index_t k = 0;

	index_t m = 2;
	index_t n = 3;

	double v2 = 7.0;

	// row major

	array2d<double> a0(0, 0);

	ASSERT_EQ( a0.nrows(), 0 );
	ASSERT_EQ( a0.ncolumns(), 0 );
	ASSERT_TRUE( cont_array_integrity_test(a0) );

	array2d<double> a1(m, n);
	k = 0;
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			a1(i, j) = src[k++];
		}
	}

	ASSERT_EQ( a1.nrows(), m );
	ASSERT_EQ( a1.ncolumns(), n );
	ASSERT_TRUE( cont_array_integrity_test(a1) );
	ASSERT_TRUE( array_equal(a1, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	array2d<double> a2(m, n, v2);

	ASSERT_EQ( a2.nrows(), m );
	ASSERT_EQ( a2.ncolumns(), n );
	ASSERT_TRUE( cont_array_integrity_test(a2) );
	ASSERT_TRUE( array_equal_scalar(a2, v2, m * n) );

	array2d<double> a3(m, n, src);

	ASSERT_EQ( a3.nrows(), m );
	ASSERT_EQ( a3.ncolumns(), n );
	ASSERT_TRUE( cont_array_integrity_test(a3) );
	ASSERT_TRUE( array_equal(a3, src, m * n) );
	ASSERT_TRUE( elemwise_operation_test(a3) );

	array2d<double> a4(a3);

	ASSERT_NE( a4.pbase(), a3.pbase() );
	ASSERT_EQ(a4.nrows(), m);
	ASSERT_EQ(a4.ncolumns(), n);
	ASSERT_TRUE( cont_array_integrity_test(a4) );
	ASSERT_TRUE( array_equal(a4, src, m * n) );
}


TEST( Array2D, Aview2DEx )
{
	double src[36];
	for (int i = 0; i < 36; ++i) src[i] = i+1;

	index_t ldim = 5;

	// id x id

	aview2d_ex<double, id_ind, id_ind> a0(src, ldim, id_ind(2), id_ind(3));
	double r0[] = {1, 2, 6, 7, 11, 12};

	ASSERT_EQ(a0.nrows(), 2);
	ASSERT_EQ(a0.ncolumns(), 3);
	ASSERT_TRUE( array_integrity_test(a0) );
	ASSERT_TRUE( array2d_equal(a0, make_caview2d(r0, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a0) );

	// id x step

	aview2d_ex<double, id_ind, step_ind> a1(src, ldim, id_ind(2), step_ind(3, 2));
	double r1[] = {1, 2, 11, 12, 21, 22};

	ASSERT_EQ(a1.nrows(), 2);
	ASSERT_EQ(a1.ncolumns(), 3);
	ASSERT_TRUE( array_integrity_test(a1) );
	ASSERT_TRUE( array2d_equal(a1, make_caview2d(r1, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a1) );

	// step x step

	aview2d_ex<double, step_ind, step_ind> a2(src, ldim, step_ind(2, 2), step_ind(3, 2));
	double r2[] = {1, 3, 11, 13, 21, 23};

	ASSERT_EQ(a2.nrows(), 2);
	ASSERT_EQ(a2.ncolumns(), 3);
	ASSERT_TRUE( array_integrity_test(a2) );
	ASSERT_TRUE( array2d_equal(a2, make_caview2d(r2, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a2) );

	// step x rep

	aview2d_ex<double, step_ind, rep_ind> a3(src, ldim, step_ind(2, 2), rep_ind(3));
	double r3[] = {1, 3, 1, 3, 1, 3};

	ASSERT_EQ(a3.nrows(), 2);
	ASSERT_EQ(a3.ncolumns(), 3);
	ASSERT_TRUE( array_integrity_test(a3) );
	ASSERT_TRUE( array2d_equal(a3, make_caview2d(r3, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a3) );

	// rep x rep

	aview2d_ex<double, rep_ind, rep_ind> a4(src, ldim, rep_ind(2), rep_ind(3));
	double r4[] = {1, 1, 1, 1, 1, 1};

	ASSERT_EQ(a4.nrows(), 2);
	ASSERT_EQ(a4.ncolumns(), 3);
	ASSERT_TRUE( array_integrity_test(a4) );
	ASSERT_TRUE( array2d_equal(a4, make_caview2d(r4, 2, 3)) );
	ASSERT_TRUE( elemwise_operation_test(a4) );
}


TEST( Array2D, Slices )
{
	double src[60];
	for (int i = 0; i < 60; ++i) src[i] = i+1;

	index_t ldim = 7;
	array2d<double> Acm( aview2d_block<double>(src, ldim, 5, 6) );

	// row

	double r0_cm[6] = {2, 9, 16, 23, 30, 37};
	ASSERT_TRUE( array1d_equal( Acm.row(1), make_caview1d(r0_cm, 6) ) );

	// row range

	double r1_cm[4] = {9, 16, 23, 30};
	ASSERT_TRUE( array1d_equal( Acm.row(1, rgn_n(1, 4)), make_caview1d(r1_cm, 4) ) );

	double r2_cm[3] = {2, 16, 30};
	ASSERT_TRUE( array1d_equal( Acm.row(1, rgn_n(0, 3, 2)), make_caview1d(r2_cm, 3) ) );

	double r3_cm[3] = {16, 16, 16};
	ASSERT_TRUE( array1d_equal( Acm.row(1, rep(2, 3)), make_caview1d(r3_cm, 3) ) );

	// column

	double c0_cm[5] = {8, 9, 10, 11, 12};
	ASSERT_TRUE( array1d_equal( Acm.column(1), make_caview1d(c0_cm, 5) ) );

	// column range

	double c1_cm[4] = {9, 10, 11, 12};
	ASSERT_TRUE( array1d_equal( Acm.column(1, rgn_n(1, 4)), make_caview1d(c1_cm, 4) ) );

	double c2_cm[3] = {8, 10, 12};
	ASSERT_TRUE( array1d_equal( Acm.column(1, rgn_n(0, 3, 2)), make_caview1d(c2_cm, 3) ) );

	double c3_cm[3] = {10, 10, 10};
	ASSERT_TRUE( array1d_equal( Acm.column(1, rep(2, 3)), make_caview1d(c3_cm, 3) ) );

}


TEST( Array2D, BlockSlices )
{
	double src[60];
	for (int i = 0; i < 60; ++i) src[i] = i+1;

	index_t ldim = 7;
	aview2d_block<double> Acm(src, ldim, 5, 6);

	// row

	double r0_cm[6] = {2, 9, 16, 23, 30, 37};
	ASSERT_TRUE( array1d_equal( Acm.row(1), make_caview1d(r0_cm, 6) ) );

	// row range

	double r1_cm[4] = {9, 16, 23, 30};
	ASSERT_TRUE( array1d_equal( Acm.row(1, rgn_n(1, 4)), make_caview1d(r1_cm, 4) ) );

	double r2_cm[3] = {2, 16, 30};
	ASSERT_TRUE( array1d_equal( Acm.row(1, rgn_n(0, 3, 2)), make_caview1d(r2_cm, 3) ) );

	double r3_cm[3] = {16, 16, 16};
	ASSERT_TRUE( array1d_equal( Acm.row(1, rep(2, 3)), make_caview1d(r3_cm, 3) ) );

	// column

	double c0_cm[5] = {8, 9, 10, 11, 12};
	ASSERT_TRUE( array1d_equal( Acm.column(1), make_caview1d(c0_cm, 5) ) );

	// column range

	double c1_cm[4] = {9, 10, 11, 12};
	ASSERT_TRUE( array1d_equal( Acm.column(1, rgn_n(1, 4)), make_caview1d(c1_cm, 4) ) );

	double c2_cm[3] = {8, 10, 12};
	ASSERT_TRUE( array1d_equal( Acm.column(1, rgn_n(0, 3, 2)), make_caview1d(c2_cm, 3) ) );

	double c3_cm[3] = {10, 10, 10};
	ASSERT_TRUE( array1d_equal( Acm.column(1, rep(2, 3)), make_caview1d(c3_cm, 3) ) );

}


TEST( Array2D, SubViews )
{
	double src0[60];
	for (int i = 0; i < 60; ++i) src0[i] = i+1;

	index_t ldim = 7;
	array2d<double> a0_cm( aview2d_block<double>(src0, ldim, 5, 6) );

	// (whole, whole)

	double a0_cm_s0[] = {
			1, 2, 3, 4, 5,
			8, 9, 10, 11, 12,
			15, 16, 17, 18, 19,
			22, 23, 24, 25, 26,
			29, 30, 31, 32, 33,
			36, 37, 38, 39, 40
	};

	ASSERT_TRUE( array2d_equal( a0_cm.V(whole(), whole()),  make_caview2d(a0_cm_s0, 5, 6) ));

	// (range, whole)

	double a0_cm_s1[] = {1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30, 31, 36, 37, 38};
	ASSERT_TRUE( array2d_equal( a0_cm.V(rgn(0, 3), whole()),  make_caview2d(a0_cm_s1, 3, 6) ));

	// (whole, range)

	double a0_cm_s2[] = {8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
	ASSERT_TRUE( array2d_equal( a0_cm.V(whole(), rgn(1, 4)),  make_caview2d(a0_cm_s2, 5, 3) ));

	// (range, range)

	double a0_cm_s3[] = {10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33};
	ASSERT_TRUE( array2d_equal( a0_cm.V(rgn(2, 5), rgn(1, 5)),  make_caview2d(a0_cm_s3, 3, 4) ));

	// (range, step_range)

	double a0_cm_s4[] = {10, 11, 12, 24, 25, 26, 38, 39, 40};
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(2, 5), rgn(1, 6, 2)),  make_caview2d(a0_cm_s4, 3, 3) ));

	// (step_range, step_range)

	double a0_cm_s5[] = {8, 10, 12, 22, 24, 26, 36, 38, 40};
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(0, 5, 2), rgn(1, 6, 2)), make_caview2d(a0_cm_s5, 3, 3) ));

	// (step_range, rev_whole)

	double a0_cm_s6[] = {36, 38, 40, 29, 31, 33, 22, 24, 26, 15, 17, 19, 8, 10, 12, 1, 3, 5};
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(0, 5, 2), rev_whole()), make_caview2d(a0_cm_s6, 3, 6) ));
}


TEST( Array2D, BlockSubViews )
{
	double src0[60];
	for (int i = 0; i < 60; ++i) src0[i] = i+1;

	index_t ldim = 7;
	aview2d_block<double> a0_cm(src0, ldim, 5, 6);

	// (whole, whole)

	double a0_cm_s0[] = {
			1, 2, 3, 4, 5,
			8, 9, 10, 11, 12,
			15, 16, 17, 18, 19,
			22, 23, 24, 25, 26,
			29, 30, 31, 32, 33,
			36, 37, 38, 39, 40
	};

	ASSERT_TRUE( array2d_equal( a0_cm.V(whole(), whole()),  make_caview2d(a0_cm_s0, 5, 6) ));

	// (range, whole)

	double a0_cm_s1[] = {1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30, 31, 36, 37, 38};
	ASSERT_TRUE( array2d_equal( a0_cm.V(rgn(0, 3), whole()),  make_caview2d(a0_cm_s1, 3, 6) ));

	// (whole, range)

	double a0_cm_s2[] = {8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
	ASSERT_TRUE( array2d_equal( a0_cm.V(whole(), rgn(1, 4)),  make_caview2d(a0_cm_s2, 5, 3) ));

	// (range, range)

	double a0_cm_s3[] = {10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33};
	ASSERT_TRUE( array2d_equal( a0_cm.V(rgn(2, 5), rgn(1, 5)),  make_caview2d(a0_cm_s3, 3, 4) ));

	// (range, step_range)

	double a0_cm_s4[] = {10, 11, 12, 24, 25, 26, 38, 39, 40};
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(2, 5), rgn(1, 6, 2)),  make_caview2d(a0_cm_s4, 3, 3) ));

	// (step_range, step_range)

	double a0_cm_s5[] = {8, 10, 12, 22, 24, 26, 36, 38, 40};
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(0, 5, 2), rgn(1, 6, 2)), make_caview2d(a0_cm_s5, 3, 3) ));

	// (step_range, rev_whole)

	double a0_cm_s6[] = {36, 38, 40, 29, 31, 33, 22, 24, 26, 15, 17, 19, 8, 10, 12, 1, 3, 5};
	ASSERT_TRUE( array2d_equal(a0_cm.V(rgn(0, 5, 2), rev_whole()), make_caview2d(a0_cm_s6, 3, 6) ));
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

	mem<double>::zero(N, a1_buf);
	mem<double>::zero(4*N, e2_buf);
	mem<double>::zero(4*N, e3_buf);
	mem<double>::zero(N, a4_buf);

	// column_major

	aview2d<double> a0_cm(a0_buf, m, n);
	aview2d<double> a1_cm(a1_buf, m, n);
	aview2d_ex<double, step_ind, step_ind> e2_cm = make_aview2d_ex(e2_buf, 2*m, 2*n, step_ind(m, 2), step_ind(n, 2));
	aview2d_ex<double, step_ind, step_ind> e3_cm = make_aview2d_ex(e3_buf, 2*m, 2*n, step_ind(m, 2), step_ind(n, 2));
	aview2d<double> a4_cm(a4_buf, m, n);

	copy(a0_cm, a1_cm);
	ASSERT_TRUE( array2d_equal(a1_cm, make_caview2d(a0_buf, m, n) ));

	copy(a1_cm, e2_cm);
	ASSERT_TRUE( array2d_equal(e2_cm, make_caview2d(a0_buf, m, n) ));

	copy(e2_cm, e3_cm);
	ASSERT_TRUE( array2d_equal(e3_cm, make_caview2d(a0_buf, m, n) ));

	copy(e3_cm, a4_cm);
	ASSERT_TRUE( array2d_equal(a4_cm, make_caview2d(a0_buf, m, n) ));
}


TEST( Array2D, SubArraySelection )
{
	const size_t m0 = 6;
	const size_t n0 = 6;
	double src[m0 * n0];
	for (size_t i = 0; i < m0 * n0; ++i) src[i] = (double)(i + 1);

	caview2d<double> Acm = make_aview2d(src, m0, n0);

	// select_elems

	const index_t sn0 = 6;
	index_t Is[sn0] = {1, 3, 4, 5, 2, 0};
	index_t Js[sn0] = {2, 2, 4, 5, 0, 0};
	caview1d<index_t> vIs(Is, sn0);
	caview1d<index_t> vJs(Js, sn0);

	array1d<double> s0_cm = select_elems(Acm, vIs, vJs);

	double s0_cm_r[sn0] = {14, 16, 29, 36, 3, 1};
	ASSERT_TRUE( array1d_equal(s0_cm, make_caview1d(s0_cm_r, sn0) ));

	// select_rows

	const index_t sn1 = 3;
	index_t rs[sn1] = {1, 4, 5};

	caview1d<index_t> rows(rs, sn1);

	array2d<double> s1_cm = select_rows(Acm, rows);

	double s1_cm_r[sn1 * n0] = {2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23, 24, 26, 29, 30, 32, 35, 36};
	ASSERT_TRUE( array2d_equal(s1_cm, make_caview2d(s1_cm_r, sn1, n0) ));

	// select_columns

	const index_t sn2 = 3;
	index_t cs[sn2] = {2, 3, 5};

	caview1d<index_t> cols(cs, sn2);

	array2d<double> s2_cm = select_columns(Acm, cols);

	double s2_cm_r[m0 * sn2] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36};
	ASSERT_TRUE( array2d_equal(s2_cm, make_caview2d(s2_cm_r, m0, sn2) ));

	// select_rows_and_cols

	const index_t sm3 = 2;  index_t rs3[sm3] = {2, 4};
	const index_t sn3 = 3;  index_t cs3[sn3] = {1, 3, 5};

	caview1d<index_t> rows3(rs3, sm3);
	caview1d<index_t> cols3(cs3, sn3);

	array2d<double> s3_cm = select_rows_cols(Acm, rows3, cols3);

	double s3_cm_r[sm3 * sn3] = {9, 11, 21, 23, 33, 35};
	ASSERT_TRUE( array2d_equal(s3_cm, make_caview2d(s3_cm_r, sm3, sn3) ));

}

