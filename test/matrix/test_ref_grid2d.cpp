/**
 * @file test_ref_grid2d.cpp
 *
 * Unit testing for ref_grid2d
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

template class bcs::cref_grid2d<double, DynamicDim, DynamicDim>;
template class bcs::cref_grid2d<double, DynamicDim, 1>;
template class bcs::cref_grid2d<double, 1, DynamicDim>;
template class bcs::cref_grid2d<double, 2, 2>;

template class bcs::ref_grid2d<double, DynamicDim, DynamicDim>;
template class bcs::ref_grid2d<double, DynamicDim, 1>;
template class bcs::ref_grid2d<double, 1, DynamicDim>;
template class bcs::ref_grid2d<double, 2, 2>;

#ifdef BCS_USE_STATIC_ASSERT
static_assert(bcs::is_base_of<
		bcs::IMatrixXpr<bcs::cref_grid2d<double>, double>,
		bcs::cref_grid2d<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixView<bcs::cref_grid2d<double>, double>,
		bcs::cref_grid2d<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixXpr<bcs::ref_grid2d<double>, double>,
		bcs::ref_grid2d<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixView<bcs::ref_grid2d<double>, double>,
		bcs::ref_grid2d<double> >::value, "Base verification failed.");

#endif


template<class Mat>
static void test_ref_grid2d(index_t m, index_t n)
{
	const int CTRows = ct_rows<Mat>::value;
	const int CTCols = ct_cols<Mat>::value;

	const index_t step = 2;
	const index_t ldim = m * step + 3;

	scoped_block<double> origin_blk(ldim * n);
	for (index_t i = 0; i < ldim * n; ++i) origin_blk[i] = double(i+1);
	double *origin = origin_blk.ptr_begin();

	// test construction

	Mat a1(origin, m, n, step, ldim);

	ASSERT_EQ(m, a1.nrows());
	ASSERT_EQ(n, a1.ncolumns());
	ASSERT_EQ(m * n, a1.nelems());
	ASSERT_EQ(step, a1.inner_step());
	ASSERT_EQ(ldim, a1.lead_dim());
	ASSERT_TRUE( a1.ptr_data() == origin );

	ASSERT_EQ( (size_t)a1.nelems(), a1.size() );
	ASSERT_EQ(a1.nelems() == 0, is_empty(a1));
	ASSERT_EQ(a1.nrows() == 1, is_row(a1));
	ASSERT_EQ(a1.ncolumns() == 1, is_column(a1));
	ASSERT_EQ(a1.nelems() == 1, is_scalar(a1));
	ASSERT_EQ(a1.nrows() == 1 || a1.ncolumns() == 1, is_vector(a1));
	ASSERT_EQ(a1.nrows() == 1 && a1.ncolumns() == 1, is_scalar(a1));

	// test element access

	bool elem_access_ok = true;

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			if (a1(i, j) != origin[i * step + j * ldim]) elem_access_ok = false;
		}
	}

	ASSERT_TRUE( elem_access_ok );

	if (CTCols == 1)
	{
		bool linear_indexing_ok = true;

		for (index_t i = 0; i < m; ++i)
		{
			if (a1[i] != origin[i * step]) linear_indexing_ok = false;
		}

		ASSERT_TRUE(linear_indexing_ok);
	}

	if (CTRows == 1)
	{
		bool linear_indexing_ok = true;

		for (index_t j = 0; j < n; ++j)
		{
			if (a1[j] != origin[j * ldim]) linear_indexing_ok = false;
		}

		ASSERT_TRUE(linear_indexing_ok);
	}

	// test copy construction

	Mat a2(a1);

	ASSERT_EQ(m, a2.nrows());
	ASSERT_EQ(n, a2.ncolumns());
	ASSERT_EQ(m * n, a2.nelems());
	ASSERT_EQ(step, a2.inner_step());
	ASSERT_EQ(ldim, a2.lead_dim());
	ASSERT_TRUE( a2.ptr_data() == a1.ptr_data() );
}


TEST( RefGrid2D, CDD )
{
	test_ref_grid2d< cref_grid2d<double, DynamicDim, DynamicDim> >( 4, 6 );
}

TEST( RefGrid2D, WDD )
{
	test_ref_grid2d< ref_grid2d<double, DynamicDim, DynamicDim> >( 4, 6 );
}

TEST( RefGrid2D, CDS )
{
	test_ref_grid2d< cref_grid2d<double, DynamicDim, 6> >( 4, 6 );
}

TEST( RefGrid2D, WDS )
{
	test_ref_grid2d< ref_grid2d<double, DynamicDim, 6> >( 4, 6 );
}

TEST( RefGrid2D, CD1 )
{
	test_ref_grid2d< cref_grid2d<double, DynamicDim, 1> >( 4, 1 );
}

TEST( RefGrid2D, WD1 )
{
	test_ref_grid2d< ref_grid2d<double, DynamicDim, 1> >( 4, 1 );
}


TEST( RefGrid2D, CSD )
{
	test_ref_grid2d< cref_grid2d<double, 4, DynamicDim> >( 4, 6 );
}

TEST( RefGrid2D, WSD )
{
	test_ref_grid2d< ref_grid2d<double, 4, DynamicDim> >( 4, 6 );
}

TEST( RefGrid2D, CSS )
{
	test_ref_grid2d< cref_grid2d<double, 4, 6> >( 4, 6 );
}

TEST( RefGrid2D, WSS )
{
	test_ref_grid2d< ref_grid2d<double, 4, 6> >( 4, 6 );
}

TEST( RefGrid2D, CS1 )
{
	test_ref_grid2d< cref_grid2d<double, 4, 1> >( 4, 1 );
}

TEST( RefGrid2D, WS1 )
{
	test_ref_grid2d< ref_grid2d<double, 4, 1> >( 4, 1 );
}


TEST( RefGrid2D, C1D )
{
	test_ref_grid2d< cref_grid2d<double, 1, DynamicDim> >( 1, 6 );
}

TEST( RefGrid2D, W1D )
{
	test_ref_grid2d< ref_grid2d<double, 1, DynamicDim> >( 1, 6 );
}

TEST( RefGrid2D, C1S )
{
	test_ref_grid2d< cref_grid2d<double, 1, 6> >( 1, 6 );
}

TEST( RefGrid2D, W1S )
{
	test_ref_grid2d< ref_grid2d<double, 1, 6> >( 1, 6 );
}

TEST( RefGrid2D, C11 )
{
	test_ref_grid2d< cref_grid2d<double, 1, 1> >( 1, 1 );
}

TEST( RefGrid2D, W11 )
{
	test_ref_grid2d< ref_grid2d<double, 1, 1> >( 1, 1 );
}



