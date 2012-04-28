/**
 * @file test_matrix_assign.cpp
 *
 * Test matrix assignment
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

template<class Mat>
static bool verify_matrix_values(const IMatrixView<Mat, double>& A, double base_v)
{
	for (index_t j = 0; j < A.ncolumns(); ++j)
	{
		for (index_t i = 0; i < A.nrows(); ++i)
		{
			double expect_val = (i + j * A.nrows() + 1) * base_v;
			if (A.elem(i, j) != expect_val) return false;
		}
	}
	return true;
}

template<class Mat>
static void fill_matrix_values(IRegularMatrix<Mat, double>& A, double base_v)
{
	for (index_t j = 0; j < A.ncolumns(); ++j)
	{
		for (index_t i = 0; i < A.nrows(); ++i)
		{
			double expect_val = (i + j * A.nrows() + 1) * base_v;
			A.elem(i, j) = expect_val;
		}
	}
}


template<int SrcRows, int SrcCols, int DstRows, int DstCols>
void test_densemat_to_densemat_assign(index_t m, index_t n, bool do_empty_assign)
{
	dense_matrix<double, SrcRows, SrcCols> A(m, n);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	const double *pa = A.ptr_data();

	if (do_empty_assign)
	{
		dense_matrix<double, DstRows, DstCols> B0;
		B0 = A;

		ASSERT_TRUE( A.ptr_data() == pa );
		ASSERT_TRUE( B0.ptr_data() != A.ptr_data() );

		ASSERT_EQ( m, B0.nrows() );
		ASSERT_EQ( n, B0.ncolumns() );

		ASSERT_TRUE( verify_matrix_values(B0, 2.0) );
	}

	dense_matrix<double, DstRows, DstCols> B(m, n);
	fill(B, 0.0);
	const double *pb = B.ptr_data();

	B = A;

	ASSERT_TRUE( A.ptr_data() == pa );
	ASSERT_TRUE( B.ptr_data() == pb );

	ASSERT_EQ( m, B.nrows() );
	ASSERT_EQ( n, B.ncolumns() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );

}


TEST( DenseMatAssign, DDtoDD ) { test_densemat_to_densemat_assign<0, 0, 0, 0>(3, 4, true); }
TEST( DenseMatAssign, DDtoDS ) { test_densemat_to_densemat_assign<0, 0, 0, 4>(3, 4, true); }
TEST( DenseMatAssign, DDtoSD ) { test_densemat_to_densemat_assign<0, 0, 3, 0>(3, 4, true); }
TEST( DenseMatAssign, DDtoSS ) { test_densemat_to_densemat_assign<0, 0, 3, 4>(3, 4, true); }

TEST( DenseMatAssign, DStoDD ) { test_densemat_to_densemat_assign<0, 4, 0, 0>(3, 4, true); }
TEST( DenseMatAssign, DStoDS ) { test_densemat_to_densemat_assign<0, 4, 0, 4>(3, 4, true); }
TEST( DenseMatAssign, DStoSD ) { test_densemat_to_densemat_assign<0, 4, 3, 0>(3, 4, true); }
TEST( DenseMatAssign, DStoSS ) { test_densemat_to_densemat_assign<0, 4, 3, 4>(3, 4, true); }

TEST( DenseMatAssign, SDtoDD ) { test_densemat_to_densemat_assign<3, 0, 0, 0>(3, 4, true); }
TEST( DenseMatAssign, SDtoDS ) { test_densemat_to_densemat_assign<3, 0, 0, 4>(3, 4, true); }
TEST( DenseMatAssign, SDtoSD ) { test_densemat_to_densemat_assign<3, 0, 3, 0>(3, 4, true); }
TEST( DenseMatAssign, SDtoSS ) { test_densemat_to_densemat_assign<3, 0, 3, 4>(3, 4, true); }

TEST( DenseMatAssign, SStoDD ) { test_densemat_to_densemat_assign<3, 4, 0, 0>(3, 4, true); }
TEST( DenseMatAssign, SStoDS ) { test_densemat_to_densemat_assign<3, 4, 0, 4>(3, 4, true); }
TEST( DenseMatAssign, SStoSD ) { test_densemat_to_densemat_assign<3, 4, 3, 0>(3, 4, true); }
TEST( DenseMatAssign, SStoSS ) { test_densemat_to_densemat_assign<3, 4, 3, 4>(3, 4, true); }


TEST( DenseMatAssign, DColtoDCol )
{
	const index_t len = 5;

	dense_col<double> A(len);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	dense_col<double> B;
	B = A;

	ASSERT_EQ(len, B.nrows());
	ASSERT_EQ(1, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() != A.ptr_data() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}


TEST( DenseMatAssign, DMattoDCol )
{
	const index_t len = 5;

	dense_matrix<double> A(len, 1);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	dense_col<double> B;
	B = A;

	ASSERT_EQ(len, B.nrows());
	ASSERT_EQ(1, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() != A.ptr_data() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}


TEST( DenseMatAssign, DRowtoDRow )
{
	const index_t len = 5;

	dense_row<double> A(len);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	dense_row<double> B;
	B = A;

	ASSERT_EQ(1, B.nrows());
	ASSERT_EQ(len, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() != A.ptr_data() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}


TEST( DenseMatAssign, DMattoDRow )
{
	const index_t len = 5;

	dense_matrix<double> A(1, len);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	dense_row<double> B;
	B = A;

	ASSERT_EQ(1, B.nrows());
	ASSERT_EQ(len, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() != A.ptr_data() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}




template<int SrcRows, int SrcCols, int DstRows, int DstCols>
void test_refmat_to_refmat_assign(index_t m, index_t n)
{
	scoped_block<double> blk_a(m * n);
	scoped_block<double> blk_b(m * n);

	double *pa = blk_a.ptr_begin();
	double *pb = blk_b.ptr_begin();

	ref_matrix<double, SrcRows, SrcCols> A(pa, m, n);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	ref_matrix<double, DstRows, DstCols> B(pb, m, n);
	fill(B, 0.0);

	B = A;

	ASSERT_TRUE( A.ptr_data() == pa );
	ASSERT_TRUE( B.ptr_data() == pb );

	ASSERT_EQ( m, B.nrows() );
	ASSERT_EQ( n, B.ncolumns() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}


TEST( RefMatAssign, DDtoDD ) { test_refmat_to_refmat_assign<0, 0, 0, 0>(3, 4); }
TEST( RefMatAssign, DDtoDS ) { test_refmat_to_refmat_assign<0, 0, 0, 4>(3, 4); }
TEST( RefMatAssign, DDtoSD ) { test_refmat_to_refmat_assign<0, 0, 3, 0>(3, 4); }
TEST( RefMatAssign, DDtoSS ) { test_refmat_to_refmat_assign<0, 0, 3, 4>(3, 4); }

TEST( RefMatAssign, DStoDD ) { test_refmat_to_refmat_assign<0, 4, 0, 0>(3, 4); }
TEST( RefMatAssign, DStoDS ) { test_refmat_to_refmat_assign<0, 4, 0, 4>(3, 4); }
TEST( RefMatAssign, DStoSD ) { test_refmat_to_refmat_assign<0, 4, 3, 0>(3, 4); }
TEST( RefMatAssign, DStoSS ) { test_refmat_to_refmat_assign<0, 4, 3, 4>(3, 4); }

TEST( RefMatAssign, SDtoDD ) { test_refmat_to_refmat_assign<3, 0, 0, 0>(3, 4); }
TEST( RefMatAssign, SDtoDS ) { test_refmat_to_refmat_assign<3, 0, 0, 4>(3, 4); }
TEST( RefMatAssign, SDtoSD ) { test_refmat_to_refmat_assign<3, 0, 3, 0>(3, 4); }
TEST( RefMatAssign, SDtoSS ) { test_refmat_to_refmat_assign<3, 0, 3, 4>(3, 4); }

TEST( RefMatAssign, SStoDD ) { test_refmat_to_refmat_assign<3, 4, 0, 0>(3, 4); }
TEST( RefMatAssign, SStoDS ) { test_refmat_to_refmat_assign<3, 4, 0, 4>(3, 4); }
TEST( RefMatAssign, SStoSD ) { test_refmat_to_refmat_assign<3, 4, 3, 0>(3, 4); }
TEST( RefMatAssign, SStoSS ) { test_refmat_to_refmat_assign<3, 4, 3, 4>(3, 4); }


TEST( RefMatAssign, DColtoDCol )
{
	const index_t len = 5;
	scoped_block<double> sblk(len);
	scoped_block<double> dblk(len);

	ref_col<double> A(sblk.ptr_begin(), len);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	ref_col<double> B(dblk.ptr_begin(), len);
	B = A;

	ASSERT_EQ(len, B.nrows());
	ASSERT_EQ(1, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() == dblk.ptr_begin() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}

TEST( RefMatAssign, DMattoDCol )
{
	const index_t len = 5;
	scoped_block<double> dblk(len);

	dense_matrix<double> A(len, 1);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	ref_col<double> B(dblk.ptr_begin(), len);
	B = A;

	ASSERT_EQ(len, B.nrows());
	ASSERT_EQ(1, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() == dblk.ptr_begin() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}

TEST( RefMatAssign, DRowtoDRow )
{
	const index_t len = 5;
	scoped_block<double> sblk(len);
	scoped_block<double> dblk(len);

	ref_row<double> A(sblk.ptr_begin(), len);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	ref_row<double> B(dblk.ptr_begin(), len);
	B = A;

	ASSERT_EQ(1, B.nrows());
	ASSERT_EQ(len, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() == dblk.ptr_begin() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}

TEST( RefMatAssign, DMattoDRow )
{
	const index_t len = 5;
	scoped_block<double> dblk(len);

	dense_matrix<double> A(1, len);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	ref_row<double> B(dblk.ptr_begin(), len);
	B = A;

	ASSERT_EQ(1, B.nrows());
	ASSERT_EQ(len, B.ncolumns());

	ASSERT_TRUE( B.ptr_data() == dblk.ptr_begin() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}







template<int SrcRows, int SrcCols, int DstRows, int DstCols>
void test_refmat_ex_to_refmat_ex_assign(index_t m, index_t n, index_t lda, index_t ldb)
{
	scoped_block<double> blk_a(lda * n, 0.0);
	scoped_block<double> blk_b(ldb * n, 0.0);

	double *pa = blk_a.ptr_begin();
	double *pb = blk_b.ptr_begin();

	ref_matrix_ex<double, SrcRows, SrcCols> A(pa, m, n, lda);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	ref_matrix_ex<double, DstRows, DstCols> B(pb, m, n, ldb);

	B = A;

	ASSERT_TRUE( A.ptr_data() == pa );
	ASSERT_TRUE( B.ptr_data() == pb );

	ASSERT_EQ( m, B.nrows() );
	ASSERT_EQ( n, B.ncolumns() );
	ASSERT_EQ( ldb, B.lead_dim() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}


TEST( RefMatExAssign, DDtoDD ) { test_refmat_ex_to_refmat_ex_assign<0, 0, 0, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, DDtoDS ) { test_refmat_ex_to_refmat_ex_assign<0, 0, 0, 4>(3, 4, 5, 6); }
TEST( RefMatExAssign, DDtoSD ) { test_refmat_ex_to_refmat_ex_assign<0, 0, 3, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, DDtoSS ) { test_refmat_ex_to_refmat_ex_assign<0, 0, 3, 4>(3, 4, 5, 6); }

TEST( RefMatExAssign, DStoDD ) { test_refmat_ex_to_refmat_ex_assign<0, 4, 0, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, DStoDS ) { test_refmat_ex_to_refmat_ex_assign<0, 4, 0, 4>(3, 4, 5, 6); }
TEST( RefMatExAssign, DStoSD ) { test_refmat_ex_to_refmat_ex_assign<0, 4, 3, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, DStoSS ) { test_refmat_ex_to_refmat_ex_assign<0, 4, 3, 4>(3, 4, 5, 6); }

TEST( RefMatExAssign, SDtoDD ) { test_refmat_ex_to_refmat_ex_assign<3, 0, 0, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, SDtoDS ) { test_refmat_ex_to_refmat_ex_assign<3, 0, 0, 4>(3, 4, 5, 6); }
TEST( RefMatExAssign, SDtoSD ) { test_refmat_ex_to_refmat_ex_assign<3, 0, 3, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, SDtoSS ) { test_refmat_ex_to_refmat_ex_assign<3, 0, 3, 4>(3, 4, 5, 6); }

TEST( RefMatExAssign, SStoDD ) { test_refmat_ex_to_refmat_ex_assign<3, 4, 0, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, SStoDS ) { test_refmat_ex_to_refmat_ex_assign<3, 4, 0, 4>(3, 4, 5, 6); }
TEST( RefMatExAssign, SStoSD ) { test_refmat_ex_to_refmat_ex_assign<3, 4, 3, 0>(3, 4, 5, 6); }
TEST( RefMatExAssign, SStoSS ) { test_refmat_ex_to_refmat_ex_assign<3, 4, 3, 4>(3, 4, 5, 6); }

template<int SrcRows, int SrcCols, int DstRows, int DstCols>
void test_densemat_to_refmat_assign(index_t m, index_t n)
{
	scoped_block<double> blk_b(m * n, 0.0);
	double *pb = blk_b.ptr_begin();

	dense_matrix<double, SrcRows, SrcCols> A(m, n);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );
	const double *pa = A.ptr_data();

	ref_matrix<double, DstRows, DstCols> B(pb, m, n);

	B = A;

	ASSERT_TRUE( A.ptr_data() == pa );
	ASSERT_TRUE( B.ptr_data() == pb );

	ASSERT_EQ( m, B.nrows() );
	ASSERT_EQ( n, B.ncolumns() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}

TEST( Dense2RefAssign, DDtoDD ) { test_densemat_to_refmat_assign<0, 0, 0, 0>(3, 4); }
TEST( Dense2RefAssign, DDtoDS ) { test_densemat_to_refmat_assign<0, 0, 0, 4>(3, 4); }
TEST( Dense2RefAssign, DDtoSD ) { test_densemat_to_refmat_assign<0, 0, 3, 0>(3, 4); }
TEST( Dense2RefAssign, DDtoSS ) { test_densemat_to_refmat_assign<0, 0, 3, 4>(3, 4); }

TEST( Dense2RefAssign, DStoDD ) { test_densemat_to_refmat_assign<0, 4, 0, 0>(3, 4); }
TEST( Dense2RefAssign, DStoDS ) { test_densemat_to_refmat_assign<0, 4, 0, 4>(3, 4); }
TEST( Dense2RefAssign, DStoSD ) { test_densemat_to_refmat_assign<0, 4, 3, 0>(3, 4); }
TEST( Dense2RefAssign, DStoSS ) { test_densemat_to_refmat_assign<0, 4, 3, 4>(3, 4); }

TEST( Dense2RefAssign, SDtoDD ) { test_densemat_to_refmat_assign<3, 0, 0, 0>(3, 4); }
TEST( Dense2RefAssign, SDtoDS ) { test_densemat_to_refmat_assign<3, 0, 0, 4>(3, 4); }
TEST( Dense2RefAssign, SDtoSD ) { test_densemat_to_refmat_assign<3, 0, 3, 0>(3, 4); }
TEST( Dense2RefAssign, SDtoSS ) { test_densemat_to_refmat_assign<3, 0, 3, 4>(3, 4); }

TEST( Dense2RefAssign, SStoDD ) { test_densemat_to_refmat_assign<3, 4, 0, 0>(3, 4); }
TEST( Dense2RefAssign, SStoDS ) { test_densemat_to_refmat_assign<3, 4, 0, 4>(3, 4); }
TEST( Dense2RefAssign, SStoSD ) { test_densemat_to_refmat_assign<3, 4, 3, 0>(3, 4); }
TEST( Dense2RefAssign, SStoSS ) { test_densemat_to_refmat_assign<3, 4, 3, 4>(3, 4); }


template<int SrcRows, int SrcCols, int DstRows, int DstCols>
void test_refmat_to_densemat_assign(index_t m, index_t n)
{
	scoped_block<double> blk_a(m * n);

	double *pa = blk_a.ptr_begin();

	ref_matrix<double, SrcRows, SrcCols> A(pa, m, n);
	fill_matrix_values(A, 2.0);
	ASSERT_TRUE( verify_matrix_values(A, 2.0) );

	dense_matrix<double, DstRows, DstCols> B;

	B = A;

	ASSERT_TRUE( A.ptr_data() == pa );
	ASSERT_TRUE( B.ptr_data() != pa );

	ASSERT_EQ( m, B.nrows() );
	ASSERT_EQ( n, B.ncolumns() );
	ASSERT_TRUE( verify_matrix_values(B, 2.0) );
}


TEST( Ref2DenseAssign, DDtoDD ) { test_refmat_to_densemat_assign<0, 0, 0, 0>(3, 4); }
TEST( Ref2DenseAssign, DDtoDS ) { test_refmat_to_densemat_assign<0, 0, 0, 4>(3, 4); }
TEST( Ref2DenseAssign, DDtoSD ) { test_refmat_to_densemat_assign<0, 0, 3, 0>(3, 4); }
TEST( Ref2DenseAssign, DDtoSS ) { test_refmat_to_densemat_assign<0, 0, 3, 4>(3, 4); }

TEST( Ref2DenseAssign, DStoDD ) { test_refmat_to_densemat_assign<0, 4, 0, 0>(3, 4); }
TEST( Ref2DenseAssign, DStoDS ) { test_refmat_to_densemat_assign<0, 4, 0, 4>(3, 4); }
TEST( Ref2DenseAssign, DStoSD ) { test_refmat_to_densemat_assign<0, 4, 3, 0>(3, 4); }
TEST( Ref2DenseAssign, DStoSS ) { test_refmat_to_densemat_assign<0, 4, 3, 4>(3, 4); }

TEST( Ref2DenseAssign, SDtoDD ) { test_refmat_to_densemat_assign<3, 0, 0, 0>(3, 4); }
TEST( Ref2DenseAssign, SDtoDS ) { test_refmat_to_densemat_assign<3, 0, 0, 4>(3, 4); }
TEST( Ref2DenseAssign, SDtoSD ) { test_refmat_to_densemat_assign<3, 0, 3, 0>(3, 4); }
TEST( Ref2DenseAssign, SDtoSS ) { test_refmat_to_densemat_assign<3, 0, 3, 4>(3, 4); }

TEST( Ref2DenseAssign, SStoDD ) { test_refmat_to_densemat_assign<3, 4, 0, 0>(3, 4); }
TEST( Ref2DenseAssign, SStoDS ) { test_refmat_to_densemat_assign<3, 4, 0, 4>(3, 4); }
TEST( Ref2DenseAssign, SStoSD ) { test_refmat_to_densemat_assign<3, 4, 3, 0>(3, 4); }
TEST( Ref2DenseAssign, SStoSS ) { test_refmat_to_densemat_assign<3, 4, 3, 4>(3, 4); }












