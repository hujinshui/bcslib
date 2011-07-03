/**
 * @file test_basic_memory.cpp
 *
 * Unit testing of basic memory facilities
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>

#include <bcslib/base/basic_mem.h>
#include <bcslib/base/block.h>
#include <bcslib/base/monitored_allocator.h>
#include <vector>

using namespace bcs;
using namespace bcs::test;

// explicit template instantiation for syntax check

template class bcs::monitored_allocator<int>;
template class bcs::aligned_allocator<int>;
template class bcs::cref_blk_t<int>;
template class bcs::ref_blk_t<int>;
template class bcs::copy_blk_t<int>;
template class bcs::const_block<int, aligned_allocator<int> >;
template class bcs::block<int, aligned_allocator<int> >;
template class bcs::const_block<int, monitored_allocator<int> >;
template class bcs::block<int, monitored_allocator<int> >;

typedef class bcs::const_block<int, monitored_allocator<int> > cblk_t;
typedef class bcs::block<int, monitored_allocator<int> > blk_t;


class MyInt
{
public:
	explicit MyInt(int v) : p_v(new int(v)) { ++ s_nobjs; }

	~MyInt() { delete p_v; -- s_nobjs; }

	explicit MyInt(const MyInt& r) : p_v(new int(r.value())) { ++ s_nobjs; }

	MyInt& operator = (const MyInt& r)
	{
		if (this != &r)
		{
			int *tmp = new int(r.value());
			delete p_v;
			p_v = tmp;
		}
		return *this;
	}

	int value() const
	{
		return *p_v;
	}

	bool operator == (const MyInt& r) const
	{
		return value() == r.value();
	}

	bool operator != (const MyInt& r) const
	{
		return value() != r.value();
	}

private:
	int *p_v;

public:
	static size_t num_objects()
	{
		return s_nobjs;
	}

private:
	static size_t s_nobjs;

};

size_t MyInt::s_nobjs = 0;


template<typename T, class Allocator>
inline static bool test_block(const const_block<T, Allocator>& blk, bool own, size_t n, const T *pbase = BCS_NULL)
{
	return blk.own_memory() == own && blk.nelems() == n && (blk.pend() == blk.pbase() + n) &&
			(pbase == BCS_NULL || blk.pbase() == pbase);
}

template<typename T, class Allocator>
inline static bool test_block(const block<T, Allocator>& blk, bool own, size_t n, const T *pbase = BCS_NULL)
{
	return blk.own_memory() == own && blk.nelems() == n && (blk.pend() == blk.pbase() + n) &&
			(pbase == BCS_NULL || blk.pbase() == pbase);
}



BCS_TEST_CASE( mem_test_clean_start )
{
	BCS_CHECK( !global_memory_allocation_monitor.has_pending() );
}


BCS_TEST_CASE( test_basic_memory_operations )
{
	// POD

	const size_t N = 5;

	int src0[N] = {1, 3, 4, 5, 2};
	const int *src = src0;

	aligned_allocator<int> alloc;

	int *p = alloc.allocate(N, BCS_NULL);
	for (size_t i = 0; i < N; ++i) p[i] = 0;

	copy_construct_elements(src, p, N);
	BCS_CHECK( collection_equal(p, p+N, src, N) );
	BCS_CHECK( elements_equal(src, p, N) );
	p[2] = 0;
	BCS_CHECK( !elements_equal(src, p, N) );

	fill_elements(p, N, 7);
	BCS_CHECK( elements_equal(7, p, N) );
	p[2] = 0;
	BCS_CHECK( !elements_equal(7, p, N) );

	copy_elements(src, p, N);
	BCS_CHECK( elements_equal(p, src, N) );

	set_zeros_to_elements(p, N);
	BCS_CHECK( elements_equal(0, p, N) );

	alloc.deallocate(p, N);

	// non-POD

	BCS_CHECK_EQUAL( MyInt::num_objects(), 0 );

	std::vector<MyInt> objvec;
	objvec.push_back( MyInt(1) );
	objvec.push_back( MyInt(3) );
	objvec.push_back( MyInt(4) );
	objvec.push_back( MyInt(5) );
	objvec.push_back( MyInt(2) );

	BCS_CHECK_EQUAL( MyInt::num_objects(), N );
	const MyInt *obj_src = objvec.data();

	aligned_allocator<MyInt> obj_alloc;

	MyInt *q = obj_alloc.allocate(N, BCS_NULL);
	BCS_CHECK_EQUAL( MyInt::num_objects(), N );

	copy_construct_elements(obj_src, q, N);
	BCS_CHECK_EQUAL( MyInt::num_objects(), 2 * N );
	BCS_CHECK( elements_equal(q, obj_src, N) );
	q[2] = MyInt(0);
	BCS_CHECK( !elements_equal(q, obj_src, N) );

	fill_elements(q, N, MyInt(1));
	BCS_CHECK_EQUAL( MyInt::num_objects(), 2 * N );
	BCS_CHECK( elements_equal(MyInt(1), q, N) );
	q[2] = MyInt(0);
	BCS_CHECK( !elements_equal(MyInt(1), q, N) );

	copy_elements(obj_src, q, N);
	BCS_CHECK_EQUAL( MyInt::num_objects(), 2 * N );
	BCS_CHECK( elements_equal(q, obj_src, N) );

	destruct_elements(q, N);
	BCS_CHECK_EQUAL( MyInt::num_objects(), N );

	obj_alloc.deallocate(q, N);

	objvec.clear();
	BCS_CHECK_EQUAL( MyInt::num_objects(), 0 );
}


#define CHECK_MEM_PENDING(k) BCS_CHECK_EQUAL( global_memory_allocation_monitor.num_pending_sections(), k )

BCS_TEST_CASE( test_const_blocks )
{
	using std::swap;

	const int *pnull = static_cast<const int*>(BCS_NULL);

	const size_t N = 5;
	int src[N] = {4, 3, 1, 2, 5};

	CHECK_MEM_PENDING( 0 );

	cblk_t B0(cref_blk(src, N));
	BCS_CHECK( test_block(B0, false, N, src) );

	cblk_t B1(N, int(2));
	BCS_CHECK( test_block(B1, true, N) );
	BCS_CHECK( elements_equal(int(2), B1.pbase(), N) );

	CHECK_MEM_PENDING( 1 );

	cblk_t B2(copy_blk(src, N));
	BCS_CHECK( test_block(B2, true, N) );
	BCS_CHECK( B2.pbase() != src );
	BCS_CHECK( elements_equal(B2.pbase(), src, N) );

	CHECK_MEM_PENDING( 2 );

	cblk_t B0c(B0);
	BCS_CHECK( test_block(B0, false, N, src) );
	BCS_CHECK( test_block(B0c, false, N, src) );

	cblk_t B0m(std::move(B0c));
	BCS_CHECK( test_block(B0c, false, 0, pnull) );
	BCS_CHECK( test_block(B0m, false, N, src) );

	CHECK_MEM_PENDING( 2 );

	const int *p2 = B2.pbase();
	cblk_t B2c(B2);
	BCS_CHECK( test_block(B2, true, N, p2) );
	BCS_CHECK( test_block(B2c, true, N) );
	BCS_CHECK( B2c.pbase() != p2 );
	BCS_CHECK( elements_equal(B2.pbase(), B2c.pbase(), N) );

	CHECK_MEM_PENDING( 3 );

	const int *p2c = B2c.pbase();
	cblk_t B2m(std::move(B2c));
	BCS_CHECK( test_block(B2m, true, N, p2c) );
	BCS_CHECK( test_block(B2c, false, 0, pnull) );

	CHECK_MEM_PENDING( 3 );

	const int *p1 = B1.pbase();
	const int *p2m = B2m.pbase();

	swap(B1, B2m);
	swap(p1, p2m);
	BCS_CHECK( test_block(B1, true, N, p1) );
	BCS_CHECK( test_block(B2m, true, N, p2m) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	BCS_CHECK( test_block(B0, true, N, p1) );
	BCS_CHECK( test_block(B1, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	BCS_CHECK( test_block(B1, true, N, p1) );
	BCS_CHECK( test_block(B0, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	B2c = std::move(B1);
	p2c = p1;
	p1 = BCS_NULL;

	BCS_CHECK( test_block(B1, false, 0, pnull) );
	BCS_CHECK( test_block(B2c, true, N, p2c) );

	CHECK_MEM_PENDING( 3 );

	B2c = std::move(B0);
	p2c = src;

	BCS_CHECK( test_block(B0, false, 0, pnull) );
	BCS_CHECK( test_block(B2c, false, N, p2c) );

	CHECK_MEM_PENDING( 2 );

	B2c = std::move(B2m);
	p2c = p2m;
	p2m = BCS_NULL;

	BCS_CHECK( test_block(B2c, true, N, p2c) );
	BCS_CHECK( test_block(B2m, false, 0, pnull) );

	CHECK_MEM_PENDING( 2 );
}

BCS_TEST_CASE( test_blocks )
{
	using std::swap;

	const int *pnull = static_cast<const int*>(BCS_NULL);

	const size_t N = 5;
	int src[N] = {4, 3, 1, 2, 5};

	CHECK_MEM_PENDING( 0 );

	blk_t B0( ref_blk(src, N));
	BCS_CHECK( test_block(B0, false, N, src) );

	blk_t B1(N, int(2));
	BCS_CHECK( test_block(B1, true, N) );
	BCS_CHECK( elements_equal(int(2), B1.pbase(), N) );

	CHECK_MEM_PENDING( 1 );

	blk_t B2(copy_blk(src, N));
	BCS_CHECK( test_block(B2, true, N) );
	BCS_CHECK( B2.pbase() != src );
	BCS_CHECK( elements_equal(B2.pbase(), src, N) );

	CHECK_MEM_PENDING( 2 );

	blk_t B0c(B0);
	BCS_CHECK( test_block(B0, false, N, src) );
	BCS_CHECK( test_block(B0c, false, N, src) );

	blk_t B0m(std::move(B0c));
	BCS_CHECK( test_block(B0c, false, 0, pnull) );
	BCS_CHECK( test_block(B0m, false, N, src) );

	CHECK_MEM_PENDING( 2 );

	const int *p2 = B2.pbase();
	blk_t B2c(B2);
	BCS_CHECK( test_block(B2, true, N, p2) );
	BCS_CHECK( test_block(B2c, true, N) );
	BCS_CHECK( B2c.pbase() != p2 );
	BCS_CHECK( elements_equal(B2.pbase(), B2c.pbase(), N) );

	CHECK_MEM_PENDING( 3 );

	const int *p2c = B2c.pbase();
	blk_t B2m(std::move(B2c));
	BCS_CHECK( test_block(B2m, true, N, p2c) );
	BCS_CHECK( test_block(B2c, false, 0, pnull) );

	CHECK_MEM_PENDING( 3 );

	const int *p1 = B1.pbase();
	const int *p2m = B2m.pbase();

	swap(B1, B2m);
	swap(p1, p2m);
	BCS_CHECK( test_block(B1, true, N, p1) );
	BCS_CHECK( test_block(B2m, true, N, p2m) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	BCS_CHECK( test_block(B0, true, N, p1) );
	BCS_CHECK( test_block(B1, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	BCS_CHECK( test_block(B1, true, N, p1) );
	BCS_CHECK( test_block(B0, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	B2c = std::move(B1);
	p2c = p1;
	p1 = BCS_NULL;

	BCS_CHECK( test_block(B1, false, 0, pnull) );
	BCS_CHECK( test_block(B2c, true, N, p2c) );

	CHECK_MEM_PENDING( 3 );

	B2c = std::move(B0);
	p2c = src;

	BCS_CHECK( test_block(B0, false, 0, pnull) );
	BCS_CHECK( test_block(B2c, false, N, p2c) );

	CHECK_MEM_PENDING( 2 );

	B2c = std::move(B2m);
	p2c = p2m;
	p2m = BCS_NULL;

	BCS_CHECK( test_block(B2c, true, N, p2c) );
	BCS_CHECK( test_block(B2m, false, 0, pnull) );

	CHECK_MEM_PENDING( 2 );

	blk_t Be(10);

	BCS_CHECK( test_block(Be, true, 10) );

	CHECK_MEM_PENDING( 3 );
}


#undef CHECK_MEM_PENDING


BCS_TEST_CASE( mem_test_clean_end )
{
	BCS_CHECK( !global_memory_allocation_monitor.has_pending() );
}




test_suite* test_basic_memory_suite()
{
	test_suite *suite = new test_suite( "test_basic_memory" );

	suite->add( new mem_test_clean_start() );

	suite->add( new test_basic_memory_operations() );
	suite->add( new test_const_blocks() );
	suite->add( new test_blocks() );

	suite->add( new mem_test_clean_end() );

	return suite;
}

