/**
 * @file test_c0x_support.cpp
 *
 * A file to test C++0x support
 *
 * @remarks: if this file can be compiled successfully,
 *           then all C++0x features required by this library is ready
 * 
 * The features tested in this file:
 *
 * - static_assert
 * - standard type_traits (not including new POD stuff, such as standard_layout)
 * - std::function and std::bind (and placeholders)
 * - rvalue, move semantics, and perfect forwarding
 * - tuple and fixed array
 * - new containers: forward_list, unordered_map, unordered_set
 * - new algorithms
 * - move_iterator
 *
 * @author Dahua Lin
 */

#include <type_traits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <functional>
#include <utility>
#include <algorithm>
#include <iterator>
#include <tuple>
#include <array>
#include <forward_list>
#include <unordered_map>
#include <unordered_set>

#define MY_STATIC_ASSERT_C(cond) static_assert(cond::value, #cond)

using namespace std;

template<typename T>
struct rgb
{
	T r;
	T g;
	T b;
};

template<typename T>
inline T add_one(T x)
{
	return x + 1;
}


int myfun2(int*, char*)
{
	return 0;
}

int myfun3(int*, int*, char*)
{
	return 0;
}


template<typename T1, typename T2>
int mygfun2(T1, T2)
{
	return 0;
}



template<typename T>
class rgb_ex
{
public:
	rgb_ex(T r_, T g_, T b_) : r(r_), g(g_), b(b_) { }

	T red() const { return r; }
	T green() const { return g; }
	T blue() const { return b;}

private:
	T r;
	T g;
	T b;
};


enum weekdays
{
	SUNDAY = 0,
	MONDAY = 1,
	TUESDAY = 2,
	WEDNESDAY = 3,
	THURSDAY = 4,
	FRIDAY = 5,
	SATURDAY = 6
};

struct my_tag_t { };


void test_static_assert_and_type_traits()
{
	MY_STATIC_ASSERT_C(is_void<void>);
	MY_STATIC_ASSERT_C(is_integral<int>);
	MY_STATIC_ASSERT_C(is_floating_point<double>);
	MY_STATIC_ASSERT_C(is_array<int[]>);
	MY_STATIC_ASSERT_C(is_pointer<int*>);
	MY_STATIC_ASSERT_C(is_lvalue_reference<int&>);
	MY_STATIC_ASSERT_C(is_rvalue_reference<int&&>);
	MY_STATIC_ASSERT_C(is_enum<weekdays>);
	MY_STATIC_ASSERT_C(is_class<rgb<int>>);
	MY_STATIC_ASSERT_C(is_function<void(void)>);

	MY_STATIC_ASSERT_C(is_reference<int&>);
	MY_STATIC_ASSERT_C(is_reference<int&&>);
	MY_STATIC_ASSERT_C(is_arithmetic<float>);
	MY_STATIC_ASSERT_C(is_object<int>);
	MY_STATIC_ASSERT_C(is_object<rgb<int>>);

	MY_STATIC_ASSERT_C(is_const<const char>);
	MY_STATIC_ASSERT_C(is_empty<my_tag_t>);
	MY_STATIC_ASSERT_C(is_pod<rgb<int>>);
	MY_STATIC_ASSERT_C(is_signed<int>);
	MY_STATIC_ASSERT_C(is_unsigned<unsigned int>);

	static_assert(is_same<int, int>::value, "is_same<int, int>::value");
	static_assert(is_convertible<float, double>::value, "is_convertible<float, double>::value");
}


void test_type_transforms()
{
	static_assert(
			is_same<remove_const<const int>::type, int>::value,
			"is_same<remove_const<const int>::type, int>::value");
	static_assert(
			is_same<remove_const<const int>::type, int>::value,
			"is_same<remove_const<const int>::type, int>::value");
	static_assert(
			is_same<add_const<int*>::type, int* const>::value,
			"is_same<add_const<int*>::type, int* const>::value");

	static_assert(
			is_same<make_signed<unsigned int>::type, int>::value,
			"is_same<make_signed<unsigned int>::type, int>::value");
	static_assert(
			is_same<make_unsigned<const int>::type, const unsigned int>::value,
			"is_same<make_unsigned<const int>::type, const unsigned int>::value");

	static_assert(
			is_same<enable_if<is_integral<int>::value>::type, void>::value,
			"enable_if testing (1)");
	static_assert(
			is_same<enable_if<is_integral<int>::value, double>::type, double>::value,
			"enable_if testing (2)");

	static_assert(
			is_same<conditional<true, char, int>::type, char>::value,
			"conditional testing (true)");
	static_assert(
			is_same<conditional<false, char, int>::type, int>::value,
			"conditional testing (false)");

	static_assert(
			is_same<result_of<plus<int>(int, int)>::type, int>::value,
			"result_of testing");
}


void test_auto_and_decltype()
{
	int a = 1;
	auto x = rgb<int>();
	static_assert(is_same<decltype(x), rgb<int>>::value, "auto-decltype testing 1");
	static_assert(is_same<decltype(add_one(a)), int>::value, "auto-decltype testing 2");
}


void test_function_and_binding()
{
	using namespace std::placeholders;

	int i = 0;
	char ch = 'a';

	function<double (short, char)> f1( mygfun2<int, char> );
	function<int (int*)> f2( bind( myfun2, _1, &ch) );
	function<int (char*)> f3( bind( myfun2, &i, _1));
	function<int (char*, int*)> f4( bind( myfun2, _2, _1) );
	function<double (int*)> f5( bind( myfun3, _1, _1, &ch) );
}


template<typename T>
class MyValue
{
public:
	typedef T value_type;

public:
	MyValue(const value_type& v)
	: m_ptrval(new value_type(v))
	{ }

	MyValue(const MyValue& src)
	: m_ptrval(new value_type(src.value()))
	{ }

	MyValue(MyValue&& src)
	: m_ptrval(move(src.m_ptrval))
	{
	}

	MyValue& operator = (const MyValue& rhs)
	{
		m_ptrval.reset(new value_type(rhs.value()));
		return *this;
	}

	MyValue& operator = (MyValue&& rhs)
	{
		m_ptrval.swap(rhs.m_ptrval);
		rhs.m_ptrval.release();
		return *this;
	}

	const value_type& value() const
	{
		return *m_ptrval;
	}

public:
	static MyValue cond_create(bool cond, const value_type& vtrue, const value_type& vfalse)
	{
		if (cond)
		{
			return MyValue(vtrue);
		}
		else
		{
			return MyValue(vfalse);
		}
	}

private:
	unique_ptr<value_type> m_ptrval;
};

template class MyValue<double>;

void test_rvalue_and_move_semantics(bool cond)
{
	MyValue<double> v1(1.0);
	MyValue<double> v2(v1);
	MyValue<double> v3(MyValue<double>::cond_create(cond, 2.0, 3.0));

	v3 = v2;
	v2 = MyValue<double>::cond_create(cond, 0.0, 1.0);
	v1 = move(v3);
}


void test_forwarding(MyValue<double>&& v)
{
	MyValue<double> v1(forward<MyValue<double>>(v));
}



int test_tuple_and_fixed_array()
{
	tuple<int, char> t(1, 'a');

	static_assert(tuple_size<tuple<int, char, int>>::value == 3, "tuple_size testing on tuple");
	static_assert(
			is_same<tuple_element<0, tuple<int, char>>::type, int>::value,
			"tuple_element testing [0]");
	static_assert(
			is_same<tuple_element<1, tuple<int, char>>::type, char>::value,
			"tuple_element testing [1]");

	array<int, 3> a;
	static_assert(tuple_size<array<int, 5>>::value == 5, "tuple_size testing on array");

	static_assert(
			is_same<tuple_element<0, array<int, 3>>::type, int>::value,
			"tuple_element testing [1]");

	return a[0];
}


void test_new_containers()
{
	forward_list<int> a;
	a.insert_after(a.end(), 1);

	unordered_set<int> s;
	s.insert(1);

	unordered_multiset<int> sm;
	sm.insert(1);

	unordered_map<int, char> m;
	m.insert(make_pair(1, 'a'));

	unordered_multimap<int, char> mm;
	mm.insert(make_pair(1, 'a'));
}


void test_new_algorithms()
{
	using namespace std::placeholders;

	int *pint = 0;
	int *pint2 = 0;
	auto pred = bind(greater<int>(), _1, 0);


	copy_n(pint, 10, pint2);
	copy_if(pint, pint+10, pint2, pred);
	partition_copy(pint, pint+10, pint2, pint2+10, pred);
	is_partitioned(pint, pint+10, pred);

	move(pint, pint+10, pint2);
	move_backward(pint, pint+10, pint2);

	is_sorted(pint, pint+10);
	is_sorted_until(pint, pint+10);
	is_heap(pint, pint+10);
	is_heap_until(pint, pint+10);

	find_if_not(pint, pint2, pred);
	mismatch(pint, pint+10, pint2);

	all_of(pint, pint2, pred);
	any_of(pint, pint2, pred);
	none_of(pint, pint2, pred);

	minmax(pint, pint+10);
	minmax_element(pint, pint+10);
}


void test_move_iterator()
{
	int *pint = 0;
	int *pint2 = 0;

	copy_n(make_move_iterator(pint), 10, pint2);
}




