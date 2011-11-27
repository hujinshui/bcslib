/**
 * @file basic_defs.h
 *
 * Library-wide basic definitions
 *
 * @author dhlin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BASIC_DEFS_H
#define BCSLIB_BASIC_DEFS_H

#include <bcslib/base/config.h>

#include <cstddef>
#include <stdint.h>
#include <utility>

// This is for temporary use,
// and will be replaced with nullptr when it is available in most compilers


#define BCS_NULL NULL

namespace bcs
{

	using ::int8_t;
	using ::int16_t;
	using ::int32_t;
	using ::int64_t;

	using ::uint8_t;
	using ::uint16_t;
	using ::uint32_t;
	using ::uint64_t;

	using std::ptrdiff_t;
	using std::size_t;

	typedef ptrdiff_t index_t;

	using std::pair;
	using std::make_pair;

	template<typename T1, typename T2>
	inline pair<T1&, T2&> tie_pair(T1& t1, T2& t2)
	{
		return pair<T1&, T2&>(t1, t2);
	}


	/**
	 * means no type is there
	 */
	struct nil_type { };

	/**
	 * The base class to make sure its derived classes are non-copyable
	 */
	class noncopyable
	{
	protected:
		noncopyable() { }
		~noncopyable() { }

	private:
		noncopyable(const noncopyable& );
		noncopyable& operator= (const noncopyable& );
	};


	/**
	 * type selection
	 */
	template<bool B, typename T1, typename T2> struct tyselect;

	template<typename T1, typename T2>
	struct tyselect<true, T1, T2>
	{
		typedef T1 type;
	};

	template<typename T1, typename T2>
	struct tyselect<false, T1, T2>
	{
		typedef T2 type;
	};


	/**
	 *  Auxiliary class for static assertion
	 */

	template<bool x> struct BCSLIB_STATIC_ASSERT_FAILURE;
	template<> struct BCSLIB_STATIC_ASSERT_FAILURE<true> { enum { value = 1 }; };
	template<int x> struct static_assert_test { };

}


// useful macros

#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
#define BCS_ALIGN(a) __declspec(align(a))
#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
#define BCS_ALIGN(a) __attribute__((aligned(a)))
#endif

#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
#define BCS_ENSURE_INLINE __forceinline
#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
#define BCS_ENSURE_INLINE __attribute__((always_inline))
#endif

// static assertion

#define BCS_CONCAT_STR(a, b) BCS_DO_CONCAT_STR(a, b)
#define BCS_DO_CONCAT_STR(a, b) BCS_DO_CONCAT_STR2(a, b)
#define BCS_DO_CONCAT_STR2(a, b) a##b

#if BCSLIB_COMPILER == BCSLIB_GCC && __GNUC_MINOR__ < 4
#define BCS_STATIC_ASSERT( cond ) \
   typedef ::bcs::static_assert_test< sizeof(::bcs::BCSLIB_STATIC_ASSERT_FAILURE< cond >) > \
         BCS_CONCAT_STR(bcslib_static_assert_typedef_, __LINE__)
#else
#define BCS_STATIC_ASSERT( cond ) \
   typedef ::bcs::static_assert_test< sizeof(::bcs::BCSLIB_STATIC_ASSERT_FAILURE< cond >) > \
         BCS_CONCAT_STR(bcslib_static_assert_typedef_, __COUNTER__)
#endif

#define BCS_STATIC_ASSERT_V(cond) BCS_STATIC_ASSERT(cond::value)

#endif



