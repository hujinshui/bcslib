/*
 * @file syntax.h
 *
 * Some useful syntax construction
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_SYNTAX_H_
#define BCSLIB_SYNTAX_H_

#include <bcslib/config/config.h>

namespace bcs
{
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


	template<typename T, bool IsReadOnly> struct access_types;

	template<typename T>
	struct access_types<T, false>
	{
		typedef T* pointer;
		typedef T& reference;
	};

	template<typename T>
	struct access_types<T, true>
	{
		typedef const T* pointer;
		typedef const T& reference;
	};

}


#define BCS_CRTP_REF \
		BCS_ENSURE_INLINE const Derived& derived() const { return *(static_cast<const Derived*>(this)); } \
		BCS_ENSURE_INLINE Derived& derived() { return *(static_cast<Derived*>(this)); }


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


#endif 
