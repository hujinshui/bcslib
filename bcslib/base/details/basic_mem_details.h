/**
 * @file basic_mem_details.h
 *
 * The details for basic_mem.h
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BASIC_MEM_DETAILS_H_
#define BCSLIB_BASIC_MEM_DETAILS_H_

#include <bcslib/base/basic_defs.h>

namespace bcs
{
	namespace _detail
	{

		/**************************************************
		 *
		 *   helpers for memory operations
		 *
		 **************************************************/

		template<typename T, bool IsTriviallyConstructible> struct _element_construct_helper;

		template<typename T>
		struct _element_construct_helper<T, true>
		{
			static void default_construct(T *, size_t) { }
		};

		template<typename T>
		struct _element_construct_helper<T, false>
		{
			static void default_construct(T *p, size_t n)
			{
				while (n --) { new (p++) T(); }
			}
		};

		template<typename T, bool IsTriviallyCopyable> struct _element_copy_helper;

		template<typename T>
		struct _element_copy_helper<T, true>
		{
			static void copy(const T *src, T *dst, size_t n)
			{
				if (n > 0) std::memcpy(dst, src, sizeof(T) * n);
			}

			static void copy_construct(const T *src, T *dst, size_t n)
			{
				if (n > 0) std::memcpy(dst, src, sizeof(T) * n);
			}

			static void copy_construct(const T& v, T *dst, size_t n)
			{
				while (n--) { *(dst++) = v; }
			}
		};

		template<typename T>
		struct _element_copy_helper<T, false>
		{
			static void copy(const T *src, T *dst, size_t n)
			{
				while (n--) { *(dst++) = *(src++); }
			}

			static void copy_construct(const T *src, T *dst, size_t n)
			{
				while (n--) { new (dst++) T(*(src++)); }
			}

			static void copy_construct(const T& v, T *dst, size_t n)
			{
				while (n--) { new (dst++) T(v); }
			}
		};


		template<typename T, bool IsTriviallyDestructible> struct _element_destruct_helper;

		template<typename T>
		struct _element_destruct_helper<T, true>
		{
			static void destruct(T *, size_t) { }
		};

		template<typename T>
		struct _element_destruct_helper<T, false>
		{
			static void destruct(T *dst, size_t n)
			{
				while (n--) { (dst++)->~T(); }
			}
		};


		template<typename T, bool IsBitwiseComparable> struct _element_compare_helper;

		template<typename T>
		struct _element_compare_helper<T, true>
		{
			static bool all_equal(const T *a, const T *b, size_t n)
			{
				return n == 0 || std::memcmp(a, b, sizeof(T) * n) == 0;
			}
		};

		template<typename T>
		struct _element_compare_helper<T, false>
		{
			static bool all_equal(const T *a, const T *b, size_t n)
			{
				for (size_t i = 0; i < n; ++i)
				{
					if (a[i] != b[i]) return false;
				}
				return true;
			}
		};


	}
}

#endif
