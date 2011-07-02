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
				for (size_t i = 0; i < n; ++i)
				{
					new (p + i) T();  // construction via placement new
				}
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
				for (size_t i = 0; i < n; ++i)
				{
					dst[i] = v;
				}
			}
		};

		template<typename T>
		struct _element_copy_helper<T, false>
		{
			static void copy(const T *src, T *dst, size_t n)
			{
				for (size_t i = 0; i < n; ++i)
				{
					dst[i] = src[i];
				}
			}

			static void copy_construct(const T *src, T *dst, size_t n)
			{
				for (size_t i = 0; i < n; ++i)
				{
					new (dst + i) T(src[i]);
				}
			}

			static void copy_construct(const T& v, T *dst, size_t n)
			{
				for (size_t i = 0; i < n; ++i)
				{
					new (dst + i) T(v);
				}
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
				for (size_t i = 0; i < n; ++i)
				{
					(dst+i)->~T();
				}
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


		/**************************************************
		 *
		 *   The internal implementation of block
		 *
		 **************************************************/

    	template<typename T, typename Allocator>
    	class block_impl
    	{
    	public:
    		typedef T value_type;
    		typedef Allocator allocator_type;

    		typedef typename allocator_type::size_type size_type;
    		typedef typename allocator_type::difference_type difference_type;

    		typedef typename allocator_type::pointer pointer;
    		typedef typename allocator_type::reference reference;
    		typedef typename allocator_type::const_pointer const_pointer;
    		typedef typename allocator_type::const_reference const_reference;

    	public:
    		block_impl(pointer p, size_type n)
    		: m_allocator()
    		, m_base(p), m_n(n), m_own(false)
    		{
    		}

    		explicit block_impl(size_type n)
    		: m_allocator()
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
    			default_construct_elements(m_base, n);
    		}

    		block_impl(size_type n, const allocator_type& allocator)
    		: m_allocator(allocator)
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
    			default_construct_elements(m_base, n);
    		}

    		block_impl(size_type n, const_reference v)
    		: m_allocator()
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
    			if (n > 0) copy_construct_elements(v, m_base, n);
    		}

    		block_impl(size_type n, const_reference v, const allocator_type& allocator)
    		: m_allocator(allocator)
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
    			if (n > 0) copy_construct_elements(v, m_base, n);
    		}

    		block_impl(size_type n, const_pointer src)
    		: m_allocator()
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
    			if (n > 0) copy_construct_elements(src, m_base, n);
    		}

    		block_impl(size_type n, const_pointer src, const allocator_type& allocator)
    		: m_allocator(allocator)
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
    			if (n > 0) copy_construct_elements(src, m_base, n);
    		}

    		~block_impl()
    		{
    			if (m_own)
    			{
    				destruct_elements(m_base, m_n);
    				safe_deallocate(m_allocator, m_base, m_n);
    			}
    		}

    		void release()
    		{
    			if (m_own)
    			{
    				safe_deallocate(m_allocator, m_base, m_n);
    			}

    			m_base = BCS_NULL;
    			m_n = 0;
    			m_own = false;
    		}

    		block_impl(const block_impl& r)
    		: m_allocator(r.m_allocator)
    		, m_base(r.m_own ? safe_allocate(m_allocator, r.m_n) : r.m_base)
    		, m_n(r.m_n)
    		, m_own(r.m_own)
    		{
    			if (r.m_n > 0) copy_construct_elements(r.m_base, m_base, r.m_n);
    		}

    		block_impl(block_impl&& r)
    		: m_allocator(std::move(r.m_allocator))
    		, m_base(r.m_base)
    		, m_n(r.m_n)
    		, m_own(r.m_own)
    		{
    			r.m_base = BCS_NULL;
    			r.m_n = 0;
    			r.m_own = false;
    		}

    		void swap(block_impl& r)
    		{
    			using std::swap;

    			swap(m_allocator, r.m_allocator);
    			swap(m_base, r.m_base);
    			swap(m_n, r.m_n);
    			swap(m_own, r.m_own);
    		}

    		void operator = (const block_impl& r)
    		{
    			if (this !=  &r)
    			{
    				block_impl tmp(r);
    				swap(tmp);
    			}
    		}

    		void operator = (block_impl&& r)
    		{
    			swap(r);
    			r.release();
    		}

    	public:
    		size_type nelems() const
    		{
    			return m_n;
    		}

    		const_pointer pbase() const
    		{
    			return m_base;
    		}

    		pointer pbase()
    		{
    			return m_base;
    		}

    		const_reference get(size_type i) const
    		{
    			return m_base[i];
    		}

    		reference get(size_type i)
    		{
    			return m_base[i];
    		}

    		const allocator_type& get_allocator() const
    		{
    			return m_allocator;
    		}

    		bool own_memory() const
    		{
    			return m_own;
    		}

    	private:
    		allocator_type m_allocator;
    		pointer m_base;
    		size_type m_n;
    		bool m_own;

    	}; // end class block_impl



	}
}

#endif
