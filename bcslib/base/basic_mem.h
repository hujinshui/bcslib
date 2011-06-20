/**
 * @file basic_mem.h
 *
 * Basic facilities for memory management
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif


#ifndef BCSLIB_BASIC_MEM_H
#define BCSLIB_BASIC_MEM_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>

#include <algorithm> 	// for swap
#include <cstring>  	// for low-level memory manipulation functions
#include <new>   		// for std::bad_alloc
#include <limits>  		// for computing max() in allocator types
#include <memory> 		// for std::allocator

// for platform-dependent aligned allocation
#include <stdlib.h>
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
#include <malloc.h>
#endif


#ifndef BCS_DEFAULT_ALIGNMENT
#define BCS_DEFAULT_ALIGNMENT 32
#endif


namespace bcs
{

	/********************************************
	 *
	 *	aligned array on stack
	 *
	 ********************************************/

	template<typename T, size_t N>
	struct aligned_array
	{
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
		__declspec(align(BCS_DEFAULT_ALIGNMENT)) T data[N];
#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
		T data[N] __attribute__((aligned(BCS_DEFAULT_ALIGNMENT)));
#endif

		const T& operator[] (size_t i) const
		{
			return data[i];
		}

		T& operator[] (size_t i)
		{
			return data[i];
		}

		size_t size() const
		{
			return N;
		}

	}; // end struct aligned_array


	/********************************************
	 *
	 *	Basic memory manipulation
	 *
	 ********************************************/

	namespace _detail
	{
		template<typename T, bool IsTriviallyCopyable> struct __element_copy_helper;

		template<typename T>
		struct __element_copy_helper<T, true>
		{
			static void copy(const T *src, T *dst, size_t n)
			{
				std::memcpy(dst, src, sizeof(T) * n);
			}

			static void copy_construct(const T *src, T *dst, size_t n)
			{
				std::memcpy(dst, src, sizeof(T) * n);
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
		struct __element_copy_helper<T, false>
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


		template<typename T, bool IsTriviallyDestructible> struct __element_destruct_helper;

		template<typename T>
		struct __element_destruct_helper<T, true>
		{
			static void destruct(T *, size_t) { }
		};

		template<typename T>
		struct __element_destruct_helper<T, false>
		{
			static void destruct(T *dst, size_t n)
			{
				for (size_t i = 0; i < n; ++i)
				{
					(dst+i)->~T();
				}
			}
		};


		template<typename T, bool IsBitwiseComparable> struct __element_compare_helper;

		template<typename T>
		struct __element_compare_helper<T, true>
		{
			static bool all_equal(const T *a, const T *b, size_t n)
			{
				return std::memcmp(a, b, sizeof(T) * n) == 0;
			}
		};

		template<typename T>
		struct __element_compare_helper<T, false>
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


    template<typename T>
    inline void copy_construct_elements(const T& v, T *dst, size_t n)
    {
    	_detail::__element_copy_helper<T, std::is_pod<T>::value>::copy_construct(v, dst, n);
    }

    template<typename T>
    inline void copy_construct_elements(const T *src, T *dst, size_t n)
    {
    	_detail::__element_copy_helper<T, std::is_pod<T>::value>::copy_construct(src, dst, n);
    }

    template<typename T>
    inline void copy_elements(const T *src, T *dst, size_t n)
    {
    	_detail::__element_copy_helper<T, std::is_pod<T>::value>::copy(src, dst, n);
    }

    template<typename T>
    inline bool elements_equal(const T *a, const T *b, size_t n)
    {
    	return _detail::__element_compare_helper<T, std::is_pod<T>::value>::all_equal(a, b, n);
    }

    template<typename T>
    inline bool elements_equal(const T& v, const T *a, size_t n)
    {
    	for (size_t i = 0; i < n; ++i)
    	{
    		if (a[i] != v) return false;
    	}
    	return true;
    }

    template<typename T>
    inline void set_zeros_to_elements(T *dst, size_t n)
    {
    	BCS_STATIC_ASSERT_V(std::is_pod<T>);
    	std::memset(dst, 0, sizeof(T) * n);
    }

    template<typename T>
    inline void fill_elements(T *dst, size_t n, const T& v)
    {
    	for (size_t i = 0; i < n; ++i) dst[i] = v;
    }


    template<typename T>
    inline void destruct_elements(T *dst, size_t n)
    {
    	_detail::__element_destruct_helper<T, std::is_pod<T>::value>::destruct(dst, n);
    }



	/********************************************
	 *
	 *	aligned allocation
	 *
	 ********************************************/

    template<typename T>
    T* aligned_allocate(size_t nelements, size_t alignment)
    {
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
    	T* p = static_cast<T*>(::_aligned_malloc(sizeof(T) * nelements, alignment));
    	if (p == 0)
		{
			throw std::bad_alloc();
		}
		return p;

#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
		char* p = 0;
		if (::posix_memalign((void**)(&p), alignment, sizeof(T) * nelements) != 0)
		{
			throw std::bad_alloc();
		}
		return static_cast<T*>((void*)p);

#endif
    }

    template<typename T>
    void aligned_deallocate(T *p)
    {
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
    	::_aligned_free(p);
#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
    	::free(p);
#endif
    }


    template<typename T>
    class aligned_allocator
    {
    public:
    	static const size_t default_memory_alignment = BCS_DEFAULT_ALIGNMENT;

    	typedef T value_type;
    	typedef T* pointer;
    	typedef T& reference;
    	typedef const T* const_pointer;
    	typedef const T& const_reference;
    	typedef size_t size_type;
    	typedef ptrdiff_t difference_type;

    	template<typename TOther>
    	struct rebind
    	{
    		typedef aligned_allocator<TOther> other;
    	};

    public:
    	aligned_allocator()
    	: m_alignment(default_memory_alignment)
    	{
    	}

    	explicit aligned_allocator(size_t align)
    	: m_alignment(align)
    	{
    	}

    	aligned_allocator(const aligned_allocator& r)
    	: m_alignment(r.alignment())
    	{
    	}

    	template<typename U>
    	aligned_allocator(const aligned_allocator<U>& r)
    	: m_alignment(r.alignment())
    	{
    	}

    	size_t alignment() const
    	{
    		return m_alignment;
    	}


    	pointer address( reference x ) const
    	{
    		return &x;
    	}

    	const_pointer address( const_reference x ) const
    	{
    		return &x;
    	}

    	size_type max_size() const
    	{
    		return std::numeric_limits<size_type>::max() / sizeof(value_type);
    	}

    	pointer allocate(size_type n, const void* hint=0)
    	{
    		return aligned_allocate<value_type>(n, m_alignment);
    	}

    	void deallocate(pointer p, size_type)
    	{
    		aligned_deallocate(p);
    	}

    	void construct (pointer p, const_reference val)
    	{
    		new (p) value_type(val);
    	}

    	void destroy (pointer p)
    	{
    		p->~value_type();
    	}

    private:
    	size_t m_alignment;

    }; // end class aligned_allocator


    template<class Allocator>
    inline typename Allocator::pointer safe_allocate(Allocator& allocator, size_t n)
    {
    	return n > 0 ? allocator.allocate(n) : static_cast<typename Allocator::pointer>(BCS_NULL);
    }

    template<class Allocator>
    inline void safe_deallocate(Allocator& allocator, typename Allocator::pointer p, size_t n)
    {
    	if (p)
    	{
    		allocator.deallocate(p, n);
    	}
    }


	/********************************************
	 *
	 *	scoped_buffer
	 *
	 ********************************************/

    /**
     * A very simple & efficient temporary buffer, which
     * can only be used within a local scope.
     *
     * No element construction when allocated.
     */
    template<typename T, class Allocator=aligned_allocator<T> >
    class scoped_buffer : private noncopyable
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
    	explicit scoped_buffer(size_type n)
    	: m_allocator()
    	, m_pbase(safe_allocate(m_allocator, n))
    	, m_n(n)
    	{
    	}

    	~scoped_buffer()
    	{
    		safe_deallocate(m_allocator, m_pbase, m_n);
    	}

		size_type nelems() const
		{
			return m_n;
		}

		const_pointer pbase() const
		{
			return m_pbase;
		}

		pointer pbase()
		{
			return m_pbase;
		}

		const_reference operator[] (size_type i) const
		{
			return m_pbase[i];
		}

		reference operator[] (size_type i)
		{
			return m_pbase[i];
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}

    private:
    	Allocator m_allocator;
    	T *m_pbase;
    	size_t m_n;

    }; // end class scoped_buffer


	/********************************************
	 *
	 *	const_block and block
	 *
	 ********************************************/

    template<typename T, class Allocator> class const_block;
    template<typename T, class Allocator> class block;


    // different ways of input memory

    template<typename T>
    class cref_blk_t
    {
    public:
    	cref_blk_t(const T* base, size_t n)
    	: m_base(base), m_n(n)
    	{
    	}

		size_t nelems() const
		{
			return m_n;
		}

		const T* pbase() const
		{
			return m_base;
		}

		const T* pend() const
		{
			return m_base + m_n;
		}

    private:
    	const T *m_base;
    	size_t m_n;
    };


    template<typename T>
    class ref_blk_t
    {
    public:
    	ref_blk_t(T* base, size_t n)
    	: m_base(base), m_n(n)
    	{
    	}

		size_t nelems() const
		{
			return m_n;
		}

		T* pbase() const
		{
			return m_base;
		}

		T* pend() const
		{
			return m_base + m_n;
		}

    private:
    	T *m_base;
    	size_t m_n;
    };


    template<typename T>
    class copy_blk_t
    {
    public:
    	copy_blk_t(const T* base, size_t n)
    	: m_base(base), m_n(n)
    	{
    	}

		size_t nelems() const
		{
			return m_n;
		}

		const T* pbase() const
		{
			return m_base;
		}

		const T* pend() const
		{
			return m_base + m_n;
		}

    private:
    	const T *m_base;
    	size_t m_n;
    };


    namespace _detail
    {
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
    		}

    		block_impl(size_type n, const allocator_type& allocator)
    		: m_allocator(allocator)
    		, m_base(safe_allocate(m_allocator, n)), m_n(n), m_own(true)
    		{
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


    template<typename T, typename Allocator=aligned_allocator<T> >
	class const_block
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
		explicit const_block(const cref_blk_t<value_type>& src)
		: m_impl(const_cast<pointer>(src.pbase()), src.nelems())
		{
		}

		const_block(const size_type& n, const_reference v)
		: m_impl(n, v)
		{
		}

		const_block(const size_type& n, const_reference v, const allocator_type& allocator)
		: m_impl(n, v, allocator)
		{
		}

		explicit const_block(const copy_blk_t<value_type>& src)
		: m_impl(src.nelems(), src.pbase())
		{
		}

		const_block(const copy_blk_t<value_type>& src, const allocator_type& allocator)
		: m_impl(src.nelems(), src.pbase(), allocator)
		{
		}


		const_block(const const_block& r)
		: m_impl(r.m_impl)
		{
		}

		const_block(const_block&& r)
		: m_impl(std::move(r.m_impl))
		{
		}

		const_block& operator = (const const_block& r)
		{
			m_impl = r.m_impl;
			return *this;
		}

		const_block& operator = (const_block&& r)
		{
			m_impl = std::move(r.m_impl);
			return *this;
		}

		void swap(const_block& r)
		{
			m_impl.swap(r.m_impl);
		}

		inline const_block(block<T, Allocator>&& r);
		inline const_block& operator = (block<T, Allocator>&& r);

	public:
		size_type nelems() const
		{
			return m_impl.nelems();
		}

		bool own_memory() const
		{
			return m_impl.own_memory();
		}

		const allocator_type& get_allocator() const
		{
			return m_impl.get_allocator();
		}

		const_pointer pbase() const
		{
			return m_impl.pbase();
		}

		const_pointer pend() const
		{
			return m_impl.pbase() + m_impl.nelems();
		}

		const_reference operator[](size_type i) const
		{
			return m_impl.get(i);
		}

	private:
		_detail::block_impl<T, Allocator> m_impl;

	}; // end class const_block


    template<typename T, typename Allocator=aligned_allocator<T> >
	class block
	{
    	friend class const_block<T, Allocator>;

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
		explicit block(const ref_blk_t<value_type>& src)
		: m_impl(src.pbase(), src.nelems())
		{
		}

		explicit block(const size_type& n)
		: m_impl(n)
		{
		}

		block(const size_type& n, const allocator_type& allocator)
		: m_impl(n, allocator)
		{
		}

		block(const size_type& n, const_reference v)
		: m_impl(n, v)
		{
		}

		block(const size_type& n, const_reference v, const allocator_type& allocator)
		: m_impl(n, v, allocator)
		{
		}

		explicit block(const copy_blk_t<value_type>& src)
		: m_impl(src.nelems(), src.pbase())
		{
		}

		block(const copy_blk_t<value_type>& src, const allocator_type& allocator)
		: m_impl(src.nelems(), src.pbase(), allocator)
		{
		}

		block(const block& r)
		: m_impl(r.m_impl)
		{
		}

		block(block&& r)
		: m_impl(std::move(r.m_impl))
		{
		}

		block& operator = (const block& r)
		{
			m_impl = r.m_impl;
			return *this;
		}

		block& operator = (block&& r)
		{
			m_impl = std::move(r.m_impl);
			return *this;
		}

		void swap(block& r)
		{
			m_impl.swap(r.m_impl);
		}

	public:
		size_type nelems() const
		{
			return m_impl.nelems();
		}

		bool own_memory() const
		{
			return m_impl.own_memory();
		}

		const allocator_type& get_allocator() const
		{
			return m_impl.get_allocator();
		}

		const_pointer pbase() const
		{
			return m_impl.pbase();
		}

		pointer pbase()
		{
			return m_impl.pbase();
		}

		const_pointer pend() const
		{
			return m_impl.pbase() + m_impl.nelems();
		}

		pointer pend()
		{
			return m_impl.pbase() + m_impl.nelems();
		}

		const_reference operator[](size_type i) const
		{
			return m_impl.get(i);
		}

		reference operator[](size_type i)
		{
			return m_impl.get(i);
		}

	private:
		_detail::block_impl<T, Allocator> m_impl;

	}; // end class block


    template<typename T, class Allocator>
	inline const_block<T, Allocator>::const_block(block<T, Allocator>&& r)
	: m_impl(std::move(r.m_impl))
	{
	}

    template<typename T, class Allocator>
	inline const_block<T, Allocator>& const_block<T, Allocator>::operator = (block<T, Allocator>&& r)
    {
    	m_impl = std::move(r.m_impl);
    	return *this;
	}


	// specialize swap for const_block and block

	template<typename T, class Allocator>
	inline void swap(bcs::const_block<T, Allocator>& lhs, bcs::const_block<T, Allocator>& rhs)
	{
		lhs.swap(rhs);
	}

	template<typename T, class Allocator>
	inline void swap(bcs::block<T, Allocator>& lhs, bcs::block<T, Allocator>& rhs)
	{
		lhs.swap(rhs);
	}


	// convenient routines

    template<typename T>
    inline cref_blk_t<T> cref_blk(const T *p, size_t n)
    {
    	return cref_blk_t<T>(p, n);
    }

    template<typename T, class Allocator>
    inline cref_blk_t<T> cref_blk(const const_block<T, Allocator>& blk)
    {
    	return cref_blk_t<T>(blk.pbase(), blk.nelems());
    }

    template<typename T, class Allocator>
    inline cref_blk_t<T> cref_blk(const block<T, Allocator>& blk)
    {
    	return cref_blk_t<T>(blk.pbase(), blk.nelems());
    }

    template<typename T>
    inline ref_blk_t<T> ref_blk(T *p, size_t n)
    {
    	return ref_blk_t<T>(p, n);
    }

    template<typename T, class Allocator>
    inline ref_blk_t<T> ref_blk(block<T, Allocator>& blk)
    {
    	return ref_blk_t<T>(blk.pbase(), blk.nelems());
    }


    template<typename T>
    inline copy_blk_t<T> copy_blk(const T *p, size_t n)
    {
    	return copy_blk_t<T>(p, n);
    }

    template<typename T, class Allocator>
    inline copy_blk_t<T> copy_blk(const const_block<T, Allocator>& blk)
    {
    	return copy_blk_t<T>(blk.pbase(), blk.nelems());
    }

    template<typename T, class Allocator>
    inline copy_blk_t<T> copy_blk(const block<T, Allocator>& blk)
    {
    	return copy_blk_t<T>(blk.pbase(), blk.nelems());
    }


    // storage_base (can serve as a private base class for classes that need to keep storage)

    struct do_share { };

    template<typename T, class Allocator=aligned_allocator<T> >
    class sharable_storage_base
    {
    public:
    	typedef block<T, Allocator> block_type;

    	sharable_storage_base(size_t n)
    	: m_sp_block(new block_type(n))
    	{
    	}

    	sharable_storage_base(size_t n, const T& v)
    	: m_sp_block(new block_type(n, v))
    	{
    	}

    	sharable_storage_base(size_t n, const T *src)
    	: m_sp_block(new block_type(copy_blk(src, n)))
    	{
    	}

    	sharable_storage_base(const sharable_storage_base& r)
    	: m_sp_block(new block_type(*(r.m_sp_block)))
    	{
    	}

    	sharable_storage_base(sharable_storage_base&& r)
    	: m_sp_block(std::move(r.m_sp_block))
    	{
    	}

    	void operator = (const sharable_storage_base& r)
    	{
    		if (this != &r)
    		{
    			m_sp_block.reset(new block_type(*(r.m_sp_block)));
    		}
    	}

    	void operator = (sharable_storage_base&& r)
    	{
    		m_sp_block = std::move(r.m_sp_block);
    	}

    	void swap(sharable_storage_base& r)
    	{
    		m_sp_block.swap(r.m_sp_block);
    	}

    	T* pointer_to_base()
    	{
    		return m_sp_block->pbase();
    	}

    	const T* pointer_to_base() const
    	{
    		return m_sp_block->pbase();
    	}

    public:
    	sharable_storage_base(const sharable_storage_base& r, do_share)
    	: m_sp_block(r.m_sp_block)
    	{
    	}

    	void share_from(const sharable_storage_base& r)
    	{
    		m_sp_block = r.m_sp_block;
    	}

    private:
    	std::shared_ptr<block_type> m_sp_block;

    }; // end class sharable_storage_base

}



#endif 
