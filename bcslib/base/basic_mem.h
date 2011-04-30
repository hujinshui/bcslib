/**
 * @file basic_mem.h
 *
 * Basic facilities for memory management
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_BASIC_MEM_H
#define BCSLIB_BASIC_MEM_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>


#include <stdlib.h>
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
#include <malloc.h>
#endif

#include <cstring>
#include <memory>
#include <new>
#include <limits>


namespace bcs
{

    template<typename T>
    inline void copy_elements(const T *src, T *dst, size_t n)
    {
            std::memcpy(dst, src, sizeof(T) * n);
    }


    template<typename T>
    inline bool elements_equal(const T *a, const T *b, size_t n)
    {
            return std::memcmp(a, b, sizeof(T) * n) == 0;
    }

    template<typename T>
    inline void set_zeros_to_elements(T *dst, size_t n)
    {
            std::memset(dst, 0, sizeof(T) * n);
    }

    template<typename T>
    inline void fill_elements(T *dst, size_t n, const T& v)
    {
            for (size_t i = 0; i < n; ++i) dst[i] = v;
    }

    template<typename T, typename TIndex>
    inline void copy_elements_attach_indices(const T *src, indexed_entry<T, TIndex>* dst, size_t n)
    {
    	for (size_t i = 0; i < n; ++i)
    	{
    		dst[i].set(src[i], static_cast<TIndex>(i));
    	}
    }

    template<typename T, typename TIndex>
    inline void copy_elements_detach_indices(const indexed_entry<T, TIndex>* src, T *dst, size_t n)
    {
    	for (size_t i = 0; i < n; ++i)
    	{
    		dst[i] = src[i].value;
    	}
    }


    const size_t default_memory_alignment = 16;

    template<typename T, size_t Alignment=default_memory_alignment>
    class aligned_allocator
    {
    public:
    	typedef T value_type;
    	typedef T* pointer;
    	typedef T& reference;
    	typedef const T* const_pointer;
    	typedef const T& const_reference;
    	typedef size_t size_type;
    	typedef ptrdiff_t difference_type;

    	static const size_t alignment = Alignment;

    public:
    	aligned_allocator() { }

    	aligned_allocator(const aligned_allocator& ) { }

    	template<typename U>
    	aligned_allocator(const aligned_allocator<U>& ) { }

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
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE

    		pointer p = (pointer)::_aligned_malloc(sizeof(value_type) * n, alignment);
    		if (p == 0)
    		{
    			throw std::bad_alloc();
    		}
    		return p;

#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE

    		pointer p = 0;
    		if (::posix_memalign((void**)(&p), alignment, sizeof(value_type) * n) != 0)
    		{
    			throw std::bad_alloc();
    		}
    		return p;
#endif
    	}

    	void deallocate(pointer p, size_type)
    	{
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
    		::_aligned_free(p);
#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
    		::free(p);
#endif
    	}

    	void construct (pointer p, const_reference val)
    	{
    		new ((void*)p) value_type(val);
    	}

    	void destroy (pointer p)
    	{
    		p->~value_type();
    	}

    }; // end class aligned_allocator




    /**
     *  The class to refer to a block of read/write memory that
     *  it takes care of its management by itself
     */
    template<typename T, typename TAlloc=aligned_allocator<T> >
	class block
	{
	public:
		typedef T value_type;
		typedef TAlloc allocator_type;

		typedef typename allocator_type::size_type size_type;
		typedef typename allocator_type::difference_type difference_type;

		typedef typename allocator_type::pointer pointer;
		typedef typename allocator_type::reference reference;
		typedef typename allocator_type::const_pointer const_pointer;
		typedef typename allocator_type::const_reference const_reference;

	public:
		block(size_type n)
		: m_base(n > 0 ? _alloc.allocate(n) : pointer(0)), m_n(n)
		{
		}

		~block()
		{
			if (m_base != 0)
			{
				_alloc.deallocate(this->pbase(), this->nelems());
			}
		}

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

		const_pointer pend() const
		{
			return m_base + m_n;
		}

		pointer pend()
		{
			return m_base + m_n;
		}

		const_reference operator[](difference_type i) const
		{
			return m_base[i];
		}

		reference operator[](difference_type i)
		{
			return m_base[i];
		}

	private:
		block(const block<T>&);
		block<T>& operator =(const block<T>&);

	private:
		pointer m_base;
		size_type m_n;

		allocator_type _alloc;

	}; // end class block


    // different ways of input memory

    template<typename T>
    class const_memory_proxy
    {
    public:
    	typedef T value_type;

    public:

    	const_memory_proxy()
    	: m_base(0), m_n(0)
    	{
    	}

    	const_memory_proxy(clone_t, size_t n, const value_type *src)
    	: m_pblock(new block<T>(n))
    	, m_base(m_pblock->pbase())
    	, m_n(n)
    	{
    		copy_elements(src, const_cast<T*>(m_base), n);
    	}

    	const_memory_proxy(ref_t, size_t n, const value_type *src)
    	: m_base(src), m_n(n)
    	{
    	}

    	const_memory_proxy(block<T>* pblk)
    	: m_pblock(pblk)
    	, m_base(pblk->pbase())
    	, m_n(pblk->nelems())
    	{
    	}

    	void reset(block<T>* pblk)
    	{
    		m_pblock.reset(pblk);
    		m_base = pblk->pbase();
    		m_n = pblk->nelems();
    	}

    	size_t nelems() const
    	{
    		return m_n;
    	}

    	const value_type *pbase() const
    	{
    		return m_base;
    	}

    	const value_type *pend() const
    	{
    		return m_base + m_n;
    	}

    	const value_type& operator[] (ptrdiff_t i) const
    	{
    		return m_base[i];
    	}

    public:
    	tr1::shared_ptr<block<T> > m_pblock;
    	const value_type* m_base;
    	size_t m_n;

    }; // end class const_memory_proxy

}


#endif 
