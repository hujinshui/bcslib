/**
 * @file block.h
 *
 * The classes representing blocks of different types
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BLOCK_H_
#define BCSLIB_BLOCK_H_

#include <bcslib/core/mem_op.h>
#include <algorithm>

namespace bcs
{
    /********************************************
     *
     *   the block interface
     *
     ********************************************/

	template<class Derived, typename T>
	class IBlock
	{
	public:
		typedef T value_type;

		typedef size_t size_type;
		typedef ptrdiff_t difference_type;

		typedef T* pointer;
		typedef T& reference;
		typedef const T* const_pointer;
		typedef const T& const_reference;

		typedef index_t index_type;

		BCS_CRTP_REF

	public:
		size_type size() const
		{
			return (size_type)(derived().nelems());
		}

		index_type nelems() const
		{
			return derived().nelems();
		}

		const_pointer ptr_begin() const
		{
			return derived().ptr_begin();
		}

		pointer ptr_begin()
		{
			return derived().ptr_begin();
		}

		const_pointer ptr_end() const
		{
			return derived().ptr_end();
		}

		pointer pend()
		{
			return derived().ptr_end();
		}

		const_reference operator[] (index_type i) const
		{
			return derived().elem(i);
		}

		reference operator[] (index_type i)
		{
			return derived().elem(i);
		}

	}; // end class IBlock


	// Operations on blocks

	template<typename T, class Derived>
	BCS_ENSURE_INLINE
	inline void copy_from(IBlock<Derived, T>& blk, const T* src)
	{
		copy_elems(blk.nelems(), src, blk.ptr_begin());
	}

	template<typename T, class Derived>
	BCS_ENSURE_INLINE
	void copy_to(const IBlock<Derived, T>& blk, T* dst)
	{
		copy_elems(blk.nelems(), blk.ptr_begin(), dst);
	}


	template<typename T, class Derived>
	BCS_ENSURE_INLINE
	void fill(IBlock<Derived, T>& blk, const T& v)
	{
		fill_elems(blk.nelems(), blk.ptr_begin(), v);
	}

	template<typename T, class Derived>
	BCS_ENSURE_INLINE
	void zero(IBlock<Derived, T>& blk)
	{
		zero_elems(blk.nelems(), blk.ptr_begin());
	}

	template<typename T, class LDerived, class RDerived>
	BCS_ENSURE_INLINE
	inline bool is_equal(const IBlock<LDerived, T>& B1, const IBlock<RDerived, T>& B2)
	{
		return B1.nelems() == B2.nelems() && elems_equal(B1.nelems(), B1.ptr_begin(), B2.ptr_begin());
	}



    /********************************************
     *
     *   block
     *
     ********************************************/

	template<typename T, typename Allocator=aligned_allocator<T> >
	class block : public IBlock<block<T, Allocator>,  T>
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

		typedef index_t index_type;

	public:
		explicit block(const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(0)
		, m_ptr(BCS_NULL)
		{
		}

		explicit block(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_ptr(alloc(len))
		{
		}

		block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_ptr(alloc(len))
		{
			fill_elems(len, m_ptr, v);
		}

		block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_ptr(alloc(len))
		{
			copy_elems(m_len, src, m_ptr);
		}

		block(const block& s, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(s.nelems())
		, m_ptr(alloc(m_len))
		{
			copy_elems(m_len, s.ptr_begin(), m_ptr);
		}

		template<class OtherDerived>
		block(const IBlock<OtherDerived, T>& s, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(s.nelems())
		, m_ptr(alloc(m_len))
		{
			copy_elems(m_len, s.ptr_begin(), m_ptr);
		}

		~block()
		{
			dealloc(m_ptr);
			m_ptr = BCS_NULL;  // make sure that it alarms when access after destructed
		}

		void swap(block& r)
		{
			using std::swap;

			swap(m_allocator, r.m_allocator);
			swap(m_len, r.m_len);
			swap(m_ptr, r.m_ptr);
		}

		block& operator = (const block& r)
		{
			if (this != &r) assign(r);
			return *this;
		}

		template<class OtherDerived>
		block& operator = (const IBlock<OtherDerived, T>& r)
		{
			assign(r);
			return *this;
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}

		void resize(index_type n)
		{
			if (n != this->nelems())
			{
				reset_size(n);
			}
		}

	public:
		size_type size() const
		{
			return static_cast<size_type>(m_len);
		}

		index_type nelems() const
		{
			return m_len;
		}

		const_pointer ptr_begin() const
		{
			return m_ptr;
		}

		pointer ptr_begin()
		{
			return m_ptr;
		}

		const_pointer ptr_end() const
		{
			return m_ptr + m_len;
		}

		pointer pend()
		{
			return m_ptr + m_len;
		}

		const_reference operator[] (index_type i) const
		{
			return m_ptr[i];
		}

		reference operator[] (index_type i)
		{
			return m_ptr[i];
		}


	private:
		pointer alloc(index_type n)
		{
			return n > 0 ? m_allocator.allocate(size_type(n)) : BCS_NULL;
		}

		void dealloc(pointer p)
		{
			if (m_ptr) m_allocator.deallocate(p, (size_type)m_len);
		}

		void reset_size(index_type n)
		{
			pointer p = n > 0 ? m_allocator.allocate(size_type(n)) : BCS_NULL;
			dealloc(m_ptr);
			m_ptr = p;
			m_len = n;
		}

		template<class OtherDerived>
		void assign(const IBlock<OtherDerived, T>& r)
		{
			if (nelems() == r.nelems())  // no need to re-allocate memory
			{
				copy_elems(nelems(), r.ptr_begin(), this->ptr_begin());
			}
			else
			{
				block tmp(r);
				swap(tmp);
			}
		}

	private:
		allocator_type m_allocator;
		index_type m_len;
		pointer m_ptr;

	}; // end class block


	template<typename T, class Allocator>
	BCS_ENSURE_INLINE
	void swap(block<T, Allocator>& a, block<T, Allocator>& b)
	{
		a.swap(b);
	}


    /********************************************
     *
     *   ScopedBlock
     *
     ********************************************/

	template<typename T, typename Allocator=aligned_allocator<T> >
	class scoped_block : public IBlock<scoped_block<T, Allocator>, T>, private noncopyable
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

		typedef index_t index_type;

	public:
		explicit scoped_block(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_ptr(alloc(len))
		{
		}

		scoped_block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_ptr(alloc(len))
		{
			fill_elems(len, m_ptr, v);
		}

		scoped_block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_ptr(alloc(len))
		{
			copy_elems(len, src, m_ptr);
		}

		~scoped_block()
		{
			dealloc(m_ptr);
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}

	public:
		size_type size() const
		{
			return static_cast<size_type>(m_len);
		}

		index_type nelems() const
		{
			return m_len;
		}

		const_pointer ptr_begin() const
		{
			return m_ptr;
		}

		pointer ptr_begin()
		{
			return m_ptr;
		}

		const_pointer ptr_end() const
		{
			return m_ptr + m_len;
		}

		pointer pend()
		{
			return m_ptr + m_len;
		}

		const_reference operator[] (index_type i) const
		{
			return m_ptr[i];
		}

		reference operator[] (index_type i)
		{
			return m_ptr[i];
		}

	private:
		pointer alloc(index_type n)
		{
			return n > 0 ? m_allocator.allocate(size_type(n)) : BCS_NULL;
		}

		void dealloc(pointer p)
		{
			if (m_ptr) m_allocator.deallocate(p, (size_type)m_len);
		}

	private:
		allocator_type m_allocator;
		index_type m_len;
		pointer m_ptr;

	}; // end class scoped_block



    /********************************************
     *
     *   static block
     *
     ********************************************/

	template<typename T, int N>
	class static_block : public IBlock<static_block<T, N>, T>
	{
	public:
		typedef T value_type;

		typedef size_t size_type;
		typedef ptrdiff_t difference_type;

		typedef T* pointer;
		typedef T& reference;
		typedef const T* const_pointer;
		typedef const T& const_reference;

		typedef index_t index_type;

	public:
		explicit static_block()
		{
		}

		explicit static_block(const value_type& v)
		{
			mem<T, N>::fill(m_arr, v);
		}

		explicit static_block(const_pointer src)
		{
			mem<T, N>::copy(src, m_arr);
		}

		static_block(const static_block& src)
		{
			mem<T, N>::copy(src.m_arr, m_arr);
		}

		template<class OtherDerived>
		static_block(const IBlock<OtherDerived, T>& src)
		{
			mem<T, N>::copy(src.ptr_begin(), m_arr);
		}

		static_block& operator = (const static_block& src)
		{
			if (this != &src)
			{
				mem<T, N>::copy(src.m_arr, m_arr);
			}
			return *this;
		}

		template<class OtherDerived>
		static_block& operator = (const IBlock<OtherDerived, T>& src)
		{
			mem<T, N>::copy(src.ptr_begin(), m_arr);
			return *this;
		}

		void swap(static_block& r)
		{
			using std::swap;

			T tmp[N];
			mem<T, N>::copy(m_arr, tmp);
			mem<T, N>::copy(r.m_arr, m_arr);
			mem<T, N>::copy(tmp, r.m_arr);
		}

	public:
		size_type size() const
		{
			return static_cast<size_t>(N);
		}

		index_type nelems() const
		{
			return N;
		}

		const_pointer ptr_begin() const
		{
			return m_arr;
		}

		pointer ptr_begin()
		{
			return m_arr;
		}

		const_pointer ptr_end() const
		{
			return m_arr + N;
		}

		pointer pend()
		{
			return m_arr + N;
		}

		const_reference operator[] (index_type i) const
		{
			return m_arr[i];
		}

		reference operator[] (index_type i)
		{
			return m_arr[i];
		}

	private:
		T m_arr[N];

	}; // end class static_block


	template<typename T, int N>
	BCS_ENSURE_INLINE
	inline void swap(static_block<T, N>& a, static_block<T, N>& b)
	{
		a.swap(b);
	}


    /********************************************
     *
     *   specialized functions
     *
     ********************************************/

	template<typename T, int N>
	BCS_ENSURE_INLINE
	inline void copy_from(static_block<T, N>& blk, const T* src)
	{
		mem<T, N>::copy(src, blk.ptr_begin());
	}

	template<typename T, int N>
	BCS_ENSURE_INLINE
	void copy_to(const static_block<T, N>& blk, T* dst)
	{
		mem<T, N>::copy(blk.ptr_begin(), dst);
	}


	template<typename T, int N>
	BCS_ENSURE_INLINE
	void fill(static_block<T, N>& blk, const T& v)
	{
		mem<T, N>::fill(blk.ptr_begin(), v);
	}

	template<typename T, int N>
	BCS_ENSURE_INLINE
	void zero(static_block<T, N>& blk)
	{
		mem<T, N>::zero(blk.ptr_begin());
	}

	template<typename T, int N>
	BCS_ENSURE_INLINE
	inline bool is_equal(const static_block<T, N>& B1, const static_block<T, N>& B2)
	{
		return mem<T, N>::equal(B1.ptr_begin(), B2.ptr_begin());
	}

	template<typename T, int N, class RDerived>
	BCS_ENSURE_INLINE
	inline bool is_equal(const static_block<T, N>& B1, const IBlock<RDerived, T>& B2)
	{
		return N == B2.nelems() && mem<T, N>::equal(B1.ptr_begin(), B2.ptr_begin());
	}

	template<typename T, int N, class LDerived>
	BCS_ENSURE_INLINE
	inline bool is_equal(const IBlock<LDerived, T>& B1, const static_block<T, N>& B2)
	{
		return N == B1.nelems() && mem<T, N>::equal(B1.ptr_begin(), B2.ptr_begin());
	}



}

#endif 
