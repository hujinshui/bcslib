/**
 * @file block.h
 *
 * The block class and relevant classes
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BLOCK_H_
#define BCSLIB_BLOCK_H_

#include <bcslib/base/mem_op.h>
#include <bcslib/base/smart_ptr.h>

namespace bcs
{

    /********************************************
     *
     *   block
     *
     ********************************************/

	template<typename T, typename Allocator>
	class block
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
		explicit block(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_data(safe_alloc(len))
		{
		}

		block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_data(safe_alloc(len))
		{
			if (len > 0) mem<T>::fill(m_data, v);
		}

		block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_data(safe_alloc(len))
		{
			if (len > 0) mem<T>::copy(src, m_data, len);
		}

		block(const block& s, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(s.m_len)
		, m_data(safe_alloc(m_len))
		{
			if (m_len > 0) mem<T>::copy(s.pbase(), m_data, m_len);
		}

		~block()
		{
			safe_dealloc(m_data, m_len);
		}

		void swap(block& r)
		{
			using std::swap;

			swap(m_allocator, r.m_allocator);
			swap(m_len, r.m_len);
			swap(m_data, r.m_data);
		}

		block& operator = (const block& r)
		{
			if (this != &r)
			{
				swap(block(r));
			}
			return *this;
		}

	public:
		size_type size() const
		{
			return (size_type)m_len;
		}

		index_type nelems() const
		{
			return m_len;
		}

		const_pointer pbase() const
		{
			return m_data;
		}

		pointer pbase()
		{
			return m_data;
		}

		const_pointer pend() const
		{
			return m_data + m_len;
		}

		pointer pend()
		{
			return m_data + m_len;
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}

		const_reference operator[] (index_type i) const
		{
			return m_data[i];
		}

		reference operator[] (index_type i)
		{
			return m_data[i];
		}

	private:
		pointer safe_alloc(index_type n)
		{
			return n > 0 ? m_allocator.allocate((size_type)n) : BCS_NULL;
		}

		void safe_dealloc(pointer p, index_type n)
		{
			m_allocator.deallocate(p, (size_type)n);
		}

	private:
		allocator_type m_allocator;
		index_type m_len;
		pointer m_data;

	}; // end class block


    /********************************************
     *
     *   scoped_block
     *
     ********************************************/

	template<typename T, typename Allocator>
	class scoped_block : private noncopyable
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
		, m_data(safe_alloc(len))
		{
		}

		scoped_block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_data(safe_alloc(len))
		{
			if (len > 0) mem<T>::fill(m_data, v);
		}

		scoped_block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		, m_len(len)
		, m_data(safe_alloc(len))
		{
			if (len > 0) mem<T>::copy(src, m_data, len);
		}

		~scoped_block()
		{
			safe_dealloc(m_data, m_len);
		}

	public:
		size_type size() const
		{
			return (size_type)m_len;
		}

		index_type nelems() const
		{
			return m_len;
		}

		const_pointer pbase() const
		{
			return m_data;
		}

		pointer pbase()
		{
			return m_data;
		}

		const_pointer pend() const
		{
			return m_data + m_len;
		}

		pointer pend()
		{
			return m_data + m_len;
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}

		const_reference operator[] (index_type i) const
		{
			return m_data[i];
		}

		reference operator[] (index_type i)
		{
			return m_data[i];
		}

	private:
		pointer safe_alloc(index_type n)
		{
			return n > 0 ? m_allocator.allocate((size_type)n) : BCS_NULL;
		}

		void safe_dealloc(pointer p, index_type n)
		{
			m_allocator.deallocate(p, (size_type)n);
		}

	private:
		allocator_type m_allocator;
		index_type m_len;
		pointer m_data;

	}; // end class scoped_block


    /********************************************
     *
     *   shared_block
     *
     ********************************************/

	template<typename T, typename Allocator>
	class shared_block
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
		typedef block<T, Allocator> block_type;

	public:
		explicit shared_block(index_type len, const allocator_type& allocator = allocator_type())
		: m_ptr(new block_type(len, allocator))
		, m_len(len)
		, m_data(m_ptr->pbase())
		{
		}

		shared_block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_ptr(new block_type(len, v, allocator))
		, m_len(len)
		, m_data(m_ptr->pbase())
		{
		}

		shared_block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_ptr(new block_type(len, src, allocator))
		, m_len(len)
		, m_data(m_ptr->pbase())
		{
		}

		shared_block(const block_type& s, const allocator_type& allocator = allocator_type())
		: m_ptr(new block_type(s, allocator))
		, m_len(s.nelems())
		, m_data(m_ptr->pbase())
		{
		}

		shared_block(shared_block& s)
		: m_ptr(s.m_ptr)
		, m_len(s.nelems())
		, m_data(s.pbase())
		{
		}

		void swap(shared_block& r)
		{
			using std::swap;
			swap(m_ptr, r.m_ptr);
			swap(m_len, r.m_len);
			swap(m_data, r.m_data);
		}

		shared_block& operator = (shared_block& r)
		{
			if (m_ptr != r.m_ptr)
			{
				swap(shared_block(r));
			}
			return *this;
		}

		shared_block clone() const
		{
			return shared_block(*m_ptr);
		}

		void make_unique()
		{
			if (!m_ptr.unique())
			{
				swap(shared_block(*m_ptr));
			}
		}

	public:
		bool unique() const
		{
			return m_ptr.unique();
		}

		size_type size() const
		{
			return (size_type)m_len;
		}

		index_type nelems() const
		{
			return m_len;
		}

		const_pointer pbase() const
		{
			return m_data;
		}

		pointer pbase()
		{
			return m_data;
		}

		const_pointer pend() const
		{
			return m_data + m_len;
		}

		pointer pend()
		{
			return m_data + m_len;
		}

		const allocator_type& get_allocator() const
		{
			return m_ptr->get_allocator();
		}

		const_reference operator[] (index_type i) const
		{
			return m_data[i];
		}

		reference operator[] (index_type i)
		{
			return m_data[i];
		}

	private:
		shared_ptr<block_type> m_ptr;
		index_type m_len;
		pointer m_data;

	}; // end class shared_block


}

#endif 
