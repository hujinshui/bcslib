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

namespace bcs
{
    /********************************************
     *
     *   block base
     *
     ********************************************/

	template<typename T>
	class block_base
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
		block_base(index_type len, pointer data)
		: m_len(len), m_data(data)
		{
		}

	protected:
		block_base() { }

		void reset(index_type len, pointer data)
		{
			m_len = len;
			m_data = data;
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

		const_reference operator[] (index_type i) const
		{
			return m_data[i];
		}

		reference operator[] (index_type i)
		{
			return m_data[i];
		}

	public:
		void copy_from(const value_type* src)
		{
			if (m_len > 0)
				mem<T>::copy((size_type)m_len, src, m_data);
		}

		void fill(const value_type& v)
		{
			if (m_len > 0)
				mem<T>::fill((size_type)m_len, m_data, v);
		}

		void set_zeros()
		{
			if (m_len > 0)
				mem<T>::zero((size_type)m_len, m_data);
		}

		void copy_to(value_type* dst) const
		{
			if (m_len > 0)
				mem<T>::copy((size_type)m_len, m_data, dst);
		}

		void swap(block_base& r)
		{
			using std::swap;
			swap(m_len, r.m_len);
			swap(m_data, r.m_data);
		}

	protected:
		index_type m_len;
		pointer m_data;

	}; // end class block_base


	template<typename T>
	inline bool is_equal(const block_base<T>& B1, const block_base<T>& B2)
	{
		return B1.nelems() == B2.nelems() && mem<T>::equal(B1.nelems(), B1.pbase(), B2.pbase());
	}



    /********************************************
     *
     *   block
     *
     ********************************************/

	template<typename T, typename Allocator=aligned_allocator<T> >
	class block : public block_base<T>
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

	private:
		typedef block_base<T> base_type;

	public:
		explicit block(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
		}

		block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->fill(v);
		}

		block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->copy_from(src);
		}

		block(const block& s, const allocator_type& allocator = allocator_type())
		: base_type(), m_allocator(allocator)
		{
			init(s.nelems());
			this->copy_from(s.pbase());
		}

		~block()
		{
			release();
		}

		void swap(block& r)
		{
			using std::swap;

			base_type::swap(r);
			swap(m_allocator, r.m_allocator);
		}

		block& operator = (const block& r)
		{
			if (this != &r)
			{
				block tmp(r);
				swap(tmp);
			}
			return *this;
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}


	private:
		void init(index_type n)
		{
			pointer p = n > 0 ? m_allocator.allocate(size_type(n)) : BCS_NULL;
			this->reset(n, p);
		}

		void release()
		{
			if (this->m_data)
			{
				m_allocator.deallocate(this->m_data, this->size());
			}
		}

	private:
		allocator_type m_allocator;

	}; // end class block


    /********************************************
     *
     *   scoped_block
     *
     ********************************************/

	template<typename T, typename Allocator=aligned_allocator<T> >
	class scoped_block : public block_base<T>, private noncopyable
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

	private:
		typedef block_base<T> base_type;

	public:
		explicit scoped_block(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
		}

		scoped_block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->fill(v);
		}

		scoped_block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->copy_from(src);
		}

		~scoped_block()
		{
			release();
		}

		const allocator_type& get_allocator() const
		{
			return m_allocator;
		}


	private:
		void init(index_type n)
		{
			pointer p = n > 0 ? m_allocator.allocate(size_type(n)) : BCS_NULL;
			this->reset(n, p);
		}

		void release()
		{
			if (this->m_data)
			{
				m_allocator.deallocate(this->m_data, this->size());
			}
		}

	private:
		allocator_type m_allocator;

	}; // end class scoped_block

}

#endif 
