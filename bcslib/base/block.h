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
	class BlockBase
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
		BlockBase(index_type len, pointer data)
		: m_len(len), m_data(data)
		{
		}

	protected:
		BlockBase() { }

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
				copy_elems((size_type)m_len, src, m_data);
		}

		void fill(const value_type& v)
		{
			if (m_len > 0)
				fill_elems((size_type)m_len, m_data, v);
		}

		void set_zeros()
		{
			if (m_len > 0)
				zero_elems((size_type)m_len, m_data);
		}

		void copy_to(value_type* dst) const
		{
			if (m_len > 0)
				copy_elems((size_type)m_len, m_data, dst);
		}

		void swap(BlockBase& r)
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
	inline bool is_equal(const BlockBase<T>& B1, const BlockBase<T>& B2)
	{
		return B1.nelems() == B2.nelems() && elems_equal(B1.nelems(), B1.pbase(), B2.pbase());
	}



    /********************************************
     *
     *   Block
     *
     ********************************************/

	template<typename T, typename Allocator=aligned_allocator<T> >
	class Block : public BlockBase<T>
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
		typedef BlockBase<T> base_type;

	public:
		explicit Block(const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(0);
		}

		explicit Block(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
		}

		Block(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->fill(v);
		}

		Block(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->copy_from(src);
		}

		Block(const Block& s, const allocator_type& allocator = allocator_type())
		: base_type(), m_allocator(allocator)
		{
			init(s.nelems());
			this->copy_from(s.pbase());
		}

		~Block()
		{
			release();
		}

		void swap(Block& r)
		{
			using std::swap;

			base_type::swap(r);
			swap(m_allocator, r.m_allocator);
		}

		Block& operator = (const Block& r)
		{
			if (this != &r)
			{
				Block tmp(r);
				swap(tmp);
			}
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
				pointer p = n > 0 ? m_allocator.allocate(size_type(n)) : BCS_NULL;
				release();
				this->reset(n, p);
			}
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

	}; // end class Block


    /********************************************
     *
     *   ScopedBlock
     *
     ********************************************/

	template<typename T, typename Allocator=aligned_allocator<T> >
	class ScopedBlock : public BlockBase<T>, private noncopyable
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
		typedef BlockBase<T> base_type;

	public:
		explicit ScopedBlock(index_type len, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
		}

		ScopedBlock(index_type len, const value_type& v, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->fill(v);
		}

		ScopedBlock(index_type len, const_pointer src, const allocator_type& allocator = allocator_type())
		: m_allocator(allocator)
		{
			init(len);
			this->copy_from(src);
		}

		~ScopedBlock()
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

	}; // end class ScopedBlock

}

#endif 
