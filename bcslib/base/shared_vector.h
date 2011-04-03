/**
 * @file shared_vector.h
 *
 * A vector class (with same interface as std::vector) managed by smart pointer
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_SHARED_VECTOR_H
#define BCSLIB_SHARED_VECTOR_H

#include <bcslib/base/basic_defs.h>
#include <memory>
#include <vector>

namespace bcs
{

	template<typename T, typename Allocator=std::allocator<T> >
	class shared_vector
	{
	public:
		typedef std::vector<T, Allocator> stdvector;

		typedef typename stdvector::value_type value_type;
		typedef typename stdvector::size_type size_type;
		typedef typename stdvector::difference_type difference_type;

		typedef typename stdvector::allocator_type allocator_type;
		typedef typename stdvector::reference reference;
		typedef typename stdvector::const_reference const_reference;
		typedef typename stdvector::pointer pointer;
		typedef typename stdvector::const_pointer const_pointer;

		typedef typename stdvector::iterator iterator;
		typedef typename stdvector::const_iterator const_iterator;
		typedef typename stdvector::reverse_iterator reverse_iterator;
		typedef typename stdvector::const_reverse_iterator const_reverse_iterator;

	public:
		// constructors

		explicit shared_vector(const allocator_type& alloc = allocator_type() )
		: m_p(new stdvector(alloc)), m_vec(*m_p)
		{
		}

		explicit shared_vector(size_type n, const value_type& v = T(), const allocator_type& alloc = allocator_type())
		: m_p(new stdvector(n, v, alloc)), m_vec(*m_p)
		{
		}

		template<typename InputIter>
		shared_vector(InputIter first, InputIter last, const allocator_type& alloc = allocator_type())
		: m_p(new stdvector(first, last, alloc)), m_vec(*m_p)
		{
		}

		shared_vector(const stdvector& src)
		: m_p(new stdvector(src)), m_vec(*m_p)
		{
		}

		// size and capacities

		size_type size() const
		{
			return m_vec.size();
		}

		size_type max_size() const
		{
			return m_vec.max_size();
		}

		void resize(size_type n, value_type c = T())
		{
			return m_vec.resize(n, c);
		}

		size_type capacity() const
		{
			return m_vec.capacity();
		}

		bool empty() const
		{
			return m_vec.empty();
		}

		void reserve()
		{
			m_vec.reserve();
		}

		// element access

		reference operator[] (size_type n)
		{
			return m_vec[n];
		}

		const_reference operator[] (size_type n) const
		{
			return m_vec[n];
		}

		reference at(size_type n)
		{
			return m_vec[n];
		}

		const_reference at(size_type n) const
		{
			return m_vec[n];
		}

		reference front()
		{
			return m_vec.front();
		}

		const_reference front() const
		{
			return m_vec.front();
		}

		reference back()
		{
			return m_vec.back();
		}

		const_reference back() const
		{
			return m_vec.back();
		}


		// modifiers

		template<typename InputIter>
		void assign(InputIter first, InputIter last)
		{
			m_vec.assign(first, last);
		}

		void assign(size_type n, const T& u)
		{
			m_vec.assign(n, u);
		}

		void push_back(const T& x)
		{
			m_vec.push_back(x);
		}

		void pop_back()
		{
			m_vec.pop_back();
		}

		iterator insert(iterator pos, const T& x)
		{
			return m_vec.insert(pos, x);
		}

		void insert(iterator pos, size_type n, const T& x)
		{
			m_vec.insert(pos, n, x);
		}

		template<class InputIter>
		void insert(iterator pos, InputIter first, InputIter last)
		{
			m_vec.insert(pos, first, last);
		}

		iterator erase(iterator pos)
		{
			m_vec.erase(pos);
		}

		iterator erase(iterator first, iterator last)
		{
			m_vec.erase(first, last);
		}

		void swap(const stdvector& vec)
		{
			m_vec.swap(vec);
		}

		void clear()
		{
			m_vec.clear();
		}

		// other

		allocator_type get_allocator() const
		{
			return m_vec.get_allocator();
		}

		stdvector& get()
		{
			return m_vec;
		}

		const stdvector& get() const
		{
			return m_vec;
		}

	private:
		tr1::shared_ptr<stdvector> m_p;
		stdvector& m_vec;


	}; // end class shared_vector

}

#endif 
