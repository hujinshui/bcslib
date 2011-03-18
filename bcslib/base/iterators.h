/**
 * @file iterators.h
 *
 * The facilities to support the implementation of iterator classes
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ITERATORS_H
#define BCSLIB_ITERATORS_H

#include <iterator>


namespace bcs
{

	/**
	 * The Concept of forward_iterator_implementer
	 * -----------
	 *
	 * typedef of value_type;
	 * typedef of reference;
	 * typedef of pointer;
	 *
	 * - void move_next();
	 * - T* ptr() const;
	 * - T& ref() const;
	 * - operator ==
	 *
	 * The Concept of bidirectional_iterator_implementer
	 * refines forward_iterator_implementer
	 * -----------
	 *
	 * - void move_prev();
	 *
	 * The Concept of random_access_iterator_implementer
	 * refines bidirectional_iterator_implementer
	 * -----------
	 *
	 * - void move_forward(int n);
	 * - void move_backward(int n);
	 *
	 * - operator <
	 * - operator >
	 *
	 * - T& at(int n) const;
	 *
	 */


	template<typename T, bool IsConst> struct pointer_and_reference;

	template<typename T>
	struct pointer_and_reference<T, true>
	{
		typedef const T* pointer;
		typedef const T& reference;
	};

	template<typename T>
	struct pointer_and_reference<T, false>
	{
		typedef T* pointer;
		typedef T& reference;
	};


	template<class Impl>
	class forward_iterator_wrapper
	{
	public:
		typedef forward_iterator_wrapper<Impl> self_type;

		typedef std::forward_iterator_tag iterator_category;
		typedef typename Impl::value_type value_type;
		typedef typename Impl::pointer pointer;
		typedef typename Impl::reference reference;
		typedef ptrdiff_t difference_type;

	public:
		forward_iterator_wrapper()
		{
		}

		forward_iterator_wrapper(const Impl& impl) : m_impl(impl)
		{
		}

		bool operator == (const self_type& rhs) const
		{
			return m_impl == rhs.m_impl;
		}

		bool operator != (const self_type& rhs) const
		{
			return !(m_impl != rhs.m_impl);
		}

		reference operator * () const
		{
			return m_impl.ref();
		}

		pointer operator -> () const
		{
			return m_impl.ptr();
		}

		self_type& operator ++ ()
		{
			m_impl.move_next();
			return *this;
		}

		self_type operator ++ (int)
		{
			Impl cimpl = m_impl;
			m_impl.move_next();
			return cimpl;
		}

	protected:
		Impl m_impl;

	}; // end class forward_iterator_wrapper


	template<class Impl>
	class bidirectional_iterator_wrapper : public forward_iterator_wrapper<Impl>
	{
	public:
		typedef bidirectional_iterator_wrapper<Impl> self_type;

		typedef std::bidirectional_iterator_tag iterator_category;
		typedef typename Impl::value_type value_type;
		typedef typename Impl::pointer pointer;
		typedef typename Impl::reference reference;
		typedef ptrdiff_t difference_type;

	public:
		bidirectional_iterator_wrapper()
		{
		}

		bidirectional_iterator_wrapper(const Impl& impl) : forward_iterator_wrapper<Impl>(impl)
		{
		}

		self_type& operator -- ()
		{
			this->m_impl.move_prev();
			return *this;
		}

		self_type operator -- (int)
		{
			Impl cimpl = this->m_impl;
			this->m_impl.move_prev();
			return cimpl;
		}

	}; // end class bidirectional_iterator_wrapper


	template<class Impl>
	class random_access_iterator_wrapper : public bidirectional_iterator_wrapper<Impl>
	{
	public:
		typedef random_access_iterator_wrapper<Impl> self_type;

		typedef std::random_access_iterator_tag iterator_category;
		typedef typename Impl::value_type value_type;
		typedef typename Impl::pointer pointer;
		typedef typename Impl::reference reference;
		typedef ptrdiff_t difference_type;

	public:
		random_access_iterator_wrapper()
		{
		}

		random_access_iterator_wrapper(const Impl& impl) : bidirectional_iterator_wrapper<Impl>(impl)
		{
		}

		bool operator < (const self_type& rhs) const
		{
			return this->m_impl < rhs.m_impl;
		}

		bool operator <= (const self_type& rhs) const
		{
			return !(this->m_impl > rhs.m_impl);
		}

		bool operator > (const self_type& rhs) const
		{
			return this->impl > rhs.m_impl;
		}

		bool operator >= (const self_type& rhs) const
		{
			return !(this->m_impl < rhs.m_impl);
		}

		self_type operator + (difference_type n) const
		{
			Impl impl(this->m_impl);
			impl.move_forward(n);
			return impl;
		}

		self_type operator - (difference_type n) const
		{
			Impl impl(this->m_impl);
			impl.move_backward(n);
			return impl;
		}

		self_type& operator += (difference_type n)
		{
			this->m_impl.move_forward(n);
			return *this;
		}

		self_type& operator -= (difference_type n)
		{
			this->m_impl.move_backward(n);
			return *this;
		}

		reference operator[] (difference_type n) const
		{
			return this->m_impl.at(n);
		}

	}; // end class random_access_iterator_wrapper



}

#endif 
