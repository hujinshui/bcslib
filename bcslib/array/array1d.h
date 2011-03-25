/**
 * @file array1d.h
 *
 * one-dimensional array
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY1D_H
#define BCSLIB_ARRAY1D_H

#include <bcslib/array/array_base.h>
#include <bcslib/array/array_index.h>

#include <bcslib/base/iterators.h>
#include <algorithm>

namespace bcs
{

	// forward declaration

	template<typename T, class TIndexer=id_ind> class const_aview1d;
	template<typename T, class TIndexer=id_ind> class aview1d;
	template<typename T, class Alloc=std::allocator<T> > class array1d;

	// iterators

	template<typename T, class TIndexer, bool IsConst>
	class aview1d_iter_implementer
	{
	public:
		typedef aview1d_iter_implementer<T, TIndexer, IsConst> self_type;
		typedef T value_type;
		typedef typename pointer_and_reference<T, IsConst>::pointer pointer;
		typedef typename pointer_and_reference<T, IsConst>::reference reference;

	public:
		aview1d_iter_implementer() : m_base(0), m_pindexer(0), m_i(0) { }

		aview1d_iter_implementer(pointer base, const TIndexer& indexer, index_t i)
		: m_base(base), m_pindexer(&indexer), m_i(i) { }

		pointer ptr() const { return m_base + offset(m_i); }
		reference ref() const { return m_base[offset(m_i)]; }
		reference at(index_t n) const { return m_base[offset(m_i + n)]; }

		void move_next() { ++ m_i; }
		void move_prev() { -- m_i; }
		void move_forward(index_t n) { m_i += n; }
		void move_backward(index_t n) { m_i -= n; }

		bool operator == (const self_type& rhs) const { return m_i == rhs.m_i; }
		bool operator < (const self_type& rhs) const { return m_i < rhs.m_i; }
		bool operator > (const self_type& rhs) const { return m_i > rhs.m_i; }

	private:
		index_t offset(index_t i) const { return m_pindexer->operator[](m_i); }

		pointer m_base;
		const TIndexer *m_pindexer;
		index_t m_i;

	}; // end class aview1d_iter_implementer


	template<typename T, bool IsConst>
	class aview1d_step_iter_implementer
	{
	public:
		typedef aview1d_step_iter_implementer<T, IsConst> self_type;
		typedef T value_type;
		typedef typename pointer_and_reference<T, IsConst>::pointer pointer;
		typedef typename pointer_and_reference<T, IsConst>::reference reference;

	public:
		aview1d_step_iter_implementer() : m_p(0), m_step(0) { }

		aview1d_step_iter_implementer(pointer p, index_t step) : m_p(p), m_step(step) { }

		pointer ptr() const { return m_p; }
		reference ref() const { return *m_p; }
		reference at(index_t n) const { return m_p[n * m_step]; }

		void move_next() { m_p += m_step; }
		void move_prev() { m_p -= m_step; }
		void move_forward(index_t n) { m_p += n * m_step; }
		void move_backward(index_t n) { m_p -= n * m_step; }

		bool operator == (const self_type& rhs) const { return m_p == rhs.m_p; }
		bool operator < (const self_type& rhs) const { return m_p < rhs.m_p; }
		bool operator > (const self_type& rhs) const { return m_p > rhs.m_p; }

	private:
		pointer m_p;
		index_t m_step;
	}; // end class aview1d_iter_implementer for step_ind


	template<typename T, class TIndexer>
	struct aview1d_iterators
	{
		typedef random_access_iterator_wrapper<aview1d_iter_implementer<T, TIndexer, true> > const_iterator;
		typedef random_access_iterator_wrapper<aview1d_iter_implementer<T, TIndexer, false> > iterator;

		static const_iterator get_const_iterator(const T *base, const TIndexer& indexer, index_t i)
		{
			typedef aview1d_iter_implementer<T, TIndexer, true> impl;
			return impl(base, indexer, i);
		}

		static iterator get_iterator(T *base, const TIndexer& indexer, index_t i)
		{
			typedef aview1d_iter_implementer<T, TIndexer, false> impl;
			return impl(base, indexer, i);
		}
	};


	template<typename T>
	struct aview1d_iterators<T, id_ind>
	{
		typedef const T* const_iterator;
		typedef T* iterator;

		static const_iterator get_const_iterator(const T *base, const id_ind& indexer, index_t i)
		{
			return base + i;
		}

		static iterator get_iterator(T *base, const id_ind& indexer, index_t i)
		{
			return base + i;
		}
	};

	template<typename T>
	struct aview1d_iterators<T, step_ind>
	{
		typedef random_access_iterator_wrapper<aview1d_step_iter_implementer<T, true> > const_iterator;
		typedef random_access_iterator_wrapper<aview1d_step_iter_implementer<T, false> > iterator;

		static const_iterator get_const_iterator(const T *base, const step_ind& indexer, index_t i)
		{
			typedef aview1d_step_iter_implementer<T, true> impl;
			return impl(base + indexer[i], indexer.step());
		}

		static iterator get_iterator(T *base, const step_ind& indexer, index_t i)
		{
			typedef aview1d_step_iter_implementer<T, false> impl;
			return impl(base + indexer[i], indexer.step());
		}
	};



	// main classes

	template<typename T, class TIndexer>
	class const_aview1d
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T)

		typedef TIndexer indexer_type;
		typedef const_aview1d<value_type, indexer_type> const_view_type;
		typedef aview1d<value_type, indexer_type> view_type;

		typedef aview1d_iterators<T, TIndexer> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		const_aview1d(const_pointer base, const indexer_type& indexer)
		: m_base(const_cast<pointer>(base)), m_ne(indexer.size()), m_indexer(indexer)
		{
		}

		dim_num_t ndims() const
		{
			return num_dims;
		}

		size_type nelems() const
		{
			return m_ne;
		}

		shape_type shape() const
		{
			return arr_shape(m_ne);
		}

		const indexer_type& get_indexer() const
		{
			return m_indexer;
		}

		// Element access

		const_pointer pbase() const
		{
			return m_base;
		}

		const_pointer ptr(index_t i) const
		{
			return m_base + m_indexer[i];
		}

		const_reference operator[] (index_t i) const
		{
			return m_base[m_indexer[i]];
		}

		const_reference operator() (index_t i) const
		{
			return m_base[m_indexer[i]];
		}

		// Iteration

		const_iterator begin() const
		{
			return _iterators::get_const_iterator(m_base, m_indexer, 0);
		}

		const_iterator end() const
		{
			return _iterators::get_const_iterator(m_base, m_indexer, m_ne);
		}

		// Sub-view

		template<class TSelector>
		const_aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>
		V(const TSelector& selector) const
		{
			index_t offset = 0;
			typename sub_indexer<indexer_type, TSelector>::type sindexer =
					sub_indexer<indexer_type, TSelector>::get(m_indexer, selector, offset);

			return const_aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>(
					m_base + offset, sindexer);
		}


	protected:
		pointer m_base;
		size_type m_ne;
		indexer_type m_indexer;

	}; // end class const_aview1d


	template<typename T, class TIndexer>
	class aview1d : public const_aview1d<T, TIndexer>
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T)

		typedef TIndexer indexer_type;
		typedef const_aview1d<value_type, indexer_type> const_view_type;
		typedef aview1d<value_type, indexer_type> view_type;

		typedef aview1d_iterators<T, TIndexer> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		aview1d(const_pointer base, const indexer_type& indexer)
		: const_view_type(base, indexer)
		{
		}

		// Element access

		const_pointer pbase() const
		{
			return this->m_base;
		}

		pointer pbase()
		{
			return this->m_base;
		}

		const_pointer ptr(index_t i) const
		{
			return this->m_base + this->m_indexer[i];
		}

		pointer ptr(index_t i)
		{
			return this->m_base + this->m_indexer[i];
		}

		const_reference operator[] (index_t i) const
		{
			return this->m_base[this->m_indexer[i]];
		}

		reference operator[] (index_t i)
		{
			return this->m_base[this->m_indexer[i]];
		}

		const_reference operator() (index_t i) const
		{
			return this->m_base[this->m_indexer[i]];
		}

		reference operator() (index_t i)
		{
			return this->m_base[this->m_indexer[i]];
		}

		// Iteration

		const_iterator begin() const
		{
			return _iterators::get_const_iterator(this->m_base, this->m_indexer, 0);
		}

		iterator begin()
		{
			return _iterators::get_iterator(this->m_base, this->m_indexer, 0);
		}

		const_iterator end() const
		{
			return _iterators::get_const_iterator(this->m_base, this->m_indexer, this->m_ne);
		}

		iterator end()
		{
			return _iterators::get_iterator(this->m_base, this->m_indexer, this->m_ne);
		}

		// Sub-view

		template<class TSelector>
		const_aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>
		V(const TSelector& selector) const
		{
			index_t offset = 0;
			typename sub_indexer<indexer_type, TSelector>::type sindexer =
					sub_indexer<indexer_type, TSelector>::get(this->m_indexer, selector, offset);

			return const_aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>(
					this->m_base + offset, sindexer);
		}

		template<class TSelector>
		aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>
		V(const TSelector& selector)
		{
			index_t offset = 0;
			typename sub_indexer<indexer_type, TSelector>::type sindexer =
					sub_indexer<indexer_type, TSelector>::get(this->m_indexer, selector, offset);

			return aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>(
					this->m_base + offset, sindexer);
		}

	}; // end class aview1d


	// dense_view judgment

	template<typename T, class TIndexer1, class TIndexer2>
	bool is_same_shape(const const_aview1d<T, TIndexer1>& a, const const_aview1d<T, TIndexer2>& b)
	{
		return a.nelems() == b.nelems();
	}


	template<typename T, class TIndexer>
	bool is_dense_view(const const_aview1d<T, TIndexer>& a)
	{
		return false;
	}

	template<typename T>
	bool is_dense_view(const const_aview1d<T, id_ind>& a)
	{
		return true;
	}

	template<typename T>
	bool is_dense_view(const const_aview1d<T, step_ind>& a)
	{
		return a.get_indexer().step() == 1;
	}


	// Overloaded operators and array manipulation

	// element-wise comparison

	template<typename T, class TIndexer, class TIndexer2>
	inline bool operator == (const const_aview1d<T, TIndexer>& lhs, const const_aview1d<T, TIndexer2>& rhs)
	{
		if (!is_same_shape(lhs, rhs)) return false;
		if (is_dense_view(lhs) && is_dense_view(rhs))
		{
			return elements_equal(lhs.pbase(), rhs.pbase(), lhs.nelems());
		}
		else
		{
			return std::equal(lhs.begin(), lhs.end(), rhs.begin());
		}
	}

	template<typename T, class TIndexer, class TIndexer2>
	inline bool operator != (const const_aview1d<T, TIndexer>& lhs, const const_aview1d<T, TIndexer2>& rhs)
	{
		return !(lhs == rhs);
	}

	// export

	template<typename T, class TIndexer, typename OutputIter>
	inline void export_to(const const_aview1d<T, TIndexer>& a, OutputIter dst)
	{
		copy_n(a.begin(), a.nelems(), dst);
	}

	template<typename T, class TIndexer>
	inline void export_to(const const_aview1d<T, TIndexer>& a, T *dst)
	{
		if (is_dense_view(a))
		{
			copy_elements(a.pbase(), dst, a.nelems());
		}
		else
		{
			copy_n(a.begin(), a.nelems(), dst);
		}
	}

	template<typename T, class TIndexer, class RView>
	inline const const_aview1d<T, TIndexer>& operator >> (const const_aview1d<T, TIndexer>& a, RView& b)
	{
		if (a.nelems() != b.nelems())
		{
			throw array_size_mismatch();
		}

		if (is_dense_view(b))
		{
			export_to(a, b.pbase());
		}
		else
		{
			export_to(a, b.begin());
		}
		return a;
	}


	// import or fill

	template<typename T, class TIndexer, typename InputIter>
	inline void import_from(aview1d<T, TIndexer>& a, InputIter src)
	{
		copy_n(src, a.nelems(), a.begin());
	}

	template<typename T, class TIndexer>
	inline void import_from(aview1d<T, TIndexer>& a, const T *src)
	{
		if (is_dense_view(a))
		{
			copy_elements(src, a.pbase(), a.nelems());
		}
		else
		{
			copy_n(src, a.nelems(), a.begin());
		}
	}

	template<typename T, class TIndexer, class RView>
	inline aview1d<T, TIndexer>& operator << (aview1d<T, TIndexer>& a, const RView& b)
	{
		if (a.nelems() != b.nelems())
		{
			throw array_size_mismatch();
		}

		if (is_dense_view(b))
		{
			import_from(a, b.pbase());
		}
		else
		{
			import_from(a, b.begin());
		}

		return a;
	}

	template<typename T, class TIndexer>
	inline void fill(aview1d<T, TIndexer>& a, const T& x)
	{
		if (is_dense_view(a))
		{
			fill_elements(a.pbase(), a.nelems(), x);
		}
		else
		{
			std::fill(a.begin(), a.end(), x);
		}
	}


	template<typename T>
	inline void set_zeros(aview1d<T, id_ind>& a)
	{
		set_zeros_to_elements(a.pbase(), a.nelems());
	}


	// stand-alone array class

	template<typename T, class Alloc>
	class array1d : public aview1d<T, id_ind>
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T)

		typedef id_ind indexer_type;
		typedef const_aview1d<value_type, indexer_type> const_view_type;
		typedef aview1d<value_type, indexer_type> view_type;

		typedef aview1d_iterators<T, id_ind> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		explicit array1d(size_type n)
		: view_type(0, n), m_pblock(new block<value_type>(n))
		{
			this->m_base = m_pblock->pbase();
		}

		array1d(size_type n, const T& x)
		: view_type(0, n), m_pblock(new block<value_type>(n))
		{
			this->m_base = m_pblock->pbase();

			fill(*this, x);
		}

		template<typename InputIter>
		array1d(size_type n, InputIter src)
		: view_type(0, n), m_pblock(new block<value_type>(n))
		{
			this->m_base = m_pblock->pbase();

			import_from(*this, src);
		}

	private:
		tr1::shared_ptr<block<value_type, Alloc> > m_pblock;

	}; // end class array1d


	// make copy

	template<typename T, class TIndexer>
	array1d<T> make_copy(const const_aview1d<T, TIndexer>& a)
	{
		array1d<T> ac(a.nelems());
		ac << a;
		return ac;
	}


}

#endif 
