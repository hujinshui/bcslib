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
#include <bcslib/array/generic_array_functions.h>

#include <bcslib/base/iterator_wrappers.h>

namespace bcs
{

	// forward declaration

	template<typename T, class TIndexer=id_ind> class aview1d;
	template<typename T, class Alloc=aligned_allocator<T> > class array1d;

	// iterators

	namespace _detail
	{

		template<typename T, class TIndexer>
		class _aview1d_iter_impl
		{
		public:
			typedef typename std::remove_const<T>::type value_type;
			typedef T* pointer;  // the const qualifier should be kept here
			typedef T& reference;

		public:
			_aview1d_iter_impl(): m_base(0), m_pindexer(0), m_i(0) { }

			_aview1d_iter_impl(pointer base, const TIndexer& indexer, index_t i)
			: m_base(base), m_pindexer(&indexer), m_i(i) { }

			pointer ptr() const { return m_base + offset(m_i); }
			reference ref() const { return m_base[offset(m_i)]; }
			reference at(index_t n) const { return m_base[offset(m_i + n)]; }

			void move_next() { ++ m_i; }
			void move_prev() { -- m_i; }
			void move_forward(ptrdiff_t n) { m_i += static_cast<index_t>(n); }
			void move_backward(ptrdiff_t n) { m_i -= static_cast<index_t>(n); }

			bool operator == (const _aview1d_iter_impl& rhs) const { return m_i == rhs.m_i; }
			bool operator < (const _aview1d_iter_impl& rhs) const { return m_i < rhs.m_i; }
			bool operator > (const _aview1d_iter_impl& rhs) const { return m_i > rhs.m_i; }

			index_t operator - (const _aview1d_iter_impl& rhs) const
			{
				return m_i - rhs.m_i;
			}

		private:
			index_t offset(index_t i) const
			{
				return m_pindexer->operator[](m_i);
			}

			pointer m_base;
			const TIndexer *m_pindexer;
			index_t m_i;

		}; // end class aview1d_iter_implementer


		template<typename T>
		class _aview1d_iter_impl<T, step_ind>
		{
		public:
			typedef typename std::remove_const<T>::type value_type;
			typedef T* pointer;  // the const qualifier should be kept here
			typedef T& reference;

		public:
			_aview1d_iter_impl() : m_p(0), m_step(0) { }

			_aview1d_iter_impl(pointer base, const step_ind& indexer, index_t i)
			: m_p(base + indexer[i]), m_step(indexer.step()) { }

			pointer ptr() const { return m_p; }
			reference ref() const { return *m_p; }
			reference at(index_t n) const { return m_p[n * m_step]; }

			void move_next() { m_p += m_step; }
			void move_prev() { m_p -= m_step; }
			void move_forward(ptrdiff_t n) { m_p += n * m_step; }
			void move_backward(ptrdiff_t n) { m_p -= n * m_step; }

			bool operator == (const _aview1d_iter_impl& rhs) const
			{
				return m_p == rhs.m_p;
			}
			bool operator < (const _aview1d_iter_impl& rhs) const
			{
				return (bool)((m_step < 0) ^ (m_p < rhs.m_p));
			}
			bool operator > (const _aview1d_iter_impl& rhs) const
			{
				return (bool)((m_step < 0) ^ (m_p > rhs.m_p));
			}

			ptrdiff_t operator - (const _aview1d_iter_impl& rhs) const
			{
				return (m_p - rhs.m_p) / m_step;
			}

		private:
			pointer m_p;
			index_t m_step;
		}; // end class aview1d_iter_implementer for step_ind
	}


	// iterator selection helpers

	template<typename T, class TIndexer>
	struct aview1d_iterators
	{
		typedef typename std::remove_const<T>::type value_type;

		typedef _detail::_aview1d_iter_impl<const value_type, TIndexer> _const_iter_implementer;
		typedef _detail::_aview1d_iter_impl<value_type, TIndexer> _iter_implementer;

		typedef random_access_iterator_wrapper<_const_iter_implementer> const_iterator;
		typedef random_access_iterator_wrapper<_iter_implementer> iterator;

		static const_iterator get_const_iterator(const T *base, const TIndexer& indexer, index_t i)
		{
			return _const_iter_implementer(base, indexer, i);
		}

		static iterator get_iterator(T *base, const TIndexer& indexer, index_t i)
		{
			return _iter_implementer(base, indexer, i);
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


	// main classes

	template<typename T, class TIndexer>
	class aview1d
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef TIndexer indexer_type;
		typedef aview1d<value_type, indexer_type> view_type;

		typedef aview1d_iterators<value_type, indexer_type> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		dim_num_t ndims() const
		{
			return num_dims;
		}

		size_type size() const
		{
			return nelems();
		}

		size_type nelems() const
		{
			return static_cast<size_type>(m_d0);
		}

		index_type dim0() const
		{
			return m_d0;
		}

		shape_type shape() const
		{
			return arr_shape(m_d0);
		}

		const indexer_type& get_indexer() const
		{
			return m_indexer;
		}

	public:
		aview1d(pointer pbase, const indexer_type& indexer)
		: m_pbase(pbase)
		, m_d0(static_cast<index_t>(indexer.size()))
		, m_indexer(indexer)
		{
		}

		aview1d(pointer pbase, indexer_type&& indexer)
		: m_pbase(pbase)
		, m_d0(static_cast<index_t>(indexer.size()))
		, m_indexer(std::move(indexer))
		{
		}

		aview1d(const aview1d& r)
		: m_pbase(r.m_pbase)
		, m_d0(r.m_d0)
		, m_indexer(r.m_indexer)
		{
		}

		aview1d(aview1d&& r)
		: m_pbase(r.m_pbase)
		, m_d0(r.m_d0)
		, m_indexer(std::move(r.m_indexer))
		{
			r.reset();
		}

		aview1d& operator = (const aview1d& r)
		{
			m_pbase = r.m_pbase;
			m_d0 = r.m_d0;
			m_indexer = r.m_indexer;
			return *this;
		}

		aview1d& operator = (aview1d&& r)
		{
			m_pbase = r.m_pbase;
			m_d0 = r.m_d0;
			m_indexer = std::move(r.m_indexer);

			r.reset();

			return *this;
		}

		inline operator array1d<T>() const;

	public:
		// Element access

		const_pointer pbase() const
		{
			return m_pbase;
		}

		pointer pbase()
		{
			return m_pbase;
		}

		const_pointer ptr(index_type i) const
		{
			return m_pbase + m_indexer[i];
		}

		pointer ptr(index_type i)
		{
			return m_pbase + m_indexer[i];
		}

		const_reference operator[] (index_type i) const
		{
			return m_pbase[m_indexer[i]];
		}

		reference operator[] (index_type i)
		{
			return m_pbase[m_indexer[i]];
		}

		const_reference operator() (index_type i) const
		{
			return m_pbase[m_indexer[i]];
		}

		reference operator() (index_type i)
		{
			return m_pbase[m_indexer[i]];
		}

		// Iteration

		const_iterator begin() const
		{
			return _iterators::get_const_iterator(pbase(), m_indexer, 0);
		}

		iterator begin()
		{
			return _iterators::get_iterator(pbase(), m_indexer, 0);
		}

		const_iterator end() const
		{
			return _iterators::get_const_iterator(pbase(), m_indexer, m_d0);
		}

		iterator end()
		{
			return _iterators::get_iterator(pbase(), m_indexer, m_d0);
		}

		// Sub-view

		template<class TSelector>
		const aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>
		V(const TSelector& selector) const
		{
			typedef typename sub_indexer<indexer_type, TSelector>::type sub_indexer_type;

			index_t offset;
			sub_indexer_type sindexer =
					sub_indexer<indexer_type, TSelector>::get(m_indexer, selector, offset);

			return aview1d<value_type, sub_indexer_type>(m_pbase + offset, sindexer);
		}

		template<class TSelector>
		aview1d<value_type, typename sub_indexer<indexer_type, TSelector>::type>
		V(const TSelector& selector)
		{
			typedef typename sub_indexer<indexer_type, TSelector>::type sub_indexer_type;

			index_t offset;
			sub_indexer_type sindexer =
					sub_indexer<indexer_type, TSelector>::get(m_indexer, selector, offset);

			return aview1d<value_type, sub_indexer_type>(m_pbase + offset, sindexer);
		}

	private:
		void reset()
		{
			m_pbase = BCS_NULL;
			m_d0 = 0;
		}

	protected:
		pointer m_pbase;
		index_type m_d0;
		indexer_type m_indexer;

	}; // end class aview1d


	// stand-alone array class

	template<typename T, class Alloc>
	class array1d : private sharable_storage_base<T, Alloc>, public aview1d<T, id_ind>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef id_ind indexer_type;
		typedef aview1d<value_type, indexer_type> view_type;

		typedef aview1d_iterators<value_type, indexer_type> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

		typedef sharable_storage_base<T, Alloc> storage_base_type;

	public:
		explicit array1d(size_type n)
		: storage_base_type(n), view_type(storage_base_type::pointer_to_base(), n)
		{
		}

		array1d(size_type n, const value_type& x)
		: storage_base_type(n, x), view_type(storage_base_type::pointer_to_base(), n)
		{
		}

		array1d(size_type n, const_pointer src)
		: storage_base_type(n, src), view_type(storage_base_type::pointer_to_base(), n)
		{
		}

		array1d(const array1d& r)
		: storage_base_type(r), view_type(storage_base_type::pointer_to_base(), r.nelems())
		{
		}

		array1d(array1d&& r)
		: storage_base_type(std::move(r)), view_type(std::move(r))
		{
		}

		template<typename ForwardIterator>
		array1d(size_type n, ForwardIterator it)
		: storage_base_type(n), view_type(storage_base_type::pointer_to_base(), n)
		{
			import_from(*this, it);
		}

		array1d& operator = (const array1d& r)
		{
			if (this != &r)
			{
				storage_base_type &s = *this;
				view_type& v = *this;

				s = r;
				v = view_type(s.pointer_to_base(), r.nelems());
			}
			return *this;
		}

		array1d& operator = (array1d&& r)
		{
			storage_base_type &s = *this;
			view_type& v = *this;

			s = std::move(r);
			v = view_type(s.pointer_to_base(), r.nelems());

			return *this;
		}

		void swap(array1d& r)
		{
			using std::swap;

			storage_base_type::swap(r);

			view_type& v = *this;
			view_type& rv = r;
			swap(v, rv);
		}

		// sharing

	public:
		array1d(const array1d& r, do_share ds)
		: storage_base_type(r, ds), view_type(storage_base_type::pointer_to_base(), r.nelems())
		{

		}

		array1d shared_copy() const
		{
			return array1d(*this, do_share());
		}

	}; // end class array1d


	template<typename T, class Alloc>
	inline void swap(array1d<T, Alloc>& lhs, array1d<T, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}


	/********************************************
	 *
	 *   Concept-required interfaces
	 *
	 ********************************************/

	template<typename T, class TIndexer>
	struct is_array_view<bcs::aview1d<T, TIndexer> > { static const bool value = true; };

	template<typename T, class Alloc>
	struct is_array_view<bcs::array1d<T, Alloc> > { static const bool value = true; };

	template<typename T, class TIndexer>
	struct is_array_view_ndim<bcs::aview1d<T, TIndexer>, 1> { static const bool value = true; };

	template<typename T, class Alloc>
	struct is_array_view_ndim<bcs::array1d<T, Alloc>, 1> { static const bool value = true; };


	template<typename T, class TIndexer>
	inline std::array<index_t, 1> get_array_shape(const bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.shape();
	}

	template<typename T, class TIndexer>
	inline size_t get_num_elems(const bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.nelems();
	}

	template<typename T, class TIndexer>
	inline const T& get(const bcs::aview1d<T, TIndexer>& arr, index_t i)
	{
		return arr(i);
	}

	template<typename T, class TIndexer>
	inline void set(const T& v, bcs::aview1d<T, TIndexer>& arr, index_t i)
	{
		arr(i) = v;
	}

	template<typename T, class TIndexer>
	inline typename bcs::aview1d<T, TIndexer>::const_iterator begin(const bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.begin();
	}

	template<typename T, class TIndexer>
	inline typename bcs::aview1d<T, TIndexer>::const_iterator end(const bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.end();
	}

	template<typename T, class TIndexer>
	inline typename bcs::aview1d<T, TIndexer>::iterator begin(bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.begin();
	}

	template<typename T, class TIndexer>
	inline typename bcs::aview1d<T, TIndexer>::iterator end(bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.end();
	}

	template<typename T, class TIndexer>
	inline const T* ptr_base(const bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.pbase();
	}

	template<typename T, class TIndexer>
	inline T* ptr_base(bcs::aview1d<T, TIndexer>& arr)
	{
		return arr.pbase();
	}

	template<typename T, class TIndexer>
	inline bool is_dense_view(const aview1d<T, TIndexer>& a)
	{
		return array_indexer_traits<TIndexer>::is_continuous(a.get_indexer());
	}



	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>::operator array1d<T>() const
	{
		array1d<T> arr(nelems());
		if (is_dense_view(*this))
		{
			import_from(arr, pbase());
		}
		else
		{
			import_from(arr, begin());
		}
		return arr;
	}


	template<typename T, class TIndexer>
	inline array1d<T> clone_array(const aview1d<T, TIndexer>& view)
	{
		return view;
	}

	template<typename T, class TIndexer>
	struct array_creater<aview1d<T, TIndexer> >
	{
		typedef std::array<index_t, 1> shape_type;
		typedef array1d<T> result_type;

		template<typename U>
		struct remap
		{
			typedef array1d<U> result_type;

			static result_type create(const shape_type& shape)
			{
				return result_type(static_cast<size_t>(shape[0]));
			}
		};

		static result_type create(const shape_type& shape)
		{
			return result_type(static_cast<size_t>(shape[0]));
		}

		static result_type copy(const aview1d<T, TIndexer>& view)
		{
			return view;
		}
	};


	/******************************************************
	 *
	 *  Overloaded operators
	 *
	 ******************************************************/

	// element-wise comparison

	template<typename T, class LIndexer, class RIndexer>
	inline bool operator == (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return equal_array(lhs, rhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline bool operator != (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return !(lhs == rhs);
	}

	// export & import

	template<typename T, class LIndexer, class RIndexer>
	inline const aview1d<T, LIndexer>&
	operator >> (const aview1d<T, LIndexer>& a, aview1d<T, RIndexer>& b)
	{
		if (get_num_elems(a) != get_num_elems(b))
		{
			throw array_dim_mismatch();
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

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>&
	operator << (aview1d<T, LIndexer>& a, const aview1d<T, RIndexer>& b)
	{
		if (get_num_elems(a) != get_num_elems(b))
		{
			throw array_dim_mismatch();
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


	/***
	 *
	 *  generic implementation of indexer-related classes using array1d
	 *
	 */

	typedef array1d<index_t> arr_ind;

	template<class TIndexer>
	struct inject_step
	{
		typedef arr_ind type;

		static type get(const TIndexer& idx0, index_t step)
		{
			array1d<index_t> a(idx0.size());
			index_t n = a.dim0();
			for (index_t i = 0; i < n; ++i)
			{
				a[i] = step * idx0[i];
			}
			return a;
		}
	};


	template<class TIndexer, class TSelector>
	struct sub_indexer
	{
		typedef arr_ind type;

		static type get(const TIndexer& base_indexer, const TSelector& sel, index_t& offset)
		{
			offset = 0;

			typedef typename index_selector_traits<TIndexer>::input_type input_t;
			array1d<index_t> a(sel.size());
			input_t n = static_cast<input_t>(sel.size());

			for (input_t i = 0; i < n; ++i)
			{
				a[i] = base_indexer[sel[i]];
			}
			return a;
		}
	};

}

#endif 
