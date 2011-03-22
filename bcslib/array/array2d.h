/**
 * @file array2d.h
 *
 * Two dimensional array classes
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY2D_H
#define BCSLIB_ARRAY2D_H

#include <bcslib/array/array1d.h>

namespace bcs
{
	// forward declaration

	template<typename T, typename TOrd, class TIndexer0=id_ind, class TIndexer1=id_ind> class const_aview2d;
	template<typename T, typename TOrd, class TIndexer0=id_ind, class TIndexer1=id_ind> class aview2d;
	template<typename T, typename TOrd, class Alloc=std::allocator<T> > class array2d;


	// order-specific offset calculation

	namespace _detail
	{
		template<typename TOrd> struct layout_aux2d;

		template<>
		struct layout_aux2d<row_major_t>
		{
			static index_t offset(size_t m, size_t n, index_t i, index_t j)
			{
				return i * n + j;
			}

			static void pass_by_end(size_t m, size_t n, index_t& i, index_t& j)
			{
				i = (index_t)m;
				j = 0;
			}
		};


		template<>
		struct layout_aux2d<column_major_t>
		{
			static index_t offset(size_t m, size_t n, index_t i, index_t j)
			{
				return i + j * m;
			}

			static void pass_by_end(size_t m, size_t n, index_t& i, index_t& j)
			{
				i = 0;
				j = (index_t)n;
			}
		};


		template<typename TOrd, class TIndexer0, class TIndexer1> struct array2d_slices0;
		template<typename TOrd, class TIndexer0, class TIndexer1> struct array2d_slices1;

		template<class TIndexer0, class TIndexer1>
		struct array2d_slices0<row_major_t, TIndexer0, TIndexer1>
		{
			typedef TIndexer1 indexer_type;

			static indexer_type get_indexer(size_t base_d0, size_t base_d1, const TIndexer1& idx1)
			{
				return idx1;
			}

			static index_t slice_offset(size_t base_d0, size_t base_d1, const TIndexer0& idx0, index_t i)
			{
				return idx0[i] * base_d1;
			}

		};

		template<class TIndexer0, class TIndexer1>
		struct array2d_slices1<row_major_t, TIndexer0, TIndexer1>
		{
			typedef typename step_injecter<TIndexer0>::type indexer_type;

			static indexer_type get_indexer(size_t base_d0, size_t base_d1, const TIndexer0& idx0)
			{
				return step_injecter<TIndexer0>::get(idx0, base_d1);
			}

			static index_t slice_offset(size_t base_d0, size_t base_d1, const TIndexer1& idx1, index_t j)
			{
				return idx1[j];
			}
		};


		template<class TIndexer0, class TIndexer1>
		struct array2d_slices0<column_major_t, TIndexer0, TIndexer1>
		{
			typedef typename step_injecter<TIndexer1>::type indexer_type;

			static indexer_type get_indexer(size_t base_d0, size_t base_d1, const TIndexer1& idx1)
			{
				return step_injecter<TIndexer1>::get(idx1, base_d0);
			}

			static index_t slice_offset(size_t base_d0, size_t base_d1, const TIndexer0& idx0, index_t i)
			{
				return idx0[i];
			}

		};

		template<class TIndexer0, class TIndexer1>
		struct array2d_slices1<column_major_t, TIndexer0, TIndexer1>
		{
			typedef TIndexer0 indexer_type;

			static indexer_type get_indexer(size_t base_d0, size_t base_d1, const TIndexer0& idx0)
			{
				return idx0;
			}

			static index_t slice_offset(size_t base_d0, size_t base_d1, const TIndexer1& idx1, index_t j)
			{
				return idx1[j] * base_d0;
			}
		};


		template<typename T, typename TOrd, class TIndexer0, class TIndexer1, bool IsConst> struct view2d_types;

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		struct view2d_types<T, TOrd, TIndexer0, TIndexer1, true>
		{
			typedef const_aview2d<T, TOrd, TIndexer0, TIndexer1> view_type;
			typedef const view_type& view_reference;
			typedef const view_type* view_pointer;
		};

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		struct view2d_types<T, TOrd, TIndexer0, TIndexer1, false>
		{
			typedef aview2d<T, TOrd, TIndexer0, TIndexer1> view_type;
			typedef view_type& view_reference;
			typedef view_type* view_pointer;
		};
	}



	// iterations (declaration)

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, bool IsConst> class aview2d_iter_implementer;

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct aview2d_iterators
	{
		typedef forward_iterator_wrapper<aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, true> > const_iterator;
		typedef forward_iterator_wrapper<aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, false> > iterator;

		typedef const_aview2d<T, TOrd, TIndexer0, TIndexer1> const_view_type;
		typedef aview2d<T, TOrd, TIndexer0, TIndexer1> view_type;

		static inline const_iterator get_const_iterator(const const_view_type& view, index_t i, index_t j);
		static inline iterator get_iterator(view_type& view, index_t i, index_t j);
	};


	// main classes

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class const_aview2d
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T)

		typedef TOrd layout_order;
		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef const_aview2d<value_type, layout_order, indexer0_type, indexer1_type> const_view_type;
		typedef aview2d<value_type, layout_order, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, TIndexer0, TIndexer1> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

		typedef typename _detail::array2d_slices0<TOrd, TIndexer0, TIndexer1>::indexer_type slices0_indexer_type;
		typedef typename _detail::array2d_slices1<TOrd, TIndexer0, TIndexer1>::indexer_type slices1_indexer_type;

	public:
		const_aview2d(const_pointer base, size_type base_d0, size_type base_d1,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_base(const_cast<pointer>(base))
		, m_base_d0(base_d0), m_base_d1(base_d1)
		, m_ne(indexer0.size() * indexer1.size())
		, m_indexer0(indexer0), m_indexer1(indexer1)
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

		size_type dim0() const
		{
			return m_indexer0.size();
		}

		size_type dim1() const
		{
			return m_indexer1.size();
		}

		size_type nrows() const
		{
			return dim0();
		}

		size_type ncolumns() const
		{
			return dim1();
		}

		shape_type shape() const
		{
			return arr_shape(dim0(), dim1());
		}

		const indexer0_type& get_indexer0() const
		{
			return m_indexer0;
		}

		const indexer1_type& get_indexer1() const
		{
			return m_indexer1;
		}

		size_type base_dim0() const
		{
			return m_base_d0;
		}

		size_type base_dim1() const
		{
			return m_base_d1;
		}

		shape_type base_shape() const
		{
			return arr_shape(base_dim0(), base_dim1());
		}


		// Element access

		const_pointer pbase() const
		{
			return m_base;
		}

		index_t offset_at(index_t i, index_t j) const
		{
			return _detail::layout_aux2d<layout_order>::offset(
					base_dim0(), base_dim1(), m_indexer0[i], m_indexer1[j]);
		}

		const_pointer ptr(index_t i, index_t j) const
		{
			return m_base + offset_at(i, j);
		}

		const_reference operator() (index_t i, index_t j) const
		{
			return m_base[offset_at(i, j)];
		}

		// Iteration

		const_iterator begin() const
		{
			return _iterators::get_const_iterator(*this, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(dim0(), dim1(), e_i, e_j);
			return _iterators::get_const_iterator(*this, e_i, e_j);
		}


		// Slice

		const_aview1d<value_type, slices0_indexer_type> sliceI0(index_t i) const
		{
			typedef _detail::array2d_slices0<layout_order, indexer0_type, indexer1_type> _slices;

			return const_aview1d<value_type, slices0_indexer_type>(
					m_base + _slices::slice_offset(m_base_d0, m_base_d1, m_indexer0, i),
					_slices::get_indexer(m_base_d0, m_base_d1, m_indexer1));
		}

		const_aview1d<value_type, slices1_indexer_type> sliceI1(index_t j) const
		{
			typedef _detail::array2d_slices1<layout_order, indexer0_type, indexer1_type> _slices;

			return const_aview1d<value_type, slices1_indexer_type>(
					m_base + _slices::slice_offset(m_base_d0, m_base_d1, m_indexer1, j),
					_slices::get_indexer(m_base_d0, m_base_d1, m_indexer0));
		}

		const_aview1d<value_type, slices0_indexer_type> row(index_t i) const
		{
			return sliceI0(i);
		}

		const_aview1d<value_type, slices1_indexer_type> column(index_t j) const
		{
			return sliceI1(j);
		}


		// Sub-View

		template<class TSelector0, class TSelector1>
		const_aview2d<value_type, layout_order,
			typename sub_indexer<indexer0_type, TSelector0>::type,
			typename sub_indexer<indexer1_type, TSelector1>::type>
		V(const TSelector0& sel0, const TSelector1& sel1) const
		{
			typedef typename sub_indexer<indexer0_type, TSelector0>::type sub_indexer0_t;
			typedef typename sub_indexer<indexer1_type, TSelector1>::type sub_indexer1_t;

			index_t o0 = 0;
			index_t o1 = 0;

			sub_indexer0_t si0 = sub_indexer<indexer0_type, TSelector0>::get(m_indexer0, sel0, o0);
			sub_indexer1_t si1 = sub_indexer<indexer1_type, TSelector1>::get(m_indexer1, sel1, o1);

			index_t offset = _detail::layout_aux2d<layout_order>::offset(m_base_d0, m_base_d1, o0, o1);

			return const_aview2d<value_type, layout_order, sub_indexer0_t, sub_indexer1_t>(
					m_base + offset, m_base_d0, m_base_d1, si0, si1);
		}


	protected:
		pointer m_base;
		size_type m_base_d0;
		size_type m_base_d1;

		size_type m_ne;
		indexer0_type m_indexer0;
		indexer1_type m_indexer1;

	}; // end class const_aview2d



	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class aview2d : public const_aview2d<T, TOrd, TIndexer0, TIndexer1>
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T)

		typedef TOrd layout_order;
		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef const_aview2d<value_type, layout_order, indexer0_type, indexer1_type> const_view_type;
		typedef aview2d<value_type, layout_order, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, TIndexer0, TIndexer1> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

		typedef typename _detail::array2d_slices0<TOrd, TIndexer0, TIndexer1>::indexer_type slices0_indexer_type;
		typedef typename _detail::array2d_slices1<TOrd, TIndexer0, TIndexer1>::indexer_type slices1_indexer_type;

	public:
		aview2d(const_pointer base, size_type base_d0, size_type base_d1,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: const_view_type(base, base_d0, base_d1, indexer0, indexer1)
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

		const_pointer ptr(index_t i, index_t j) const
		{
			return this->m_base + this->offset_at(i, j);
		}

		pointer ptr(index_t i, index_t j)
		{
			return this->m_base + this->offset_at(i, j);
		}

		const_reference operator() (index_t i, index_t j) const
		{
			return this->m_base[this->offset_at(i, j)];
		}

		reference operator() (index_t i, index_t j)
		{
			return this->m_base[this->offset_at(i, j)];
		}

		// Iteration

		const_iterator begin() const
		{
			return _iterators::get_const_iterator(*this, 0, 0);
		}

		iterator begin()
		{
			return _iterators::get_iterator(*this, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(this->dim0(), this->dim1(), e_i, e_j);
			return _iterators::get_const_iterator(*this, e_i, e_j);
		}

		iterator end()
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(this->dim0(), this->dim1(), e_i, e_j);
			return _iterators::get_iterator(*this, e_i, e_j);
		}


		// Slice

		const_aview1d<value_type, slices0_indexer_type> sliceI0(index_t i) const
		{
			typedef _detail::array2d_slices0<layout_order, indexer0_type, indexer1_type> _slices;

			return const_aview1d<value_type, slices0_indexer_type>(
					this->m_base + _slices::slice_offset(this->m_base_d0, this->m_base_d1, this->m_indexer0, i),
					_slices::get_indexer(this->m_base_d0, this->m_base_d1, this->m_indexer1));
		}

		aview1d<value_type, slices0_indexer_type> sliceI0(index_t i)
		{
			typedef _detail::array2d_slices0<layout_order, indexer0_type, indexer1_type> _slices;

			return aview1d<value_type, slices0_indexer_type>(
					this->m_base + _slices::slice_offset(this->m_base_d0, this->m_base_d1, this->m_indexer0, i),
					_slices::get_indexer(this->m_base_d0, this->m_base_d1, this->m_indexer1));
		}

		const_aview1d<value_type, slices1_indexer_type> sliceI1(index_t j) const
		{
			typedef _detail::array2d_slices1<layout_order, indexer0_type, indexer1_type> _slices;

			return const_aview1d<value_type, slices1_indexer_type>(
					this->m_base + _slices::slice_offset(this->m_base_d0, this->m_base_d1, this->m_indexer1, j),
					_slices::get_indexer(this->m_base_d0, this->m_base_d1, this->m_indexer0));
		}

		aview1d<value_type, slices1_indexer_type> sliceI1(index_t j)
		{
			typedef _detail::array2d_slices1<layout_order, indexer0_type, indexer1_type> _slices;

			return aview1d<value_type, slices1_indexer_type>(
					this->m_base + _slices::slice_offset(this->m_base_d0, this->m_base_d1, this->m_indexer1, j),
					_slices::get_indexer(this->m_base_d0, this->m_base_d1, this->m_indexer0));
		}

		const_aview1d<value_type, slices0_indexer_type> row(index_t i) const
		{
			return sliceI0(i);
		}

		aview1d<value_type, slices0_indexer_type> row(index_t i)
		{
			return sliceI0(i);
		}

		const_aview1d<value_type, slices1_indexer_type> column(index_t j) const
		{
			return sliceI1(j);
		}

		aview1d<value_type, slices1_indexer_type> column(index_t j)
		{
			return sliceI1(j);
		}

		// Sub-view

		template<class TSelector0, class TSelector1>
		const_aview2d<value_type, layout_order,
			typename sub_indexer<indexer0_type, TSelector0>::type,
			typename sub_indexer<indexer1_type, TSelector1>::type>
		V(const TSelector0& sel0, const TSelector1& sel1) const
		{
			typedef typename sub_indexer<indexer0_type, TSelector0>::type sub_indexer0_t;
			typedef typename sub_indexer<indexer1_type, TSelector1>::type sub_indexer1_t;

			index_t o0 = 0;
			index_t o1 = 0;

			sub_indexer0_t si0 = sub_indexer<indexer0_type, TSelector0>::get(this->m_indexer0, sel0, o0);
			sub_indexer1_t si1 = sub_indexer<indexer1_type, TSelector1>::get(this->m_indexer1, sel1, o1);

			index_t offset = _detail::layout_aux2d<layout_order>::offset(this->m_base_d0, this->m_base_d1, o0, o1);

			return const_aview2d<value_type, layout_order, sub_indexer0_t, sub_indexer1_t>(
					this->m_base + offset, this->m_base_d0, this->m_base_d1, si0, si1);
		}

		template<class TSelector0, class TSelector1>
		aview2d<value_type, layout_order,
			typename sub_indexer<indexer0_type, TSelector0>::type,
			typename sub_indexer<indexer1_type, TSelector1>::type>
		V(const TSelector0& sel0, const TSelector1& sel1)
		{
			typedef typename sub_indexer<indexer0_type, TSelector0>::type sub_indexer0_t;
			typedef typename sub_indexer<indexer1_type, TSelector1>::type sub_indexer1_t;

			index_t o0 = 0;
			index_t o1 = 0;

			sub_indexer0_t si0 = sub_indexer<indexer0_type, TSelector0>::get(this->m_indexer0, sel0, o0);
			sub_indexer1_t si1 = sub_indexer<indexer1_type, TSelector1>::get(this->m_indexer1, sel1, o1);

			index_t offset = _detail::layout_aux2d<layout_order>::offset(this->m_base_d0, this->m_base_d1, o0, o1);

			return aview2d<value_type, layout_order, sub_indexer0_t, sub_indexer1_t>(
					this->m_base + offset, this->m_base_d0, this->m_base_d1, si0, si1);
		}


	}; // end class aview2d


	// functions to make dense view

	template<typename T, typename TOrd>
	const_aview2d<T, TOrd, id_ind, id_ind> dense_const_aview2d(const T *pbase, size_t m, size_t n, TOrd ord)
	{
		return const_aview2d<T, TOrd, id_ind, id_ind>(pbase, m, n, m, n);
	}

	template<typename T, typename TOrd>
	aview2d<T, TOrd, id_ind, id_ind> dense_aview2d(T *pbase, size_t m, size_t n, TOrd ord)
	{
		return aview2d<T, TOrd, id_ind, id_ind>(pbase, m, n, m, n);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	bool is_dense_view(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& view)
	{
		return false;
	}

	template<typename T, typename TOrd>
	bool is_dense_view(const const_aview2d<T, TOrd, id_ind, id_ind>& view)
	{
		return view.dim0() == view.base_dim0() && view.dim1() == view.base_dim1();
	}

	template<typename T, typename TOrd>
	bool is_dense_view(const const_aview2d<T, TOrd, step_ind, id_ind>& view)
	{
		return view.dim0() == view.base_dim0() && view.dim1() == view.base_dim1()
				&& view.get_indexer0().step() == 1;
	}

	template<typename T, typename TOrd>
	bool is_dense_view(const const_aview2d<T, TOrd, id_ind, step_ind>& view)
	{
		return view.dim0() == view.base_dim0() && view.dim1() == view.base_dim1()
				&& view.get_indexer1().step() == 1;
	}

	template<typename T, typename TOrd>
	bool is_dense_view(const const_aview2d<T, TOrd, step_ind, step_ind>& view)
	{
		return view.dim0() == view.base_dim0() && view.dim1() == view.base_dim1()
				&& view.get_indexer0().step() == 1 && view.get_indexer1().step() == 1;
	}

	// Iterations

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, bool IsConst>
	class aview2d_iter_implementer;

	template<typename T, class TIndexer0, class TIndexer1, bool IsConst>
	class aview2d_iter_implementer<T, row_major_t, TIndexer0, TIndexer1, IsConst>
	{
	public:

		typedef T value_type;
		typedef typename pointer_and_reference<T, IsConst>::pointer pointer;
		typedef typename pointer_and_reference<T, IsConst>::reference reference;

		typedef row_major_t layout_order;
		typedef typename _detail::view2d_types<T, layout_order, TIndexer0, TIndexer1, IsConst>::view_reference view_reference;
		typedef typename _detail::view2d_types<T, layout_order, TIndexer0, TIndexer1, IsConst>::view_pointer view_pointer;

		typedef aview2d_iter_implementer<T, layout_order, TIndexer0, TIndexer1, IsConst> self_type;

	public:
		aview2d_iter_implementer()
		: m_pView(0), m_d0_m1(0), m_d1_m1(0), m_i(0), m_j(0), m_p(0)
		{
		}

		aview2d_iter_implementer(view_reference view, index_t i, index_t j)
		: m_pView(&view), m_d0_m1((index_t)view.dim0() - 1), m_d1_m1((index_t)view.dim1() - 1), m_i(i), m_j(j)
		, m_p(view.ptr(i, j))
		{
		}

		pointer ptr() const
		{
			return m_p;
		}

		reference ref() const
		{
			return *m_p;
		}

		bool operator == (const self_type& rhs) const
		{
			return m_i == rhs.m_i && m_j == rhs.m_j;
		}

		void move_next()
		{
			if (m_j < m_d1_m1)
			{
				m_p += m_pView->get_indexer1().step_at(m_j ++);
			}
			else
			{
				++ m_i;
				m_j = 0;
				m_p = m_pView->ptr(m_i, m_j);
			}
		}

	private:
		view_pointer m_pView;

		index_t m_d0_m1;
		index_t m_d1_m1;

		index_t m_i;
		index_t m_j;
		pointer m_p;

	}; // end class aview2d_iter_implementer for row_major


	template<typename T, class TIndexer0, class TIndexer1, bool IsConst>
	class aview2d_iter_implementer<T, column_major_t, TIndexer0, TIndexer1, IsConst>
	{
	public:

		typedef T value_type;
		typedef typename pointer_and_reference<T, IsConst>::pointer pointer;
		typedef typename pointer_and_reference<T, IsConst>::reference reference;

		typedef column_major_t layout_order;
		typedef typename _detail::view2d_types<T, layout_order, TIndexer0, TIndexer1, IsConst>::view_reference view_reference;
		typedef typename _detail::view2d_types<T, layout_order, TIndexer0, TIndexer1, IsConst>::view_pointer view_pointer;

		typedef aview2d_iter_implementer<T, layout_order, TIndexer0, TIndexer1, IsConst> self_type;

	public:
		aview2d_iter_implementer()
		: m_pView(0), m_d0_m1(0), m_d1_m1(0), m_i(0), m_j(0), m_p(0)
		{
		}

		aview2d_iter_implementer(view_reference view, index_t i, index_t j)
		: m_pView(&view), m_d0_m1((index_t)view.dim0() - 1), m_d1_m1((index_t)view.dim1() - 1), m_i(i), m_j(j)
		, m_p(view.ptr(i, j))
		{
		}

		pointer ptr() const
		{
			return m_p;
		}

		reference ref() const
		{
			return *m_p;
		}

		bool operator == (const self_type& rhs) const
		{
			return m_i == rhs.m_i && m_j == rhs.m_j;
		}

		void move_next()
		{
			if (m_i < m_d0_m1)
			{
				m_p += m_pView->get_indexer0().step_at(m_i ++);
			}
			else
			{
				++ m_j;
				m_i = 0;
				m_p = m_pView->ptr(m_i, m_j);
			}
		}

	private:
		view_pointer m_pView;

		index_t m_d0_m1;
		index_t m_d1_m1;

		index_t m_i;
		index_t m_j;
		pointer m_p;

	}; // end class aview2d_iter_implementer for row_major


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline typename aview2d_iterators<T, TOrd, TIndexer0, TIndexer1>::const_iterator
	aview2d_iterators<T, TOrd, TIndexer0, TIndexer1>::get_const_iterator(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& view, index_t i, index_t j)
	{
		return aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, true>(view, i, j);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline typename aview2d_iterators<T, TOrd, TIndexer0, TIndexer1>::iterator
	aview2d_iterators<T, TOrd, TIndexer0, TIndexer1>::get_iterator(aview2d<T, TOrd, TIndexer0, TIndexer1>& view, index_t i, index_t j)
	{
		return aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, false>(view, i, j);
	}




	// Overloaded operators and array manipulation

	// element-wise comparison

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline bool is_same_shape(
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return lhs.dim0() == rhs.dim0() && lhs.dim1() == rhs.dim1();
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline bool operator == (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
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

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline bool operator != (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return !(lhs == rhs);
	}

	// export

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename OutputIter>
	inline void export_to(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& a, OutputIter dst)
	{
		std::copy(a.begin(), a.end(), dst);

	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline void export_to(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& a, T* dst)
	{
		if (is_dense_view(a))
		{
			copy_elements(a.pbase(), dst, a.nelems());
		}
		else
		{
			std::copy(a.begin(), a.end(), dst);
		}
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RView>
	inline const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& operator >> (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& a, RView& b)
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

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename InputIter>
	inline void import_from(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, InputIter src)
	{
		copy_n(src, a.nelems(), a.begin());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline void import_from(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T *src)
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


	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RView>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator << (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const RView& b)
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

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline void fill(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& x)
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


	template<typename T, typename TOrd>
	inline void set_zeros(aview2d<T, TOrd, id_ind, id_ind>& a)
	{
		if (is_dense_view(a))
		{
			set_zeros_to_elements(a.pbase(), a.nelems());
		}
		else
		{
			std::fill(a.begin(), a.end(), T(0));
		}
	}


	// stand-alone array class

	template<typename T, typename TOrd, class Alloc>
	class array2d : public aview2d<T, TOrd, id_ind, id_ind>
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T)

		typedef TOrd layout_order;
		typedef id_ind indexer0_type;
		typedef id_ind indexer1_type;
		typedef const_aview2d<value_type, layout_order, indexer0_type, indexer1_type> const_view_type;
		typedef aview2d<value_type, layout_order, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, id_ind, id_ind> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		explicit array2d(size_type m, size_type n)
		: view_type(0, m, n, m, n), m_pblock(new block<value_type>(m * n))
		{
			this->m_base = m_pblock->pbase();
		}

		array2d(size_type m, size_type n, const T& x)
		: view_type(0, m, n, m, n), m_pblock(new block<value_type>(m * n))
		{
			this->m_base = m_pblock->pbase();

			fill(*this, x);
		}

		template<typename InputIter>
		array2d(size_type m, size_type n, InputIter src)
		: view_type(0, m, n, m, n), m_pblock(new block<value_type>(m * n))
		{
			this->m_base = m_pblock->pbase();

			import_from(*this, src);
		}

	private:
		tr1::shared_ptr<block<value_type, Alloc> > m_pblock;

	}; // end class array2d





}

#endif
