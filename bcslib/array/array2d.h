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
#include <bcslib/array/generic_array_transpose.h>

namespace bcs
{
	// forward declaration

	template<typename T, typename TOrd, class TIndexer0=id_ind, class TIndexer1=id_ind> class aview2d;
	template<typename T, typename TOrd, class Alloc=aligned_allocator<T> > class array2d;


	// order-specific offset calculation

	namespace _detail
	{

		template<typename TOrd, class TIndexer0, class TIndexer1> struct array2d_slices;

		template<class TIndexer0, class TIndexer1>
		struct array2d_slices<row_major_t, TIndexer0, TIndexer1>
		{
			typedef TIndexer1 slice0_indexer_type;
			typedef typename inject_step<TIndexer0>::type slice1_indexer_type;

			static slice0_indexer_type slice0_indexer(index_t base_d0, index_t base_d1, const TIndexer1& idx1)
			{
				return idx1;
			}

			static index_t slice0_offset(index_t base_d0, index_t base_d1, const TIndexer0& idx0, index_t i)
			{
				return idx0[i] * base_d1;
			}

			static slice1_indexer_type slice1_indexer(index_t base_d0, index_t base_d1, const TIndexer0& idx0)
			{
				return inject_step<TIndexer0>::get(idx0, base_d1);
			}

			static index_t slice1_offset(index_t base_d0, index_t base_d1, const TIndexer1& idx1, index_t j)
			{
				return idx1[j];
			}

		};


		template<class TIndexer0, class TIndexer1>
		struct array2d_slices<column_major_t, TIndexer0, TIndexer1>
		{
			typedef typename inject_step<TIndexer1>::type slice0_indexer_type;
			typedef TIndexer0 slice1_indexer_type;

			static slice0_indexer_type slice0_indexer(index_t base_d0, index_t base_d1, const TIndexer1& idx1)
			{
				return inject_step<TIndexer1>::get(idx1, base_d0);
			}

			static index_t slice0_offset(index_t base_d0, index_t base_d1, const TIndexer0& idx0, index_t i)
			{
				return idx0[i];
			}

			static slice1_indexer_type slice1_indexer(index_t base_d0, index_t base_d1, const TIndexer0& idx0)
			{
				return idx0;
			}

			static index_t slice1_offset(index_t base_d0, index_t base_d1, const TIndexer1& idx1, index_t j)
			{
				return idx1[j] * base_d0;
			}
		};


		// 2D indexing core

		template<typename TOrd, class TIndexer0, class TIndexer1>
		class _aview2d_index_core
		{
		public:
			typedef _detail::array2d_slices<TOrd, TIndexer0, TIndexer1> _slices;

			typedef typename _slices::slice0_indexer_type slice0_indexer_type;
			typedef typename _slices::slice1_indexer_type slice1_indexer_type;

		public:
			_aview2d_index_core(const index_t& base_d0, const index_t& base_d1, const TIndexer0& ind0, const TIndexer1& ind1)
			: m_base_d0(base_d0), m_base_d1(base_d1)
			, m_d0(static_cast<index_t>(ind0.size()))
			, m_d1(static_cast<index_t>(ind1.size()))
			, m_ne(ind0.size() * ind1.size())
			, m_indexer0(ind0), m_indexer1(ind1)
			{
			}

			_aview2d_index_core(const index_t& base_d0, const index_t& base_d1, TIndexer0&& ind0, TIndexer1&& ind1)
			: m_base_d0(base_d0), m_base_d1(base_d1)
			, m_d0(static_cast<index_t>(ind0.size()))
			, m_d1(static_cast<index_t>(ind1.size()))
			, m_ne(ind0.size() * ind1.size())
			, m_indexer0(std::move(ind0)), m_indexer1(std::move(ind1))
			{
			}

			_aview2d_index_core(const _aview2d_index_core& r)
			: m_base_d0(r.m_base_d0), m_base_d1(r.m_base_d1), m_d0(r.m_d0), m_d1(r.m_d1), m_ne(r.m_ne)
			, m_indexer0(r.m_indexer0), m_indexer1(r.m_indexer1)
			{
			}

			_aview2d_index_core(_aview2d_index_core&& r)
			: m_base_d0(r.m_base_d0), m_base_d1(r.m_base_d1), m_d0(r.m_d0), m_d1(r.m_d1), m_ne(r.m_ne)
			, m_indexer0(std::move(r.m_indexer0)), m_indexer1(std::move(r.m_indexer1))
			{
				r.reset();
			}

			void operator = (const _aview2d_index_core& r)
			{
				m_base_d0 = r.m_base_d0;
				m_base_d1 = r.m_base_d1;
				m_d0 = r.m_d0;
				m_d1 = r.m_d1;
				m_indexer0 = r.m_indexer0;
				m_indexer1 = r.m_indexer1;
			}

			void operator = (_aview2d_index_core&& r)
			{
				m_base_d0 = r.m_base_d0;
				m_base_d1 = r.m_base_d1;
				m_d0 = r.m_d0;
				m_d1 = r.m_d1;

				m_indexer0 = std::move(r.m_indexer0);
				m_indexer1 = std::move(r.m_indexer1);

				r.reset();
			}

		public:
			index_t base_dim0() const
			{
				return m_base_d0;
			}

			index_t base_dim1() const
			{
				return m_base_d1;
			}

			index_t dim0() const
			{
				return m_d0;
			}

			index_t dim1() const
			{
				return m_d1;
			}

			size_t nelems() const
			{
				return m_ne;
			}

			index_t offset(index_t i, index_t j) const
			{
				return layout_aux2d<TOrd>::offset(m_base_d0, m_base_d1, m_indexer0[i], m_indexer1[j]);
			}

			const TIndexer0& get_indexer0() const
			{
				return m_indexer0;
			}

			const TIndexer1& get_indexer1() const
			{
				return m_indexer1;
			}

			void get_pass_by_end_indices(index_t& e_i, index_t& e_j) const
			{
				layout_aux2d<TOrd>::pass_by_end(m_d0, m_d1, e_i, e_j);
			}

		public:
			index_t sliceI0_offset(const index_t& i) const
			{
				return _slices::slice0_offset(m_base_d0, m_base_d1, m_indexer0, i);
			}

			index_t sliceI1_offset(const index_t& j) const
			{
				return _slices::slice1_offset(m_base_d0, m_base_d1, m_indexer1, j);
			}

			slice0_indexer_type sliceI0_indexer() const
			{
				return _slices::slice0_indexer(m_base_d0, m_base_d1, m_indexer1);
			}

			slice1_indexer_type sliceI1_indexer() const
			{
				return _slices::slice1_indexer(m_base_d0, m_base_d1, m_indexer0);
			}

			template<class TSelector0, class TSelector1>
			_aview2d_index_core<TOrd,
				typename sub_indexer<TIndexer0, TSelector0>::type,
				typename sub_indexer<TIndexer1, TSelector1>::type>
			subview_index_core(const TSelector0& sel0, const TSelector1& sel1, index_t& offset) const
			{
				typedef typename sub_indexer<TIndexer0, TSelector0>::type sub_indexer0_t;
				typedef typename sub_indexer<TIndexer1, TSelector1>::type sub_indexer1_t;
				typedef _aview2d_index_core<TOrd, sub_indexer0_t, sub_indexer1_t> result_type;

				index_t o0, o1;
				sub_indexer0_t si0 = sub_indexer<TIndexer0, TSelector0>::get(get_indexer0(), sel0, o0);
				sub_indexer1_t si1 = sub_indexer<TIndexer1, TSelector1>::get(get_indexer1(), sel1, o1);

				offset = layout_aux2d<TOrd>::offset(m_base_d0, m_base_d1, o0, o1);
				return result_type(m_base_d0, m_base_d1, std::move(si0), std::move(si1));
			}

		private:
			void reset()
			{
				m_base_d0 = 0;
				m_base_d1 = 0;
				m_d0 = 0;
				m_d1 = 0;
				m_ne = 0;
			}

		private:
			index_t m_base_d0;
			index_t m_base_d1;
			index_t m_d0;
			index_t m_d1;
			size_t m_ne;
			TIndexer0 m_indexer0;
			TIndexer1 m_indexer1;

		}; // end class _aview2d_index_core


		// iteration implementation

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1> class _aview2d_iter_impl;
		// T can be const or non-const here

		template<typename T, class TIndexer0, class TIndexer1>
		class _aview2d_iter_impl<T, row_major_t, TIndexer0, TIndexer1>
		{
		public:
			typedef T value_type;
			typedef value_type* pointer;
			typedef value_type& reference;
			typedef row_major_t layout_order;

			typedef _aview2d_index_core<layout_order, TIndexer0, TIndexer1> index_core_t;

		public:
			_aview2d_iter_impl()
			: m_pbase(BCS_NULL), m_p_index_core(BCS_NULL), m_j_ub(0), m_i(0), m_j(0), m_p(BCS_NULL)
			{
			}

			_aview2d_iter_impl(pointer pbase, const index_core_t& idxcore, index_t i, index_t j)
			: m_pbase(pbase), m_p_index_core(&idxcore), m_j_ub(idxcore.dim1() - 1), m_i(i), m_j(j)
			, m_p(m_pbase + idxcore.offset(i, j))
			{
			}

			pointer ptr() const { return m_p; }

			reference ref() const { return *m_p; }

			bool operator == (const _aview2d_iter_impl& rhs) const { return m_i == rhs.m_i && m_j == rhs.m_j; }

			void move_next()
			{
				if (m_j < m_j_ub)
				{
					++ m_j;
				}
				else
				{
					++ m_i;
					m_j = 0;
				}
				m_p = m_pbase + m_p_index_core->offset(m_i, m_j);
			}

		private:
			pointer m_pbase;
			const index_core_t* m_p_index_core;
			index_t m_j_ub;

			index_t m_i;
			index_t m_j;
			pointer m_p;

		}; // end class _aview2d_iter_impl for row_major


		template<typename T, class TIndexer0, class TIndexer1>
		class _aview2d_iter_impl<T, column_major_t, TIndexer0, TIndexer1>
		{
		public:
			typedef T value_type;
			typedef value_type* pointer;
			typedef value_type& reference;
			typedef column_major_t layout_order;

			typedef _aview2d_index_core<layout_order, TIndexer0, TIndexer1> index_core_t;

		public:
			_aview2d_iter_impl()
			: m_pbase(BCS_NULL), m_p_index_core(BCS_NULL), m_i_ub(0), m_i(0), m_j(0), m_p(BCS_NULL)
			{
			}

			_aview2d_iter_impl(pointer pbase, const index_core_t& idxcore, index_t i, index_t j)
			: m_pbase(pbase), m_p_index_core(&idxcore), m_i_ub(idxcore.dim0() - 1), m_i(i), m_j(j)
			, m_p(m_pbase + idxcore.offset(i, j))
			{
			}

			pointer ptr() const { return m_p; }

			reference ref() const { return *m_p; }

			bool operator == (const _aview2d_iter_impl& rhs) const { return m_j == rhs.m_j && m_i == rhs.m_i; }

			void move_next()
			{
				if (m_i < m_i_ub)
				{
					++ m_i;
				}
				else
				{
					++ m_j;
					m_i = 0;
				}
				m_p = m_pbase + m_p_index_core->offset(m_i, m_j);
			}

		private:
			pointer m_pbase;
			const index_core_t* m_p_index_core;
			index_t m_i_ub;

			index_t m_i;
			index_t m_j;
			pointer m_p;

		}; // end class _aview2d_iter_imple for column_major


		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		class _arr2d_transpose_t
		{
		public:
			typedef T value_type;
			typedef aview2d<T, TOrd, TIndexer0, TIndexer1> aview_type;

			explicit _arr2d_transpose_t(const aview_type& a) : m_cref(a) { }

			_arr2d_transpose_t(const _arr2d_transpose_t& r) : m_cref(r.m_cref) { }

			const aview_type& get() const
			{
				return m_cref;
			}

			inline operator array2d<T, TOrd>();

		private:
			const aview_type& m_cref;
		};


	} // end namespace _detail



	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct aview2d_iterators
	{
		typedef typename std::remove_const<T>::type value_type;
		typedef _detail::_aview2d_iter_impl<const value_type, TOrd, TIndexer0, TIndexer1> const_iter_impl;
		typedef _detail::_aview2d_iter_impl<value_type, TOrd, TIndexer0, TIndexer1> iter_impl;

		typedef forward_iterator_wrapper<const_iter_impl> const_iterator;
		typedef forward_iterator_wrapper<iter_impl> iterator;
		typedef _detail::_aview2d_index_core<TOrd, TIndexer0, TIndexer1> _index_core_t;

		static const_iterator get_const_iterator(const T* pbase, const _index_core_t& idxcore, index_t i, index_t j)
		{
			return const_iter_impl(pbase, idxcore, i, j);
		}

		static iterator get_iterator(T* pbase, const _index_core_t& idxcore, index_t i, index_t j)
		{
			return iter_impl(pbase, idxcore, i, j);
		}
	};


	// main classes

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class aview2d
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef aview2d<value_type, layout_order, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, TIndexer0, TIndexer1> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

		typedef _detail::_aview2d_index_core<TOrd, TIndexer0, TIndexer1> _index_core_type;
		typedef typename _index_core_type::slice0_indexer_type slice0_indexer_type;
		typedef typename _index_core_type::slice1_indexer_type slice1_indexer_type;

	private:
		typedef _detail::_arr2d_transpose_t<T, TOrd, TIndexer0, TIndexer1> _transpose_t;

	public:
		aview2d(pointer pbase, index_type base_d0, index_type base_d1,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(pbase), m_idxcore(base_d0, base_d1, indexer0, indexer1)
		{
		}

		aview2d(const aview2d& r)
		: m_pbase(r.m_pbase), m_idxcore(r.m_idxcore)
		{
		}

		aview2d(aview2d&& r)
		: m_pbase(r.m_pbase), m_idxcore(std::move(r.m_idxcore))
		{
			r.m_pbase = BCS_NULL;
		}

		aview2d& operator = (const aview2d& r)
		{
			if (this != &r)
			{
				m_pbase = r.m_pbase;
				m_idxcore = r.m_idxcore;
			}
			return *this;
		}

		aview2d& operator = (aview2d&& r)
		{
			m_pbase = r.m_pbase;
			m_idxcore = std::move(r.m_idxcore);
			r.m_pbase = BCS_NULL;
			return *this;
		}

	public:
		aview2d(pointer pbase, const _index_core_type& idxcore)
		: m_pbase(pbase), m_idxcore(idxcore)
		{
		}

		const _index_core_type& _index_core() const
		{
			return m_idxcore;
		}

	public:
		dim_num_t ndims() const
		{
			return num_dims;
		}

		size_type nelems() const
		{
			return m_idxcore.nelems();
		}

		index_type dim0() const
		{
			return m_idxcore.dim0();
		}

		index_type dim1() const
		{
			return m_idxcore.dim1();
		}

		size_type nrows() const
		{
			return static_cast<size_type>(dim0());
		}

		size_type ncolumns() const
		{
			return static_cast<size_type>(dim1());
		}

		shape_type shape() const
		{
			return arr_shape(dim0(), dim1());
		}

		const indexer0_type& get_indexer0() const
		{
			return m_idxcore.get_indexer0();
		}

		const indexer1_type& get_indexer1() const
		{
			return m_idxcore.get_indexer1();
		}

		index_type base_dim0() const
		{
			return m_idxcore.base_dim0();
		}

		index_type base_dim1() const
		{
			return m_idxcore.base_dim1();
		}

		shape_type base_shape() const
		{
			return arr_shape(base_dim0(), base_dim1());
		}

		// just for temporary use in an expression, not allowed to pass around
		_transpose_t Tp() const
		{
			return _transpose_t(*this);
		}

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

		const_pointer ptr(index_type i, index_type j) const
		{
			return m_pbase + m_idxcore.offset(i, j);
		}

		pointer ptr(index_type i, index_type j)
		{
			return m_pbase + m_idxcore.offset(i, j);
		}

		const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[m_idxcore.offset(i, j)];
		}

		reference operator() (index_type i, index_type j)
		{
			return m_pbase[m_idxcore.offset(i, j)];
		}

		// Iteration

		const_iterator begin() const
		{
			return _iterators::get_const_iterator(pbase(), m_idxcore, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			m_idxcore.get_pass_by_end_indices(e_i, e_j);
			return _iterators::get_const_iterator(pbase(), m_idxcore, e_i, e_j);
		}

		iterator begin()
		{
			return _iterators::get_iterator(pbase(), m_idxcore, 0, 0);
		}

		iterator end()
		{
			index_t e_i, e_j;
			m_idxcore.get_pass_by_end_indices(e_i, e_j);
			return _iterators::get_iterator(pbase(), m_idxcore, e_i, e_j);
		}


		// Slice

		const aview1d<value_type, slice0_indexer_type> sliceI0(index_type i) const
		{
			return aview1d<value_type, slice0_indexer_type>(
					m_pbase + m_idxcore.sliceI0_offset(i), m_idxcore.sliceI0_indexer());
		}

		const aview1d<value_type, slice1_indexer_type> sliceI1(index_type j) const
		{
			return aview1d<value_type, slice1_indexer_type>(
					m_pbase + m_idxcore.sliceI1_offset(j), m_idxcore.sliceI1_indexer());
		}

		aview1d<value_type, slice0_indexer_type> sliceI0(index_type i)
		{
			return aview1d<value_type, slice0_indexer_type>(
					m_pbase + m_idxcore.sliceI0_offset(i), m_idxcore.sliceI0_indexer());
		}

		aview1d<value_type, slice1_indexer_type> sliceI1(index_type j)
		{
			return aview1d<value_type, slice1_indexer_type>(
					m_pbase + m_idxcore.sliceI1_offset(j), m_idxcore.sliceI1_indexer());
		}

		const aview1d<value_type, slice0_indexer_type> row(index_type i) const
		{
			return sliceI0(i);
		}

		aview1d<value_type, slice0_indexer_type> row(index_type i)
		{
			return sliceI0(i);
		}

		const aview1d<value_type, slice1_indexer_type> column(index_type j) const
		{
			return sliceI1(j);
		}

		aview1d<value_type, slice1_indexer_type> column(index_type j)
		{
			return sliceI1(j);
		}

		// Sub-view

		template<class TSelector>
		const aview1d<value_type, typename sub_indexer<slice0_indexer_type, TSelector>::type>
		V(index_type i, const TSelector& sel) const
		{
			return sliceI0(i).V(sel);
		}

		template<class TSelector>
		aview1d<value_type, typename sub_indexer<slice0_indexer_type, TSelector>::type>
		V(index_type i, const TSelector& sel)
		{
			return sliceI0(i).V(sel);
		}

		template<class TSelector>
		const aview1d<value_type, typename sub_indexer<slice1_indexer_type, TSelector>::type>
		V(const TSelector& sel, index_type j) const
		{
			return sliceI1(j).V(sel);
		}

		template<class TSelector>
		aview1d<value_type, typename sub_indexer<slice1_indexer_type, TSelector>::type>
		V(const TSelector& sel, index_type j)
		{
			return sliceI1(j).V(sel);
		}


		template<class TSelector0, class TSelector1>
		const aview2d<value_type, layout_order,
			typename sub_indexer<indexer0_type, TSelector0>::type,
			typename sub_indexer<indexer1_type, TSelector1>::type>
		V(const TSelector0& sel0, const TSelector1& sel1) const
		{
			typedef typename sub_indexer<indexer0_type, TSelector0>::type sub_indexer0_t;
			typedef typename sub_indexer<indexer1_type, TSelector1>::type sub_indexer1_t;

			index_t offset;
			auto sub_idxcore = m_idxcore.subview_index_core(sel0, sel1, offset);

			return aview2d<value_type, layout_order, sub_indexer0_t, sub_indexer1_t>(
					m_pbase + offset, std::move(sub_idxcore));
		}


		template<class TSelector0, class TSelector1>
		aview2d<value_type, layout_order,
			typename sub_indexer<indexer0_type, TSelector0>::type,
			typename sub_indexer<indexer1_type, TSelector1>::type>
		V(const TSelector0& sel0, const TSelector1& sel1)
		{
			typedef typename sub_indexer<indexer0_type, TSelector0>::type sub_indexer0_t;
			typedef typename sub_indexer<indexer1_type, TSelector1>::type sub_indexer1_t;

			index_t offset;
			auto sub_idxcore = m_idxcore.subview_index_core(sel0, sel1, offset);

			return aview2d<value_type, layout_order, sub_indexer0_t, sub_indexer1_t>(
					m_pbase + offset, std::move(sub_idxcore));
		}

	protected:
		pointer m_pbase;
		_index_core_type m_idxcore;

	}; // end class aview2d


	template<typename T>
	inline const aview2d<T, row_major_t> get_aview2d_rm(const T *base, size_t m, size_t n)
	{
		return aview2d<T, row_major_t>(const_cast<T*>(base), (index_t)m, (index_t)n, m, n);
	}

	template<typename T>
	inline const aview2d<T, column_major_t> get_aview2d_cm(const T *base, size_t m, size_t n)
	{
		return aview2d<T, column_major_t>(const_cast<T*>(base), (index_t)m, (index_t)n, m, n);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline const aview2d<T, row_major_t, TIndexer0, TIndexer1> get_aview2d_rm_ex(const T *base, size_t base_m, size_t base_n,
			const TIndexer0& idxer0, const TIndexer1& idxer1)
	{
		return aview2d<T, row_major_t, TIndexer0, TIndexer1>(const_cast<T*>(base), (index_t)base_m, (index_t)base_n, idxer0, idxer1);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline const aview2d<T, column_major_t, TIndexer0, TIndexer1> get_aview2d_cm_ex(const T *base, size_t base_m, size_t base_n,
			const TIndexer0& idxer0, const TIndexer1& idxer1)
	{
		return aview2d<T, column_major_t, TIndexer0, TIndexer1>(const_cast<T*>(base), (index_t)base_m, (index_t)base_n, idxer0, idxer1);
	}


	// stand-alone array2d

	template<typename T, typename TOrd, class Alloc>
	class array2d : private sharable_storage_base<T, Alloc>, public aview2d<T, TOrd, id_ind, id_ind>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)
		static const bool is_readable = true;
		static const bool is_writable = true;

		typedef id_ind indexer0_type;
		typedef id_ind indexer1_type;
		typedef aview2d<value_type, layout_order, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, id_ind, id_ind> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

		typedef sharable_storage_base<T, Alloc> storage_base_type;

	public:
		explicit array2d(size_type m, size_type n)
		: storage_base_type(m * n)
		, view_type(storage_base_type::pointer_to_base(), static_cast<index_t>(m), static_cast<index_t>(n), m, n)
		{
		}

		array2d(size_type m, size_type n, const T& x)
		: storage_base_type(m * n, x)
		, view_type(storage_base_type::pointer_to_base(), static_cast<index_t>(m), static_cast<index_t>(n), m, n)
		{
		}

		array2d(size_type m, size_type n, const_pointer src)
		: storage_base_type(m * n, src)
		, view_type(storage_base_type::pointer_to_base(), static_cast<index_t>(m), static_cast<index_t>(n), m, n)
		{
		}

		array2d(const array2d& r)
		: storage_base_type(r), view_type(storage_base_type::pointer_to_base(), r._index_core())
		{
		}

		array2d(array2d&& r)
		: storage_base_type(std::move(r)), view_type(std::move(r))
		{
		}

		template<typename ForwardIterator>
		array2d(size_type m, size_type n, ForwardIterator it)
		: storage_base_type(m * n)
		, view_type(storage_base_type::pointer_to_base(), static_cast<index_t>(m), static_cast<index_t>(n), m, n)
		{
			import_from(*this, it);
		}


		array2d& operator = (const array2d& r)
		{
			if (this != &r)
			{
				storage_base_type &s = *this;
				view_type& v = *this;

				s = r;
				v = view_type(s.pointer_to_base(), r._index_core());
			}
			return *this;
		}

		array2d& operator = (array2d&& r)
		{
			storage_base_type &s = *this;
			view_type& v = *this;

			s = std::move(r);
			v = view_type(s.pointer_to_base(), std::move(r._index_core()));

			return *this;
		}

		void swap(array2d& r)
		{
			using std::swap;

			storage_base_type::swap(r);

			view_type& v = *this;
			view_type& rv = r;
			swap(v, rv);
		}

		// sharing

	public:
		array2d(const array2d& r, do_share ds)
		: storage_base_type(r, ds)
		, view_type(storage_base_type::pointer_to_base(), r._index_core())
		{
		}

		array2d shared_copy() const
		{
			return array2d(*this, do_share());
		}

	}; // end class array2d

	template<typename T, typename TOrd, class Alloc>
	inline void swap(array2d<T, TOrd, Alloc>& lhs, array2d<T, TOrd, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline _detail::_arr2d_transpose_t<T, TOrd, TIndexer0, TIndexer1>::operator array2d<T, TOrd>()
	{
		return transpose(m_cref);
	}


	/********************************************
	 *
	 *   Concept-required interfaces
	 *
	 ********************************************/

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct is_array_view<bcs::aview2d<T, TOrd, TIndexer0, TIndexer1> > { static const bool value = true; };

	template<typename T, typename TOrd, class Alloc>
	struct is_array_view<bcs::array2d<T, TOrd, Alloc> > { static const bool value = true; };

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct is_array_view_ndim<bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>, 2> { static const bool value = true; };

	template<typename T, typename TOrd, class Alloc>
	struct is_array_view_ndim<bcs::array2d<T, TOrd, Alloc>, 2> { static const bool value = true; };


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline std::array<index_t, 2> get_array_shape(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.shape();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline size_t get_num_elems(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.nelems();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline const T& get(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr, index_t i, index_t j)
	{
		return arr(i, j);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline void set(const T& v, bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr, index_t i, index_t j)
	{
		arr(i, j) = v;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline typename bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>::const_iterator
	begin(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.begin();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline typename bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>::const_iterator
	end(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.end();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline typename bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>::iterator
	begin(bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.begin();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline typename bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>::iterator
	end(bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.end();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline const T* ptr_base(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.pbase();
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline T* ptr_base(bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& arr)
	{
		return arr.pbase();
	}


	template<typename T, class TIndexer0, class TIndexer1>
	inline bool is_dense_view(const aview2d<T, row_major_t, TIndexer0, TIndexer1>& view)
	{
		return (array_indexer_traits<TIndexer1>::is_continuous(view.get_indexer1())  // each row is continuous
				|| view.dim1() == 1)
				&& (
				( array_indexer_traits<TIndexer0>::is_continuous(view.get_indexer0()) &&  // rows are consecutive
				  view.dim1() == view.base_dim1() )
				|| view.dim0() == 1);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline bool is_dense_view(const aview2d<T, column_major_t, TIndexer0, TIndexer1>& view)
	{
		return (array_indexer_traits<TIndexer0>::is_continuous(view.get_indexer0()) // each column is continuous
				|| view.dim0() == 1)
				&& (
				( array_indexer_traits<TIndexer1>::is_continuous(view.get_indexer1()) && // columns are consecutive
				  view.dim0() == view.base_dim0() )
				|| view.dim1() == 1);
	}


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> clone_array(const aview2d<T, TOrd, TIndexer0, TIndexer1>& view)
	{
		array2d<T, TOrd> arr(view.nrows(), view.ncolumns());
		if (is_dense_view(view))
		{
			import_from(arr, view.pbase());
		}
		else
		{
			import_from(arr, view.begin());
		}
		return arr;
	}

	/******************************************************
	 *
	 *  Overloaded operators
	 *
	 ******************************************************/

	// element-wise comparison

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline bool operator == (
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return equal_array(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline bool operator != (
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return !(lhs == rhs);
	}

	// export & import

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline const aview2d<T, TOrd, LIndexer0, LIndexer1>&
	operator >> (const aview2d<T, TOrd, LIndexer0, LIndexer1>& a, aview2d<T, TOrd, RIndexer0, RIndexer1>& b)
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

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>&
	operator << (aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const aview2d<T, TOrd, RIndexer0, RIndexer1>& b)
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


	/******************************************************
	 *
	 *  Creaters
	 *
	 ******************************************************/

	namespace _detail
	{
		template<typename T, typename TOrd>
		struct _array_creater<T, 2, TOrd>
		{
			typedef array2d<T, TOrd> result_type;
			typedef std::array<index_t, 2> shape_type;

			static result_type create(const shape_type& shape)
			{
				size_t m = static_cast<size_t>(shape[0]);
				size_t n = static_cast<size_t>(shape[1]);

				return result_type(m, n);
			}
		};
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct array_creater<aview2d<T, TOrd, TIndexer0, TIndexer1> > : public _detail::_array_creater<T, 2, TOrd>
	{
		typedef array2d<T, TOrd> result_type;

		template<typename U=T, dim_num_t D=2>
		struct remap
		{
			typedef _detail::_array_creater<U, D, TOrd> type;
		};
	};


}


#endif

