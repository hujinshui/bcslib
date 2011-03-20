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
	}



	// iterations

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, bool IsConst> class aview2d_iter_implementer;

	template<typename T, class TIndexer0, class TIndexer1, bool IsConst>
	class aview2d_iter_implementer<T, row_major_t, TIndexer0, TIndexer1, IsConst>
	{
	public:
		typedef row_major_t layout_order;
		typedef aview2d_iter_implementer<T, layout_order, TIndexer0, TIndexer1, IsConst> self_type;

		typedef T value_type;
		typedef typename pointer_and_reference<T, IsConst>::pointer pointer;
		typedef typename pointer_and_reference<T, IsConst>::reference reference;

	public:
		aview2d_iter_implementer()
		: m_pbase(0), m_base_d0(0), m_base_d1(0)
		, m_pidx0(0), m_pidx1(0), m_d0_m1(0), m_d1_m1(0)
		, m_i(0), m_j(0), m_p(0)
		{
		}

		aview2d_iter_implementer(pointer pbase, size_t base_d0, size_t base_d1,
				const TIndexer0& idx0, const TIndexer1& idx1, index_t i, index_t j)
		: m_pbase(pbase), m_base_d0(base_d0), m_base_d1(base_d1)
		, m_pidx0(&idx0), m_pidx1(&idx1), m_d0_m1((index_t)idx0.size()-1), m_d1_m1((index_t)idx1.size()-1)
		, m_i(i), m_j(j), m_p(pbase + _detail::layout_aux2d<layout_order>::offset(base_d0, base_d1, i, j))
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
				m_p += m_pidx1->step_at(m_j);
				++ m_j;
			}
			else
			{
				++ m_i;
				m_j = 0;
				m_p = m_pbase + m_pidx0->operator[](m_i) * m_base_d1;
			}
		}

	private:
		pointer m_pbase;
		size_t m_base_d0;
		size_t m_base_d1;

		const TIndexer0 *m_pidx0;
		const TIndexer1 *m_pidx1;
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
		typedef column_major_t layout_order;
		typedef aview2d_iter_implementer<T, layout_order, TIndexer0, TIndexer1, IsConst> self_type;

		typedef T value_type;
		typedef typename pointer_and_reference<T, IsConst>::pointer pointer;
		typedef typename pointer_and_reference<T, IsConst>::reference reference;

	public:
		aview2d_iter_implementer()
		: m_pbase(0), m_base_d0(0), m_base_d1(0)
		, m_pidx0(0), m_pidx1(0), m_d0_m1(0), m_d1_m1(0)
		, m_i(0), m_j(0), m_p(0)
		{
		}

		aview2d_iter_implementer(pointer pbase, size_t base_d0, size_t base_d1,
				const TIndexer0& idx0, const TIndexer1& idx1, index_t i, index_t j)
		: m_pbase(pbase), m_base_d0(base_d0), m_base_d1(base_d1)
		, m_pidx0(&idx0), m_pidx1(&idx1), m_d0_m1((index_t)idx0.size()-1), m_d1_m1((index_t)idx1.size()-1)
		, m_i(i), m_j(j), m_p(pbase + _detail::layout_aux2d<layout_order>::offset(base_d0, base_d1, i, j))
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
				m_p += m_pidx0->step_at(m_i);
				++ m_i;
			}
			else
			{
				++ m_j;
				m_i = 0;
				m_p = m_pbase + m_pidx1->operator[](m_j) * m_base_d0;
			}
		}

	private:
		pointer m_pbase;
		size_t m_base_d0;
		size_t m_base_d1;

		const TIndexer0 *m_pidx0;
		const TIndexer1 *m_pidx1;
		index_t m_d0_m1;
		index_t m_d1_m1;

		index_t m_i;
		index_t m_j;
		pointer m_p;

	}; // end class aview2d_iter_implementer for column_major




	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct aview2d_iterators
	{
		typedef forward_iterator_wrapper<aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, true> > const_iterator;
		typedef forward_iterator_wrapper<aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, false> > iterator;

		static const_iterator get_const_iterator(const T *pbase, size_t base_d0, size_t base_d1,
				const TIndexer0& indexer0, const TIndexer1& indexer1, index_t i, index_t j)
		{
			typedef aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, true> impl;
			return impl(pbase, base_d0, base_d1, indexer0, indexer1, i, j);
		}

		static iterator get_iterator(T *pbase, size_t base_d0, size_t base_d1,
				const TIndexer0& indexer0, const TIndexer1& indexer1, index_t i, index_t j)
		{
			typedef aview2d_iter_implementer<T, TOrd, TIndexer0, TIndexer1, false> impl;
			return impl(pbase, base_d0, base_d1, indexer0, indexer1, i, j);
		}
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
			return _detail::layout_aux2d<layout_order>::offset(base_dim0(), base_dim1(), i, j);
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
			return _iterators::get_const_iterator(m_base, m_base_d0, m_base_d1, m_indexer0, m_indexer1, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(dim0(), dim1(), e_i, e_j);
			return _iterators::get_const_iterator(m_base, m_base_d0, m_base_d1, m_indexer0, m_indexer1, e_i, e_j);
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
			return _iterators::get_const_iterator(this->m_base, this->m_base_d0, this->m_base_d1,
					this->m_indexer0, this->m_indexer1, 0, 0);
		}

		iterator begin()
		{
			return _iterators::get_iterator(this->m_base, this->m_base_d0, this->m_base_d1,
					this->m_indexer0, this->m_indexer1, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(this->dim0(), this->dim1(), e_i, e_j);
			return _iterators::get_const_iterator(this->m_base, this->m_base_d0, this->m_base_d1,
					this->m_indexer0, this->m_indexer1, e_i, e_j);
		}

		iterator end()
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(this->dim0(), this->dim1(), e_i, e_j);
			return _iterators::get_iterator(this->m_base, this->m_base_d0, this->m_base_d1,
					this->m_indexer0, this->m_indexer1, e_i, e_j);
		}

	}; // end class aview2d


	// functions to make dense view

	template<typename T, typename TOrd>
	const_aview2d<T, TOrd, id_ind, id_ind> dense_const_aview2d(const T *pbase, size_t m, size_t n)
	{
		return const_aview2d<T, TOrd, id_ind, id_ind>(pbase, m, n, m, n);
	}

	template<typename T, typename TOrd>
	aview2d<T, TOrd, id_ind, id_ind> dense_const_aview2d(T *pbase, size_t m, size_t n)
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
