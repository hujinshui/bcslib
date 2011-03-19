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
				i = (index_t)m + 1;
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
				j = (index_t)n + 1;
			}
		};
	}



	// iterations

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1> struct aview2d_iterators;

	// main classes

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class const_aview2d
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T)

		typedef TOrd layout_order;
		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef const_aview2d<value_type, indexer0_type, indexer1_type> const_view_type;
		typedef aview2d<value_type, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, TIndexer0, TIndexer1> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		const_aview2d(const_pointer base, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_base(const_cast<pointer>(base))
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

		// Element access

		const_pointer pbase() const
		{
			return m_base;
		}

		index_t offset_at(index_t i, index_t j) const
		{
			return _detail::layout_aux2d<layout_order>::offset(dim0(), dim1(), i, j);
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
			return _iterators::get_const_iterator(m_base, m_indexer0, m_indexer1, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(dim0(), dim1(), e_i, e_j);
			return _iterators::get_const_iterator(m_base, m_indexer0, m_indexer1, e_i, e_j);
		}

	protected:
		pointer m_base;
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
		typedef const_aview2d<value_type, indexer0_type, indexer1_type> const_view_type;
		typedef aview2d<value_type, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, TIndexer0, TIndexer1> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		aview2d(const_pointer base, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: const_view_type(base, indexer0, indexer1)
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
			return _iterators::get_const_iterator(this->m_base, this->m_indexer0, this->m_indexer1, 0, 0);
		}

		const_iterator end() const
		{
			index_t e_i, e_j;
			_detail::layout_aux2d<layout_order>::pass_by_end(this->dim0(), this->dim1(), e_i, e_j);
			return _iterators::get_const_iterator(this->m_base, this->m_indexer0, this->m_indexer1, e_i, e_j);
		}

	protected:
		pointer m_base;
		size_type m_ne;
		indexer0_type m_indexer0;
		indexer1_type m_indexer1;

	}; // end class aview2d


	// stand-alone array class

	template<typename T, typename TOrd, class Alloc>
	class array2d : public aview2d<T, TOrd, id_ind, id_ind>
	{
	public:
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T)

		typedef TOrd layout_order;
		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef const_aview2d<value_type, indexer0_type, indexer1_type> const_view_type;
		typedef aview2d<value_type, indexer0_type, indexer1_type> view_type;

		typedef aview2d_iterators<T, TOrd, TIndexer0, TIndexer1> _iterators;
		typedef typename _iterators::const_iterator const_iterator;
		typedef typename _iterators::iterator iterator;

	public:
		explicit array2d(size_type m, size_type n)
		: view_type(0, m, n), m_pblock(new block<value_type>(n))
		{
			this->m_base = m_pblock->pbase();
		}

		array2d(size_type m, size_type n, const T& x)
		: view_type(0, m, n), m_pblock(new block<value_type>(n))
		{
			this->m_base = m_pblock->pbase();

			fill(*this, x);
		}

		template<typename InputIter>
		array2d(size_type m, size_type n, InputIter src)
		: view_type(0, m, n), m_pblock(new block<value_type>(n))
		{
			this->m_base = m_pblock->pbase();

			import_from(*this, src);
		}

	private:
		tr1::shared_ptr<block<value_type, Alloc> > m_pblock;

	}; // end class array2d

}

#endif
