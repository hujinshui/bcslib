/**
 * @file array_eval.h
 *
 * Evaluation on arrays
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY_EVAL_H
#define BCSLIB_ARRAY_EVAL_H

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <bcslib/base/arg_check.h>
#include <type_traits>
#include <algorithm>

namespace bcs
{
	template<typename T>
	class scoped_caview : private noncopyable
	{
	public:
		template<typename TOrd>
		explicit scoped_caview(const caview2d<T, TOrd>& a)
		{
			if (a.is_continuous() || a.nelems() == 0)
			{
				m_data = const_cast<T*>(a.pbase());
				m_n = a.nelems();
				m_own = false;
			}
			else
			{
				m_data = aligned_allocate<T>(a.nelems(), BCS_DEFAULT_ALIGNMENT);
				export_to(a, m_data);
				m_n = a.nelems();
				m_own = true;
			}
		}

		~scoped_caview()
		{
			if (m_own)
			{
				aligned_deallocate(m_data);
			}
		}

		const T* begin() const
		{
			return m_data;
		}

		const T* end() const
		{
			return m_data + m_n;
		}

		size_t size() const
		{
			return m_n;
		}

		bool own() const
		{
			return m_own;
		}

	private:
		T *m_data;
		size_t m_n;
		bool m_own;

	};  // end class scoped_caview


	// map

	template<class Func, typename T>
	inline array1d<typename std::result_of<Func(T)>::type > map(Func f, const caview1d<T>& x)
	{
		typedef typename std::result_of<Func(T)>::type result_t;
		array1d<result_t> r(x.nelems());
		std::transform(x.pbase(), x.pbase() + x.nelems(), r.pbase(), f);
		return r;
	}

	template<class Func, typename T1, typename T2>
	inline array1d<typename std::result_of<Func(T1, T2)>::type > map(Func f, const caview1d<T1>& x1, const caview1d<T2>& x2)
	{
		check_arg(x1.nelems() == x2.nelems(), "bcs::map: inconsistent dimensions.");

		typedef typename std::result_of<Func(T1, T2)>::type result_t;
		array1d<result_t> r(x1.nelems());
		std::transform(x1.pbase(), x1.pbase() + x1.nelems(), x2.pbase(), r.pbase(), f);

		return r;
	}

	template<class Func, typename T, typename TOrd>
	inline array2d<typename std::result_of<Func(T)>::type, TOrd> map(Func f, const caview2d<T, TOrd>& x)
	{
		typedef typename std::result_of<Func(T)>::type result_t;
		array2d<result_t, TOrd> r(x.nrows(), x.ncolumns());

		scoped_caview<T> xp(x);
		std::transform(xp.begin(), xp.end(), r.pbase(), f);

		return r;
	}

	template<class Func, typename T1, typename T2, typename TOrd>
	inline array2d<typename std::result_of<Func(T1, T2)>::type, TOrd> map(Func f, const caview2d<T1, TOrd>& x1, const caview2d<T2, TOrd>& x2)
	{
		check_arg(x1.dim0() == x2.dim0() && x1.dim1() == x2.dim1(), "bcs::map: inconsistent dimensions.");

		typedef typename std::result_of<Func(T1, T2)>::type result_t;
		array2d<result_t, TOrd> r(x1.nrows(), x1.ncolumns());

		scoped_caview<T1> x1p(x1);
		scoped_caview<T2> x2p(x2);
		std::transform(x1p.begin(), x1p.end(), x2p.begin(), r.pbase(), f);

		return r;
	}

	// vreduce

	struct per_row { };
	struct per_col { };

	template<class VFunc, typename T>
	inline typename VFunc::result_type vreduce(VFunc vf, const caview1d<T>& x)
	{
		return vf(x.nelems(), x.pbase());
	}

	template<class VFunc, typename T1, typename T2>
	inline typename VFunc::result_type vreduce(VFunc vf, const caview1d<T1>& x1, const caview1d<T2>& x2)
	{
		check_arg(x1.nelems() == x2.nelems(), "bcs::vreduce: inconsistent dimensions.");
		return vf(x1.nelems(), x1.pbase(), x2.pbase());
	}


	template<class VFunc, typename T, typename TOrd>
	inline typename VFunc::result_type vreduce(VFunc vf, const caview2d<T, TOrd>& x)
	{
		scoped_caview<T> xp(x);
		return vf(xp.size(), xp.begin());
	}

	template<class VFunc, typename T1, typename T2, typename TOrd>
	inline typename VFunc::result_type vreduce(VFunc vf, const caview2d<T1, TOrd>& x1, const caview2d<T2, TOrd>& x2)
	{
		check_arg(x1.dim0() == x2.dim0() && x1.dim1() == x2.dim1(), "bcs::vreduce: inconsistent dimensions.");

		scoped_caview<T1> x1p(x1);
		scoped_caview<T2> x2p(x2);

		return vf(x1p.size(), x1p.begin(), x2p.begin());
	}


	template<class VFunc, typename T>
	array1d<typename VFunc::result_type> vreduce(VFunc vf, per_row, const caview2d<T, row_major_t>& x)
	{
		array1d<typename VFunc::result_type> r(x.nrows());

		index_t n = x.dim0();
		for (index_t i = 0; i < n; ++i)
		{
			r(i) = vreduce(vf, x.row(i));
		}
		return r;
	}

	template<class VFunc, typename T1, typename T2>
	array1d<typename VFunc::result_type> vreduce(VFunc vf, per_row, const caview2d<T1, row_major_t>& x1, const caview2d<T2, row_major_t>& x2)
	{
		check_arg(x1.dim0() == x2.dim0() && x1.dim1() == x2.dim1(), "bcs::vreduce: inconsistent dimensions.");

		array1d<typename VFunc::result_type> r(x1.nrows());

		index_t n = x1.dim0();
		for (index_t i = 0; i < n; ++i)
		{
			r(i) = vreduce(vf, x1.row(i), x2.row(i));
		}
		return r;
	}

	template<class VFunc, typename T>
	inline array1d<typename VFunc::result_type> vreduce(VFunc vf, per_col, const caview2d<T, row_major_t>& x)
	{
		return vreduce(vf, per_row(), transpose(x));
	}

	template<class VFunc, typename T1, typename T2>
	inline array1d<typename VFunc::result_type> vreduce(VFunc vf, per_col, const caview2d<T1, row_major_t>& x1, const caview2d<T2, row_major_t>& x2)
	{
		return vreduce(vf, per_row(), transpose(x1), transpose(x2));
	}


	template<class VFunc, typename T>
	array1d<typename VFunc::result_type> vreduce(VFunc vf, per_col, const caview2d<T, column_major_t>& x)
	{
		array1d<typename VFunc::result_type> r(x.ncolumns());

		index_t n = x.dim1();
		for (index_t i = 0; i < n; ++i)
		{
			r(i) = vreduce(vf, x.column(i));
		}
		return r;
	}

	template<class VFunc, typename T1, typename T2>
	array1d<typename VFunc::result_type> vreduce(VFunc vf, per_col, const caview2d<T1, column_major_t>& x1, const caview2d<T2, column_major_t>& x2)
	{
		check_arg(x1.dim0() == x2.dim0() && x1.dim1() == x2.dim1(), "bcs::vreduce: inconsistent dimensions.");

		array1d<typename VFunc::result_type> r(x1.ncolumns());

		index_t n = x1.dim1();
		for (index_t i = 0; i < n; ++i)
		{
			r(i) = vreduce(vf, x1.column(i), x2.column(i));
		}
		return r;
	}

	template<class VFunc, typename T>
	inline array1d<typename VFunc::result_type> vreduce(VFunc vf, per_row, const caview2d<T, column_major_t>& x)
	{
		return vreduce(vf, per_col(), transpose(x));
	}

	template<class VFunc, typename T1, typename T2>
	inline array1d<typename VFunc::result_type> vreduce(VFunc vf, per_row, const caview2d<T1, column_major_t>& x1, const caview2d<T2, column_major_t>& x2)
	{
		return vreduce(vf, per_col(), transpose(x1), transpose(x2));
	}


}

#endif
