/**
 * @file array_eval.h
 *
 * Evaluation of stats, norms, or other quantities on arrays
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_EVAL_H
#define BCSLIB_ARRAY_EVAL_H

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <bcslib/veccomp/vecstat.h>
#include <bcslib/veccomp/vecnorm.h>


namespace bcs
{
	/***
	 * The Concept of a vector Evaluator class
	 * -------
	 *
	 * - result_type (typedef)
	 * - result_type operator() (size_t n, const T *x) const
	 *
	 */


	template<typename T, class TIndexer, typename Evaluator>
	typename Evaluator::result_type array_eval(const const_aview1d<T, TIndexer>& x, Evaluator evaluator)
	{
		if (is_dense_view(x))
		{
			return evaluator(x.nelems(), x.pbase());
		}
		else
		{
			array1d<T> xc = make_copy(x);
			return evaluator(xc.nelems(), xc.pbase());
		}
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Evaluator>
	typename Evaluator::result_type array_eval(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, Evaluator evaluator)
	{
		if (is_dense_view(x))
		{
			return evaluator(x.nelems(), x.pbase());
		}
		else
		{
			array2d<T, TOrd> xc = make_copy(x);
			return evaluator(xc.nelems(), xc.pbase());
		}
	}


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Evaluator>
	array1d<typename Evaluator::result_type> array_rows_eval(
			const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, Evaluator evaluator)
	{
		typedef typename Evaluator::result_type result_t;

		array1d<result_t> r(x.nrows());

		index_t m = (index_t)x.nrows();
		for (index_t i = 0; i < m; ++i)
		{
			r(i) = array_eval(x.row(i), evaluator);
		}

		return r;
	}


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Evaluator>
	array1d<typename Evaluator::result_type> array_columns_eval(
			const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, Evaluator evaluator)
	{
		typedef typename Evaluator::result_type result_t;

		array1d<result_t> r(x.ncolumns());

		index_t n = (index_t)x.ncolumns();
		for (index_t j = 0; j < n; ++j)
		{
			r(j) = array_eval(x.column(j), evaluator);
		}

		return r;
	}


	// Auxiliary routines

	template<typename T1, typename T2, class TIndexer>
	std::pair<array1d<T1>, array1d<T2> > unzip(const const_aview1d<std::pair<T1, T2>, TIndexer>& pairs)
	{
		array1d<T1> a1(pairs.nelems());
		array1d<T2> a2(pairs.nelems());

		index_t n = (index_t)pairs.nelems();
		for (index_t i = 0; i < n; ++i)
		{
			const std::pair<T1, T2>& p = pairs(i);

			a1[i] = p.first;
			a2[i] = p.second;
		}

		return std::make_pair(a1, a2);
	}


	// sum

	template<typename T>
	struct vec_sum_evaluator
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_sum(n, x);
		}
	};

	template<typename T, class TIndexer>
	T sum(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_sum_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_eval(x, vec_sum_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> row_sum(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_sum_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> column_sum(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_sum_evaluator<T>());
	}


	// prod

	template<typename T>
	struct vec_prod_evaluator
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_prod(n, x);
		}
	};

	template<typename T, class TIndexer>
	T prod(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_prod_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T prod(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_eval(x, vec_prod_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> row_prod(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_prod_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> column_prod(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_prod_evaluator<T>());
	}


	// max

	template<typename T>
	struct vec_max_evaluator
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_max(n, x);
		}
	};

	template<typename T, class TIndexer>
	T max(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_max_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T max(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_eval(x, vec_max_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> row_max(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_max_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> column_max(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_max_evaluator<T>());
	}


	// min

	template<typename T>
	struct vec_min_evaluator
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_min(n, x);
		}
	};

	template<typename T, class TIndexer>
	T min(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_min_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T min(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_eval(x, vec_min_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> row_min(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_min_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<T> column_min(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_min_evaluator<T>());
	}

	// minmax


	template<typename T>
	struct vec_minmax_evaluator
	{
		typedef std::pair<T, T> result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_minmax(n, x);
		}
	};

	template<typename T, class TIndexer>
	std::pair<T, T> minmax(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_minmax_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::pair<T, T> minmax(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_eval(x, vec_minmax_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<std::pair<T, T> > row_minmax(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_minmax_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<std::pair<T, T> > column_minmax(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_minmax_evaluator<T>());
	}



	// index max

	template<typename T>
	struct vec_index_max_evaluator
	{
		typedef std::pair<index_t, T> result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_index_max(n, x);
		}
	};

	template<typename T, class TIndexer>
	std::pair<index_t, T> index_max(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_index_max_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::pair<index_pair, T> index_max(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		std::pair<index_t, T> ret = array_eval(x, vec_index_max_evaluator<T>());
		index_pair p = _detail::layout_aux2d<TOrd>::ind2sub(x.dim0(), x.dim1(), ret.first);
		return std::make_pair(p, ret.second);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<std::pair<index_t, T> > row_index_max(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_index_max_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<std::pair<index_t, T> > column_index_max(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_index_max_evaluator<T>());
	}


	// index min

	template<typename T>
	struct vec_index_min_evaluator
	{
		typedef std::pair<index_t, T> result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_index_min(n, x);
		}
	};

	template<typename T, class TIndexer>
	std::pair<index_t, T> index_min(const const_aview1d<T, TIndexer>& x)
	{
		return array_eval(x, vec_index_min_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::pair<index_pair, T> index_min(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		std::pair<index_t, T> ret = array_eval(x, vec_index_min_evaluator<T>());
		index_pair p = _detail::layout_aux2d<TOrd>::ind2sub(x.dim0(), x.dim1(), ret.first);
		return std::make_pair(p, ret.second);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<std::pair<index_t, T> > row_index_min(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_rows_eval(x, vec_index_min_evaluator<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array1d<std::pair<index_t, T> > column_index_min(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_columns_eval(x, vec_index_min_evaluator<T>());
	}



};

#endif 
