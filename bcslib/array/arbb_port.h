/**
 * @file arbb_port.h
 *
 * The interfaces with Intel ArBB
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARBB_PORT_H_
#define BCSLIB_ARBB_PORT_H_

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <arbb.hpp>

#include <algorithm>

namespace bcs
{

	/*****
	 *
	 *  The relation between bcs arrays and arbb containers
	 *
	 *  Let T be a C-type, and CT be the captured type of T
	 *
	 *  bcs::array1d<T> x ----  arbb::dense<CT, 1> a:
	 *  	- x.nelems() ---- a.length()
	 *
	 *  bcs::array2d<T, row_major_t>  ----  arbb::dense<CT, 2>:
	 * 		- x.nrows()    ---- a.num_rows()
	 * 		- x.ncolumns() ---- a.num_cols()
	 *
	 *  bcs::array2d<T, column_major_t> ---- arbb::dense<T, 2>: (switch row/column)
	 *  	- x.nrows()    ---- a.num_cols()
	 *  	- x.ncolumns() ---- a.num_rows()
	 */


	// copy

	template<typename T>
	inline void copy(const caview1d<T>& src, arbb::dense<typename arbb::captured<T>::type, 1>& dst)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t n = src.nelems();
		check_arg(n == arbb::value(dst.length()), "bcs::copy: caview1d -> arbb: inconsistent array dimensions");

		arbb::range<CT> rgn = dst.write_only_range();
		std::copy_n(src.begin(), n, rgn.begin());
	}

	template<typename T>
	inline void copy(const arbb::dense<typename arbb::captured<T>::type, 1>& src, aview1d<T>& dst)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t n = dst.nelems();
		check_arg(n == arbb::value(src.length()), "bcs::copy: arbb -> aview1d: inconsistent array dimensions");

		arbb::const_range<CT> rgn = src.read_only_range();
		std::copy_n(rgn.begin(), n, dst.begin());
	}

	template<typename T>
	inline void copy(const caview2d<T, row_major_t>& src, arbb::dense<typename arbb::captured<T>::type, 2>& dst)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t m = src.nrows();
		size_t n = src.ncolumns();

		check_arg(m == arbb::value(dst.num_rows()) && n == arbb::value(dst.num_cols()),
				"bcs::copy: caview2d -> arbb: inconsistent array dimensions");

		arbb::range<CT> rgn = dst.write_only_range();
		std::copy_n(src.begin(), m * n, rgn.begin());
	}

	template<typename T>
	inline void copy(const arbb::dense<typename arbb::captured<T>::type, 2>& src, aview2d<T, row_major_t>& dst)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t m = dst.nrows();
		size_t n = dst.ncolumns();

		check_arg(m == arbb::value(src.num_rows()) && n == arbb::value(src.num_cols()),
				"bcs::copy: arbb -> aview2d: inconsistent array dimensions");

		arbb::const_range<CT> rgn = src.read_only_range();
		std::copy_n(rgn.begin(), m * n, dst.begin());
	}

	template<typename T>
	inline void copy(const caview2d<T, column_major_t>& src, arbb::dense<typename arbb::captured<T>::type, 2>& dst)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t m = src.nrows();
		size_t n = src.ncolumns();

		check_arg(n == arbb::value(dst.num_rows()) && m == arbb::value(dst.num_cols()),
				"bcs::copy: caview2d -> arbb: inconsistent array dimensions");

		arbb::range<CT> rgn = dst.write_only_range();
		std::copy_n(src.begin(), m * n, rgn.begin());
	}

	template<typename T>
	inline void copy(const arbb::dense<typename arbb::captured<T>::type, 2>& src, aview2d<T, column_major_t>& dst)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t m = dst.nrows();
		size_t n = dst.ncolumns();

		check_arg(n == arbb::value(src.num_rows()) && m == arbb::value(src.num_cols()),
				"bcs::copy: arbb -> aview2d: inconsistent array dimensions");

		arbb::const_range<CT> rgn = src.read_only_range();
		std::copy_n(rgn.begin(), m * n, dst.begin());
	}


	// clone (convert)

	template<typename T>
	inline arbb::dense<typename arbb::captured<T>::type, 1> to_arbb(const caview1d<T>& a)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t n = a.nelems();
		arbb::dense<CT, 1> r(n);
		copy(a, r);

		return r;
	}

	template<typename CT>
	inline array1d<typename arbb::uncaptured<CT>::type> from_arbb(const arbb::dense<CT, 1>& a)
	{
		typedef typename arbb::uncaptured<CT>::type T;

		size_t n = arbb::value(a.length());
		array1d<T> r(n);
		copy(a, r);

		return r;
	}

	template<typename T>
	inline arbb::dense<typename arbb::captured<T>::type, 2> to_arbb(const caview2d<T, row_major_t>& a)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t m = a.nrows();
		size_t n = a.ncolumns();

		arbb::dense<CT, 2> r(n, m);
		copy(a, r);

		return r;
	}

	template<typename CT>
	inline array2d<typename arbb::uncaptured<CT>::type, row_major_t> from_arbb(const arbb::dense<CT, 2>& a, row_major_t)
	{
		typedef typename arbb::uncaptured<CT>::type T;

		size_t m = arbb::value(a.num_rows());
		size_t n = arbb::value(a.num_cols());

		array2d<T, row_major_t> r(m, n);
		copy(a, r);

		return r;
	}

	template<typename T>
	inline arbb::dense<typename arbb::captured<T>::type, 2> to_arbb(const caview2d<T, column_major_t>& a)
	{
		typedef typename arbb::captured<T>::type CT;

		size_t m = a.nrows();
		size_t n = a.ncolumns();

		arbb::dense<CT, 2> r(m, n);
		copy(a, r);

		return r;
	}

	template<typename CT>
	inline array2d<typename arbb::uncaptured<CT>::type, column_major_t> from_arbb(const arbb::dense<CT, 2>& a, column_major_t)
	{
		typedef typename arbb::uncaptured<CT>::type T;

		size_t n = arbb::value(a.num_rows());
		size_t m = arbb::value(a.num_cols());

		array2d<T, column_major_t> r(m, n);
		copy(a, r);

		return r;
	}


	// bind

	template<typename T>
	inline void bind(arbb::dense<typename arbb::captured<T>::type, 1>& managed, aview1d<T>& native)
	{
		arbb::bind(managed, native.pbase(), native.nelems());
	}

	template<typename T>
	inline void bind(arbb::dense<typename arbb::captured<T>::type, 2>& managed, aview2d<T, row_major_t>& native)
	{
		check_arg( native.base_dim1() == (index_t)native.ncolumns(), "bcs::bind: arbb with aview2d: can only bind to continuous views.");
		arbb::bind(managed, native.pbase(), native.ncolumns(), native.nrows());
	}

	template<typename T>
	inline void bind(arbb::dense<typename arbb::captured<T>::type, 2>& managed, aview2d<T, column_major_t>& native)
	{
		check_arg( native.base_dim0() == (index_t)native.nrows(), "bcs::bind: arbb with aview2d: can only bind to continuous views.");
		arbb::bind(managed, native.pbase(), native.nrows(), native.ncolumns());
	}

}


#endif
