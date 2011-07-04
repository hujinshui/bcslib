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


}


#endif
