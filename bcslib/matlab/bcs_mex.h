/**
 * @file bcs_mex.h
 *
 * The interface between bcslib and matlab mex
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_MEX_H
#define BCSLIB_MEX_H

#include <bcslib/matlab/matlab_base.h>
#include <bcslib/matlab/marray.h>
#include <vector>
#include <iterator>

namespace bcs
{

namespace matlab
{

	// view from marray

	template<typename T>
	const_aview1d<T> view1d(const_marray a)
	{
		return const_aview1d<T>(a.data<T>(), a.nelems());
	}


	template<typename T>
	aview1d<T> view1d(marray a)
	{
		return aview1d<T>(a.data<T>(), a.nelems());
	}


	template<typename T>
	const_aview2d<T, column_major_t> view2d(const_marray a)
	{
		return const_aview2d<T, column_major_t>(a.data<T>(), a.nrows(), a.ncolumns(), a.nrows(), a.ncolumns());
	}


	template<typename T>
	aview2d<T, column_major_t> view2d(marray a)
	{
		return aview2d<T, column_major_t>(a.data<T>(), a.nrows(), a.ncolumns(), a.nrows(), a.ncolumns());
	}


	// to matlab vector

	template<typename T, class TIndexer>
	marray to_matlab_row(const const_aview1d<T, TIndexer>& v)
	{
		marray a = create_marray<T>(1, v.nelems());
		view1d<T>(a) << v;
		return a;
	}

	template<typename T, class TIndexer>
	marray to_matlab_column(const const_aview1d<T, TIndexer>& v)
	{
		marray a = create_marray<T>(v.nelems(), 1);
		view1d<T>(a) << v;
		return a;
	}


	template<typename T>
	marray to_matlab_row(std::vector<T>& v)
	{
		size_t n = v.size();
		marray a = create_marray<T>(1, n);
		copy_elements(&(v[0]), a.data<T>(), n);
		return a;
	}

	template<typename T>
	marray to_matlab_column(std::vector<T>& v)
	{
		size_t n = v.size();
		marray a = create_marray<T>(n, 1);
		copy_elements(&(v[0]), a.data<T>(), n);
		return a;
	}

	template<typename T, typename TFunc>
	marray to_matlab_row(std::vector<T>& v, TFunc f)
	{
		typedef typename TFunc::result_type R;
		size_t n = v.size();
		marray a = create_marray<R>(1, n);
		aview1d<R> av = view1d<R>(a);

		for (size_t i = 0; i < n; ++i)
		{
			av[i] = f(v[i]);
		}
		return a;
	}

	template<typename T, typename TFunc>
	marray to_matlab_column(std::vector<T>& v, TFunc f)
	{
		typedef typename TFunc::result_type R;
		size_t n = v.size();
		marray a = create_marray<R>(n, 1);
		aview1d<R> av = view1d<R>(a);

		for (size_t i = 0; i < n; ++i)
		{
			av[i] = f(v[i]);
		}
		return a;
	}


	template<typename TIter>
	marray to_matlab_row(TIter first, TIter last)
	{
		typedef typename std::iterator_traits<TIter>::value_type T;
		std::vector<T> vec(first, last);
		return to_matlab_row(vec);
	}

	template<typename TIter>
	marray to_matlab_column(TIter first, TIter last)
	{
		typedef typename std::iterator_traits<TIter>::value_type T;
		std::vector<T> vec(first, last);
		return to_matlab_column(vec);
	}


	template<typename TIter, typename TFunc>
	marray to_matlab_row(TIter first, TIter last, TFunc f)
	{
		typedef typename std::iterator_traits<TIter>::value_type T;
		std::vector<T> vec(first, last);
		return to_matlab_row(vec, f);
	}

	template<typename TIter, typename TFunc>
	marray to_matlab_column(TIter first, TIter last, TFunc f)
	{
		typedef typename std::iterator_traits<TIter>::value_type T;
		std::vector<T> vec(first, last);
		return to_matlab_column(vec, f);
	}


	// to matlab matrix

	template<typename T, class TIndexer0, class TIndexer1>
	marray to_matlab_matrix(const const_aview2d<T, column_major_t, TIndexer0, TIndexer1>& v)
	{
		marray a = create_marray<T>(v.nrows(), v.ncolumns());
		view2d<T>(a) << v;
		return a;
	}

}


}


#define BCSMEX_MAINDEF \
		void bcsmex_main(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]); \
		void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) \
		{ \
			try { \
				bcsmex_main(nlhs, plhs, nrhs, prhs); \
			} \
			catch(bcs::matlab::mexception& mexc) { \
				mexErrMsgIdAndTxt(mexc.identifier(), mexc.message()); \
			} \
		}


#endif 
