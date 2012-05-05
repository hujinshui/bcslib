/**
 * @file matrix_io.h
 *
 * Basic Input/Output of matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_IO_H_
#define BCSLIB_MATRIX_IO_H_

#include <bcslib/matrix/matrix_concepts.h>
#include <cstdio>

namespace bcs
{
	// printf_mat

	template<typename T, class Mat>
	void printf_mat(const char *fmt, const IMatrixView<Mat, T>& X,
			const char *pre_line=BCS_NULL, const char *delim="\n")
	{
		index_t m = X.nrows();
		index_t n = X.ncolumns();

		for (index_t i = 0; i < m; ++i)
		{
			if (pre_line) std::printf("%s", pre_line);
			for (index_t j = 0; j < n; ++j)
			{
				std::printf(fmt, X.elem(i, j));
			}
			if (delim) std::printf("%s", delim);
		}
	}

}



#endif /* MATRIX_IO_H_ */
