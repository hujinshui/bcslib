/**
 * @file arg_check.h
 *
 * Facilities for argument checking
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARG_CHECK_H_
#define BCSLIB_ARG_CHECK_H_

#include <bcslib/base/basic_defs.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace bcs
{

	// generic argument checking

	inline void check_arg(bool cond)
	{
		if (!cond)
		{
			throw std::invalid_argument("Invalid argument");
		}
	}


	inline void check_arg(bool cond, const std::string& message)
	{
		if (!cond)
		{
			throw std::invalid_argument(message);
		}
	}

	inline void check_arg(bool cond, const char* message)
	{
		if (!cond)
		{
			throw std::invalid_argument(message);
		}
	}

}


#endif 
