/**
 * @file basic_defs.h
 *
 * Library-wide basic definitions
 *
 * @author dhlin
 */


#ifndef BCSLIB_BASIC_DEFS_H
#define BCSLIB_BASIC_DEFS_H

#include <bcslib/base/config.h>
#include <stddef.h>

#if BCSLIB_COMPILER == BCSLIB_MSVC

#include <array>
#include <tuple>
#include <memory>
#include <type_traits>

#elif BCSLIB_COMPILER == BCSLIB_GCC

#include <tr1/array>
#include <tr1/tuple>
#include <tr1/memory>
#include <tr1/type_traits>

#endif

#include <string>


namespace bcs
{

#ifdef BCS_STDINT_INCLUDED

	using ::uint8_t;
	using ::int8_t;
	using ::uint16_t;
	using ::int16_t;
	using ::uint32_t;
	using ::int32_t;
	using ::uint64_t;
	using ::int64_t;

#else

	typedef signed char int8_t;
	typedef short int16_t;
	typedef int int32_t;

	typedef unsigned char uint8_t;
	typedef unsigned short uint16_t;
	typedef unsigned int uint32_t;

#endif

	using ::ptrdiff_t;
	using ::size_t;

	typedef uint8_t byte;

	typedef struct clone_t { };
	typedef struct own_t { };

	class base_exception
	{
	public:
		virtual ~base_exception() { }

		base_exception(const char *msg) : m_message(msg)
		{
		}

		const char *message() const
		{
			return m_message.c_str();
		}

	private:
		std::string m_message;
	};



	namespace tr1
	{

		// import tr1 names

		using std::tr1::shared_ptr;
		using std::tr1::array;
		using std::tr1::tuple;
		using std::tr1::result_of;

	}

}


#endif
