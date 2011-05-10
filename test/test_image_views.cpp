/**
 * @file test_image_views.cpp
 *
 * The unit testing for image views
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/image/image.h>


using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::const_imview<uint8_t>;
template class bcs::const_imview<mcpixel<uint8_t, 3> >;
template class bcs::imview<uint8_t>;
template class bcs::imview<mcpixel<uint8_t, 3> >;


template<typename TPixel>
bool test_image_view(const_imview<TPixel> im, size_t w, size_t h, img_index_t s, const TPixel *src)
{
	if (im.width() != w) return false;
	if (im.height() != h) return false;
	if (im.stride() != s) return false;
	if (im.npixels() != w * h) return false;

	for (img_index_t i = 0; i < (img_index_t)h; ++i)
	{
		for (img_index_t j = 0; j < (img_index_t)w; ++j)
		{
			if (im(i, j) != *(src++)) return false;
		}
	}

	return true;
}



BCS_TEST_CASE( test_view_ch1 )
{
	uint8_t srcs[] = {1, 2, 3, 4, 5, 6, 7, 8};

	imview<uint8_t> im1(srcs, 3, 2);
	BCS_CHECK( test_image_view(im1, 3, 2, (img_index_t)(3 * sizeof(uint8_t)), srcs) );

	imview<uint8_t> im2(srcs, 3, 2, 4 * sizeof(uint8_t));
	uint8_t r2[] = {1, 2, 3, 5, 6, 7};
	BCS_CHECK( test_image_view(im2, 3, 2, (img_index_t)(4 * sizeof(uint8_t)), r2) );
}


BCS_TEST_CASE( test_view_ch2 )
{
	typedef mcpixel<uint8_t, 2> pix_t;

	uint8_t srcs0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	pix_t *srcs = (pix_t*)srcs0;

	imview<pix_t> im1(srcs, 3, 2);
	BCS_CHECK( test_image_view(im1, 3, 2, (img_index_t)(3 * sizeof(pix_t)), srcs) );

	imview<pix_t> im2(srcs, 3, 2, 4 * sizeof(pix_t));
	uint8_t r2i[] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14};
	BCS_CHECK( test_image_view(im2, 3, 2, (img_index_t)(4 * sizeof(pix_t)), (const pix_t*)r2i) );
}



test_suite* test_image_views_suite()
{
	test_suite *suite = new test_suite( "test_image_views" );

	suite->add( new test_view_ch1() );
	suite->add( new test_view_ch2() );

	return suite;
}


