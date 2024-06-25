#include<cstdio>
#include<stdint.h>
#include<cmath>
#include<complex>
#include<cstdlib>
#include<iostream>
// #include <graphics.h>
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif
#define _USE_MATH_DEFINES
enum ImageType{
    PNG, JPG, BMP, TGA
};

struct Image
{
    uint8_t* data = NULL;
    size_t size = 0;
    int w;
    int h;
    int channels;
    char* filename;
public:
    Image(const char* filename);
    Image(int w, int h, int channels);
    Image(const Image& img);
    ~Image();
    int getWidth();
    int getHeight();
public:
    bool read(const char* filename);
    bool write(const char* filename);
    static uint32_t bit_reverse_help(uint32_t n, uint32_t a);
    static void bit_reverse(uint32_t n, std::complex<double> a[], std::complex<double>* A);
    ImageType getFileType(const char* filename);
    Image& oldTVFilter(int x1, int y1, int x2, int y2, bool copy = true);
    Image& crop( int x1, int y1, int x2, int y2, bool copy = true);
    /// @brief Crops a certain circular region off the image
    /// @param x x-coordinate of center of circle
    /// @param y y-coordinate of center of circle
    /// @param radius radius of cut circle
    /// @param bg_color background color of the cropped image(if applicable), Array of size 3, {R,G,B} in order.
    /// @return returns a pointer to another image with the desired changes.
    Image& crop_circle(int x, int y, int radius, uint8_t bg_color[3]);
    Image& grayscale_avg();
    Image& grayscale();
    Image& maskRGB(float a, float b, float c, bool copy = true);
    Image& createDiffmap(Image& img, uint8_t scl= 0);
    /// @param copy Please KEEP it True
    Image& flipX(int numchannels, bool copy = true);
    Image& flipY(int numchannels,bool copy = true);
    Image& overlay(Image& source , int x, int y, bool copy = true);
    /// @brief Overlays The src image onto the current image, Please Note: THIS IS NOT APLHA BLENDING, just a similar approach
    /// @param src Source image (Image*)
    /// @param x x-coordinate of overlay start
    /// @param y y-coordinate of overlay start
    /// @param aplha fraction of the current image visible in the overlay region  
    /// @return returns a pointer to another image with the desired changes.
    Image& overlay_alpha(Image& source, int x, int y, double aplha);
    Image& gaussianBlur( int x1, int y1, int x2, int y2, bool copy = true);
    Image& Blur(uint8_t scale,int x1, int y1, int x2, int y2, bool copy = true);
    Image& emboss(int x1, int y1, int x2, int y2,bool want_grayscale = true, bool copy = true);
    /// @brief Unlike other filter functions this one just returns a new image with sepia tone set. Refrence:https://stackoverflow.com/questions/1061093/how-is-a-sepia-tone-created
    /// @return returns a pointer to another image with the desired changes.
    Image& set_sepia_tone(int x1, int y1, int x2, int y2);
    Image& sharpen(int x1, int y1, int x2, int y2,bool want_grayscale = true, bool copy = true);
    Image& TSobel(int x1, int y1, int x2, int y2, bool copy = true);
    Image& BSobel(int x1, int y1, int x2, int y2, bool copy = true);
    Image& create_grayscale_outline(int x1, int y1, int x2, int y2, bool copy = true);
    /// @brief I call this, ```The Weird Function```, It does weird things to ur image
    /// @return returns a pointer to another image with the desired changes.
    Image& laplace_of_gaussian(int x1, int y1, int x2, int y2, bool copy = true);
    /// @brief Creates an new image with just the edges of the current image
    /// @param rgbedges set this to true if u want to visualize edge gradients
    /// @return returns a pointer to another image with the desired changes.
    Image& get_Contour(bool rgbedges = true);
    /// @brief factor is between -1.0 and 1.0
    /// @return returns a pointer to another image with the desired changes.
    Image& change_Brightness(double factor);
    Image& increase_Contrast();
    Image& increase_Contrast_RGB();
    /// @brief Detects the colors in the specified ranges ```lower_color_bound``` to ```upper_color_bound```, and show a grayscale version of detected areas.
    /// @param lower_color_bound array of size 3 representing the lower RGB bound of the color
    /// @param lower_color_bound array of size 3 representing the upper RGB bound of the color
    /// @return returns a pointer to another image with the desired changes.
    Image& detect_color(uint8_t lower_color_bound[3], uint8_t upper_color_bound[3]);
    /// @brief Don't use this Please, unless You want to! It's a Very Buggy function based on the old graphics.h library
    void imshow_slow();
    /// @brief Draws a Rectangle with bounds (x1,y1) to (x2, y2)
    /// @return returns a pointer to another image with the desired changes.
    Image& draw_rect(int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, uint16_t thickness, bool fill = false );
    /// @brief Draws a circle with center (x,y) and radius
    /// @return returns a pointer to another image with the desired changes.
    Image& draw_circle(int x, int y, int radius, uint8_t r, uint8_t g, uint8_t b, uint16_t thickness, bool fill = false);
    /// @brief Draws a line from (x1,y1) to (x2, y2) with given thickness, This has jagged ends, to prevent that, circles may be drawn at ends
    /// @return returns a pointer to another image with the desired changes.
    Image& draw_line(int x1,int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, uint16_t thickness);
    /// @brief gives a frame of given color to the picture
    /// @return returns a pointer to another image with the desired changes.
    Image& framify_color(uint8_t r, uint8_t g, uint8_t b, int thickness);
    /// @brief This function resizes the image to the dimensions (new_h, new_w), Using bilinear interpolation, Please note this currently works for 3 channel images, to resize grayscales please open them as three channel images. This may not wokr well with pixelated images.
    /// @return returns a pointer to another image with the desired changes.
    Image& resize( int new_w, int new_h);


private:
    static void fft(uint32_t n, std::complex<double> x[], std::complex<double>* X);
    static void ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x);
    static void dft_2D(uint32_t n, uint32_t m, std::complex<double> x[], std::complex<double>* X);
    static void idft_2D(uint32_t n, uint32_t m, std::complex<double> X[], std::complex<double>* x);
    static void pad_kernel(uint32_t kew_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker);
    static inline void pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p);
    Image& fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
    Image& convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
    Image& std_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
    Image& applykernel(uint8_t channels, uint8_t ker_w, uint8_t ker_h, double ker[], bool copy = true);
    //TODO: 
    // Image Filters:
    //  DONE:
    //     1.Mosiac
    //     2.Cropping
    //     3.Flipping
    //     4.RBG Changing and GrayScale
    //     5.Putting images over the Other
    //     6.Gaussian Blur
    //     7.Scaled Blur from 5 to 15 intensity
    //     8.Emboss
    //     9.Sharpen
    //     16.Sepia
    //     10.LOG(laplace of gaussian edge detection)
    //     11.Edge Detection 
    //     12.Brightness
    //     13.Contrast
    //     14.ImageDraw(Easy Elements only)
    //     15.Color detection
    //     17.Alpha Overlay
    //     18.resize image

};

