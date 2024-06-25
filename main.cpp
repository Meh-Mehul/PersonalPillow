#include "src/Image.h"
int main(int argc, char** argv){
    // Sample usage of most of the functions of this library, some are broken, might be fixed later.

    Image test("test.jpg"); // Loading the Image
    Image test2("test2.jpg");
    // 1.Mosiac filter
    Image test_mosiac = test.oldTVFilter(test.getWidth()/4,test.getHeight()/4,test.getWidth()/2 , test.getHeight()/2, true);
    test_mosiac.write("imgs/mosaic.jpg");

    // 2.Cropping
    //Rectangular
    uint8_t color[3] = {0,0,0};
    Image test_crop_rect = test.crop(test.getWidth()/2,test.getHeight()/8,3*test.getWidth()/4 , test.getHeight()/2, true);
    test_crop_rect.write("imgs/crop_rect.jpg");
    //Circular (may be further cropped into smaller rectangular shape if needed)
    Image test_crop_circle = test.crop_circle(3*test.getWidth()/4, test.getHeight()/2,250, color);
    test_crop_circle.write("imgs/crop_circle.jpg");

    // 3. Flipping
    Image test_flip_x = test.flipX(3);
    Image test_flip_y = test.flipY(3);
    test_flip_x.write("imgs/flipx.jpg");
    test_flip_y.write("imgs/flipy.jpg");

    // 4. Grayscale and color manipulation
    Image test_grayscale = test.grayscale();
    test_grayscale.write("imgs/gray.jpg");
    Image test_mask = test.maskRGB(0.2 , 0.1, 0);
    test_mask.write("imgs/maskRGB.jpg");

    // 5. Overlaying images
    // This is a simple implementation , but if one wants, they can resize and then overlay also.
    Image test_overlay= test.overlay(test2, test.getWidth()/2, test.getHeight()/2);
    Image test_overlay_alpha = test.overlay_alpha(test2, 0, 0, 0.3);// Gives us choice of amount of visibility of Images
    test_overlay_alpha.write("imgs/overlay_alpha.jpg");
    test_overlay.write("imgs/overlay.jpg");

    // 6. Blurs
    Image test_Gblur = test.gaussianBlur(0,0,test.getWidth(), test.getHeight());
    Image test_Blured = test.Blur(11, 0,0,test.getWidth(), test.getHeight());
    test_Gblur.write("imgs/GaussianBlur.jpg");
    test_Blured.write("imgs/blur.jpg");

    // 7. Emboss
    Image test_emboss = test.emboss(0,0,test.getWidth(), test.getHeight(),false);// set want_grayscale to true if want grayscale :?
    test_emboss.write("imgs/emboss.jpg");

    // 8. Sharpen
    Image test_sharpen = test.sharpen(0,0,test.getWidth(), test.getHeight(), true);// set want_grayscale to true if want grayscale :?, btw the rgb one is kinda broken :(
    test_sharpen.write("imgs/sharpen.jpg");

    // 9. Sepia
    Image test_sepia = test.set_sepia_tone(0,0,test.getWidth(), test.getHeight());
    test_sepia.write("imgs/sepia.jpg");

    // 10.LOG (my favourite)
    Image test_log = test.laplace_of_gaussian(0,0,test.getWidth(), test.getHeight());
    test_log.write("imgs/log.jpg");

    // 11. Edge detection
    Image test_edge = test.get_Contour(true);// set to true to visualize sobel gradients
    test_edge.write("imgs/edge.jpg");

    // 12. Brighten
    Image test_brighten = test.change_Brightness(0.8); // between -1.0 to 1.0 , can inc or dec the brightness
    test_brighten.write("imgs/bright.jpg");

    // 13. Contrast
    Image test_contrast = test.increase_Contrast();
    Image test_contrast_rgb = test.increase_Contrast_RGB(); // same as above but with rgb
    test_contrast.write("imgs/contrast.jpg");
    test_contrast_rgb.write("imgs/contrast_rgb.jpg");

    // 14. ImageDraw
    // rectangle
    Image test_rect = test.draw_rect(test.getWidth()/4, test.getHeight()/4,test.getWidth()/2, test.getHeight()/2, 20 , 20, 230, 450, false);
    test_rect.write("imgs/draw_rect.jpg");
    // circle
    Image test_circle = test.draw_circle(test.getWidth()/2, test.getHeight()/2 , 100 , 0,0,0,20, false);
    test_circle.write("imgs/draw_circle.jpg");
    // line
    Image test_line = test.draw_line(test.getWidth()/3, test.getHeight()/3, test.getWidth(), test.getHeight(), 69, 150, 210, 5);
    test_line.write("imgs/draw_line.jpg");
    
    // 15. Color Detection
    // Detect colors lying in a specific range
    uint8_t lower[3] ={100,100,100};
    uint8_t upper[3] = {110,150,210};
    Image test_color = test.detect_color(lower, upper);
    test_color.write("imgs/color_detect.jpg");

    // 16. Resizing image
    // resizing test to test2's dimensions
    Image test_resize = test.resize(test2.getWidth(), test2.getHeight());
    test_resize.write("imgs/resize.jpg");
    return 0;
}