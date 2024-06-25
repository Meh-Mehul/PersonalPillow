#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BOUND(value) value <0 ? 0: (value>255 ? 255:value)
#include "Image.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include<cstdio>
#include<stdint.h>
#include<iostream>
#include<time.h>
#include<map>
#include<vector>
#include<algorithm>
// Simple Library for image filters in CPP Using the STB_Image Library
// For more info on kernels used : https://setosa.io/ev/image-kernels/

Image::Image(const char* filename){
    if(read(filename)){
        printf("Read %s\n", filename);
        size = w*h*channels;
    }
    else{
        printf("Failed to read %s\n", filename);
    }
}
Image::Image(int w, int h, int channels):w(w),h(h), channels(channels) {
    size = w*h*channels;
    data= new uint8_t[size];
}
Image::Image(const Image& img):Image(img.w, img.h, img.channels){
    memcpy(data, img.data, size);//copy image data into our own image data

}
Image::~Image(){
    stbi_image_free(data);
}

bool Image::read(const char* filename){
    data = stbi_load(filename, &w, &h, &channels, 0);
    return data!=NULL;
}
ImageType Image::getFileType(const char* filename){
    const char* extention = strrchr(filename, '.');
    if(extention != nullptr){
        if(strcmp(extention, ".png") == 0){
            return PNG;
        }
        else if(strcmp(extention, ".jpg") == 0){
            return JPG;
        }
        else if(strcmp(extention, ".bmp") == 0){
            return BMP;
        }
        else if(strcmp(extention, ".tga") == 0){
            return TGA;
        }
    }
    return PNG;
}
bool Image::write(const char* filename){
    ImageType type = getFileType(filename);
    int succ;
    switch (type)
    {
    case PNG:
            succ = stbi_write_png(filename, w, h, channels, data,w*channels);
        break;
    case BMP:
            succ = stbi_write_bmp(filename, w, h, channels,data);
        break;
    case JPG:
            succ = stbi_write_jpg(filename, w, h, channels, data, 100);
        break;
    case TGA:
            succ = stbi_write_tga(filename, w, h, channels, data);
        break;
    }
    return succ!=0;
}
    int Image::getWidth(){
        return w;
    }
    int Image::getHeight(){
        return h;
    }
//  filename would only work when copy is enabled to false, otherwise it has no use
// copy allows you to save the image as an entirely new image to some other file, during runtime, or just operate on current image.
// Currently works for rectangle bounded area only
// assume yi row
// start of each row of the array :- x1*channels + yi*w*channels
// end of each row of the array :- x2*channels + yi*w*channels
Image& Image::oldTVFilter(int x1, int y1, int x2, int y2, bool copy){
    srand(time(0));
// so we iterate through yi from y1 to y2 and mosiac all the datapoints between these endpoints
// Check for bounds
    if(y2>h || x2>w){
        std::cout<<"Exception caused, the location of filter Exceeds the location for the Image data";
        return *this;
    }
    if(!copy){for(int i = y1; i<y2; ++i){
            for(int j = x1*channels + i*w*channels; j< x2*channels + i*w*channels;++j){
                data[j] = rand()%255 + 1;
            }
        }
        return *this;}
    else{
        Image* cpy = new Image(this->w, this->h, this->channels);
        for(int i = 0; i<w*h*channels; ++i){
            cpy->data[i] = data[i];
        }
        for(int i = y1; i<y2; ++i){
            for(int j = x1*channels + i*w*channels; j< x2*channels + i*w*channels;++j){
                cpy->data[j] = rand()%255 + 1;
            }
        }
        // cpy->write(filename);
        return *cpy;

    }
}

Image& Image::crop( int x1, int y1, int x2, int y2, bool copy){
// The filename arg is useful only when copy
// The Implementation is similar to the previous except this time we only take the desired part.
    if(x2<=x1 || y2<=y1){
        std::cout<<"[ERROR]Second point is 'behind' the first, It should have x2>x1 and y2>y1.\n";
        return *this;
    }
    if(x2>w || y2>h || x1<0 || y1<0){
        std::cout<<"[ERROR]Region Selected is out of bounds of the image.\n";
        return *this;
    }
    if(copy){
        Image* cpy = new Image(x2-x1, y2-y1, channels);
        int cntr = 0;
        for(int i = y1; i<y2; ++i){
            for(int j = x1*channels + i*w*channels; j< x2*channels + i*w*channels;++j){
                cpy->data[cntr] = data[j];
                cntr++;
            }
        }
        // cpy->write(filename);
        return *cpy;
    }
    else{
        Image cpy(x2-x1, y2-y1, channels);
        int cntr = 0;
        for(int i = y1; i<y2; ++i){
            for(int j = x1*channels + i*w*channels; j< x2*channels + i*w*channels;++j){
                cpy.data[cntr] = data[j];
                cntr++;
            }
        }
        w = cpy.w;
        h = cpy.h;
        size = cpy.w*cpy.h*cpy.channels;
        // stbi_image_free(data);
        data = new uint8_t[size];
        for(int i = 0; i<size; i++){
            data[i] = cpy.data[i];
        }
        // write(filename);
        return *this;
        
    }
}

Image& Image::crop_circle(int x, int y, int radius, uint8_t bg_color[3]){
    if(radius <=0){
        printf("[Error] Radius should be > 0.\n");
        return *this;
    }
    Image* cpy = new Image(*this);
    for(uint32_t k = 0; k<w*h*channels; k += channels){
        int col = (k/channels)%w;
        int row = (k/channels)/w + 1;
        float distance = sqrt(pow((row-y),2)+pow((col-x),2));
        if(distance> radius){
            cpy->data[k] = bg_color[0];
            cpy->data[k+1] = bg_color[1];
            cpy->data[k+2] = bg_color[2];
        }
    }
    return *cpy;
}
Image& Image::set_sepia_tone(int x1, int y1, int x2, int y2){
    Image* cpy= new Image(*this);
    for(uint32_t k = 0; k<w*h*channels; k += channels){
        int col = (k/channels)%w;
        int row = (k/channels)/w + 1;
        if(col <= fmax(x1,x2) && col >= fmin(x1,x2) && row <= fmax(y1,y2) && row >= fmin(x1,x2)){
            double inputRed  = cpy->data[k];
            double inputGreen = cpy->data[k+1];
            double inputBlue  = cpy->data[k+2];
            uint8_t outputRed = BOUND((inputRed * 0.393) + (inputGreen *0.769) + (inputBlue * 0.189));
            uint8_t outputGreen = BOUND((inputRed * 0.349) + (inputGreen *0.686) + (inputBlue * 0.168));
            uint8_t outputBlue = BOUND((inputRed * 0.272) + (inputGreen *0.534) + (inputBlue * 0.131));
            cpy->data[k] = outputRed;
            cpy->data[k+1] = outputGreen;
            cpy->data[k+2] = outputBlue;
        }
    }
    return *cpy;
}
Image& Image::grayscale_avg(){
// new vale = (r+b+g)/3
    if(channels<3){
        std::cout<<"[ERROR]Image cant be GrayScaled as Channels are already less than 3\n";
        return *this;
    }
    else{
        for(int i = 0; i<size; i+= channels){
            int new_ = (data[i]+data[i+1]+data[i+2])/3;
            memset(data+i, new_, 3);
        }
        return *this;
    }

}
Image& Image::grayscale(){
// This permanently changes the data of the image
// for refrence : https://en.wikipedia.org/wiki/Grayscale
// Values obtained by gamma compressed model formulae
    if(channels<3){
        std::cout<<"[ERROR]Image cant be GrayScaled as Channels are already less than 3\n";
        return *this;
    }
    else{
        Image* cpy = new Image(*this);
        for(int i = 0; i<size; i+= channels){
            int new_ = 0.2126*data[i]+0.7152*data[i+1]+0.0722*data[i+2];
            memset(cpy->data+i, new_, 3);
        }
        return *cpy;
    }
}
Image& Image::maskRGB(float a, float b, float c, bool copy){
    // returns a pointer to an image and DOES change current Image data if copy is set to false.
    if(this->channels != 3){
        std::cout<<"[ERROR]Image cant be masked, we need exactly 3 channels for RBG masking\n";
        return *this;
    }
    if(a >1 || b>1 || c>1){
        std::cout<<"[ERROR]Image cant be masked, The params a,b,c each should be in the range 0 to 1.\n";
        return *this;
    }
    if(copy){
        Image* cpy = new Image(*this);
        for(int i = 0; i<size; i+= channels){
            cpy->data[i] *=a;
            cpy->data[i+1] *= b;
            cpy->data[i+2] *= c;
        }
        return *cpy;
    }

    for(int i = 0; i <size; i+= channels){
        data[i] *=a;
        data[i+1] *= b;
        data[i+2] *= c;
    }
    return *this;
}
Image& Image::createDiffmap(Image& img, uint8_t scl){
    int minw = fmin(w, img.w);
    int minh = fmin(h, img.h);
    int minchannels = fmin(channels, img.channels);
    Image* cpy = new Image(this->w, this->h, this->channels);
    uint8_t largest = 0;
    for(uint32_t i = 0; i<minh; ++i){
        for(uint32_t j = 0; j<minw; ++j){
            for(uint8_t k = 0; k<minchannels; ++k){
                cpy->data[(i*w+j)*channels +k] = BOUND(abs(data[(i*w+j)*channels +k] - img.data[(i*img.w+j)*img.channels +k]));
                largest = fmax(largest,cpy->data[(i*w+j)*channels +k] );
            }
        }
    }
    scl  = 255/fmax(1, fmax(scl, largest));// between 0 and 255
    for(int i = 0; i<size; ++i){
        cpy->data[i] *= scl;
    }
    return *cpy;
}
Image& Image::flipY(int numchannels,bool copy){
    if(numchannels >channels){
        std::cout<<"[ERROR]Image cant be Flipped as numchannels exceed image channels.\n";
        return *this;
    }
    if(copy){
        uint8_t temp[4];
        uint8_t* pixel1;
        uint8_t* pixel2;
        Image* cpy = new Image(*this);
        for (int y = 0; y < h; ++y){
            for(int x = 0; x<w/2; ++x){
                pixel1 = &cpy->data[(x + y*w)*channels];
                pixel2 = &cpy->data[((w-1-x) + y*w)*channels];
                memcpy(temp, pixel1, numchannels);
                memcpy(pixel1, pixel2, numchannels);
                memcpy(pixel2, temp, numchannels);
            }
        }
        return *cpy;
    }
    uint8_t temp[4];
    uint8_t* pixel1;
    uint8_t* pixel2;
    for (int y = 0; y < h; ++y){
        for(int x = 0; x<w/2; ++x){
            pixel1 = &data[(x + y*w)*channels];
            pixel2 = &data[((w-1-x) + y*w)*channels];
            memcpy(temp, pixel1, numchannels);
            memcpy(pixel1, pixel2, numchannels);
            memcpy(pixel2, temp, numchannels);
        }
    }
    return *this;
}
Image& Image::flipX(int numchannels,bool copy){
    if(numchannels >channels){
        std::cout<<"[ERROR]Image cant be Flipped as numchannels exceed image channels.\n";
        return *this;
    }    
    if(copy){
        uint8_t temp[4];
        uint8_t* pixel1;
        uint8_t* pixel2;
        Image* cpy = new Image(*this);
        for (int x = 0; x < w; ++x){
            for(int y = 0; y<h/2; ++y){
                pixel1 = &cpy->data[(x + y*w)*channels];
                pixel2 = &cpy->data[(x + (h-1-y)*w)*channels];
                memcpy(temp, pixel1, numchannels);
                memcpy(pixel1, pixel2, numchannels);
                memcpy(pixel2, temp, numchannels);
            }
        }
        return *cpy;
    }
    uint8_t temp[4];
    uint8_t* pixel1;
    uint8_t* pixel2;
    for (int x = 0; x < w; ++x){
        for(int y = 0; y<h/2; ++y){
            pixel1 = &data[(y + x*h)*channels];
            pixel2 = &data[((h-1-y) + x*h)*channels];
            memcpy(temp, pixel1, numchannels);
            memcpy(pixel1, pixel2, numchannels);
            memcpy(pixel2, temp, numchannels);
        }
    }
    return *this;
}
Image& Image::overlay(Image& source , int x, int y, bool copy){
    if(!copy){
    uint8_t* srcpx;
    uint8_t* dstpx;
    for(int sy = 0; sy<source.h; ++sy){
        if(sy + y<0){continue;}
        else if (sy + y >= h){break;}
        for(int sx = 0; sx<source.w; ++sx){
            if(sx +x<0){continue;}
            else if (sx+ x >= w){break;}
            srcpx = &source.data[(sx + sy*source.w)*source.channels];
            dstpx = &data[(sx +x + (sy+y)*w)*channels];

            float salpha  = source.channels < 4 ?1 : srcpx[3]/255.f;
            float dalpha  = channels < 4 ?1 : dstpx[3]/255.f;

            if(salpha > 0.99 && dalpha > 0.99){
                if(source.channels >= channels){memcpy(dstpx, srcpx, channels);}
                else{memset(dstpx, srcpx[0], channels);}
            }
            else{
                float oalpha = salpha + dalpha*(1-salpha);
                if(oalpha < 0.01){
                    memset(dstpx, 0, channels);
                }
                else{
                    for(int chnl = 0; chnl < channels; ++chnl ){
                    dstpx[chnl] = (uint8_t)BOUND((srcpx[chnl]/255.f*salpha + dstpx[chnl]/255.f*dalpha*(1-salpha))/oalpha * 255.f);
                    }
                    if(channels > 3) {dstpx[3] = (uint8_t)BOUND(oalpha*255.f);}
                }
            }
        }
    }
    return *this;
}
    else{
    Image* cpy = new Image(*this);
    uint8_t* srcpx;
    uint8_t* dstpx;
    for(int sy = 0; sy<source.h; ++sy){
        if(sy + y<0){continue;}
        else if (sy + y >= h){break;}
        for(int sx = 0; sx<source.w; ++sx){
            if(sx +x<0){continue;}
            else if (sx+ x >= w){break;}
            srcpx = &source.data[(sx + sy*source.w)*source.channels];
            dstpx = &cpy->data[(sx +x + (sy+y)*w)*channels];

            float salpha  = source.channels < 4 ?1 : srcpx[3]/255.f;
            float dalpha  = cpy->channels < 4 ?1 : dstpx[3]/255.f;
            if(salpha > 0.99 && dalpha > 0.99){
                if(source.channels >= channels){memcpy(dstpx, srcpx, channels);}
                else{memset(dstpx, srcpx[0], channels);}
            }
            else{
                float oalpha = salpha + dalpha*(1-salpha);
                if(oalpha < 0.01){
                    memset(dstpx, 0, channels);
                }
                else{
                    for(int chnl = 0; chnl < cpy->channels; ++chnl ){
                    dstpx[chnl] = (uint8_t)BOUND((srcpx[chnl]/255.f*salpha + dstpx[chnl]/255.f*dalpha*(1-salpha))/oalpha * 255.f);
                    }
                    if(cpy->channels > 3) {dstpx[3] = (uint8_t)BOUND(oalpha*255.f);}
                }
            }
        }
    }

        return *cpy;
    }
}
Image& Image::overlay_alpha(Image& source, int x, int y, double aplha){
    if(channels != 3){
        printf("[Code Error]The current implementation is only working with 3 channel images;( .\n");
        return *this;
    }
    Image* cpy = new Image(*this);
    uint8_t* srcpx;
    uint8_t* dstpx;
    for(int sy = 0; sy<source.h; ++sy){
        if(sy + y<0){continue;}
        else if (sy + y >= h){break;}
        for(int sx = 0; sx<source.w; ++sx){
            if(sx +x<0){continue;}
            else if (sx+ x >= w){break;}
            srcpx = &source.data[(sx + sy*source.w)*source.channels];
            dstpx = &cpy->data[(sx +x + (sy+y)*w)*channels];
            dstpx[0] = BOUND(aplha*dstpx[0] + (1-aplha)*srcpx[0]);
            dstpx[1] = BOUND(aplha*dstpx[1] + (1-aplha)*srcpx[1]);
            dstpx[2] = BOUND(aplha*dstpx[2] + (1-aplha)*srcpx[2]);
            // float salpha  = source.channels < 4 ?1 : srcpx[3]/255.f;
            // float dalpha  = cpy->channels < 4 ?1 : dstpx[3]/255.f;
            // if(salpha > 0.99 && dalpha > 0.99){
            //     if(source.channels >= channels){memcpy(dstpx, srcpx, channels);}
            //     else{memset(dstpx, srcpx[0], channels);}
            // }
            // else{
            //     float oalpha = salpha + dalpha*(1-salpha);
            //     if(oalpha < 0.01){
            //         memset(dstpx, 0, channels);
            //     }
            //     else{
            //         for(int chnl = 0; chnl < cpy->channels; ++chnl ){
            //         dstpx[chnl] = (uint8_t)BOUND((srcpx[chnl]/255.f*salpha + dstpx[chnl]/255.f*dalpha*(1-salpha))/oalpha * 255.f);
            //         }
            //         if(cpy->channels > 3) {dstpx[3] = (uint8_t)BOUND(oalpha*255.f);}
            //     }
            // }
        }
    }

        return *cpy;
}

uint32_t Image::bit_reverse_help(uint32_t n, uint32_t a){
    uint8_t max_bits = (uint8_t)ceil(log2(n));
    uint32_t reved_a = 0;
    for(uint8_t i = 0; i<max_bits; ++i){
        if(a&(i<<i)){
            reved_a |= (1<<(max_bits-i-1));
        }
    }
    return reved_a;
}
void Image::bit_reverse(uint32_t n, std::complex<double> a[], std::complex<double>* A){
    for(uint8_t i = 0; i<n; ++i){
        A[bit_reverse_help(n,i)] = a[i];
    }
} 
void Image::fft(uint32_t n, std::complex<double> x[], std::complex<double>* X){
    if(x != X){
        memcpy(X, x, n*sizeof(std::complex<double>));
    }
	//Gentleman-Sande butterfly
	uint32_t sub_probs = 1;
	uint32_t sub_prob_size = n;
	uint32_t half;
	uint32_t i;
	uint32_t j_begin;
	uint32_t j_end;
	uint32_t j;
	std::complex<double> w_step;
	std::complex<double> w;
	std::complex<double> tmp1, tmp2;
	while(sub_prob_size>1) {
		half = sub_prob_size>>1;
		w_step = std::complex<double>(cos(-2*M_PI/sub_prob_size), sin(-2*M_PI/sub_prob_size));
		for(i=0; i<sub_probs; ++i) {
			j_begin = i*sub_prob_size;
			j_end = j_begin+half;
			w = std::complex<double>(1,0);
			for(j=j_begin; j<j_end; ++j) {
				tmp1 = X[j];
				tmp2 = X[j+half];
				X[j] = tmp1+tmp2;
				X[j+half] = (tmp1-tmp2)*w;
				w *= w_step;
			}
		}
		sub_probs <<= 1;
		sub_prob_size = half;
	}

}
void Image::ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x){
    if(X != x) {
		memcpy(x, X, n*sizeof(std::complex<double>));
	}

	//Cooley-Tukey butterfly
	uint32_t sub_probs = n>>1;
	uint32_t sub_prob_size;
	uint32_t half = 1;
	uint32_t i;
	uint32_t j_begin;
	uint32_t j_end;
	uint32_t j;
	std::complex<double> w_step;
	std::complex<double> w;
	std::complex<double> tmp1, tmp2;
	while(half<n) {
		sub_prob_size = half<<1;
		w_step = std::complex<double>(cos(2*M_PI/sub_prob_size), sin(2*M_PI/sub_prob_size));
		for(i=0; i<sub_probs; ++i) {
			j_begin = i*sub_prob_size;
			j_end = j_begin+half;
			w = std::complex<double>(1,0);
			for(j=j_begin; j<j_end; ++j) {
				tmp1 = x[j];
				tmp2 = w*x[j+half];
				x[j] = tmp1+tmp2;
				x[j+half] = tmp1-tmp2;
				w *= w_step;
			}
		}
		sub_probs >>= 1;
		half = sub_prob_size;
	}
	for(uint32_t i=0; i<n; ++i) {
		x[i] /= n;
	}
}
void Image::dft_2D(uint32_t n, uint32_t m, std::complex<double> x[], std::complex<double>* X){
	//x in row-major & standard order
	std::complex<double>* intermediate = new std::complex<double>[m*n];
	//rows
	for(uint32_t i=0; i<m; ++i) {
		fft(n, x+i*n, intermediate+i*n);
	}
	//cols
	for(uint32_t j=0; j<n; ++j) {
		for(uint32_t i=0; i<m; ++i) {
			X[j*m+i] = intermediate[i*n+j]; //row-major --> col-major
		}
		fft(m, X+j*m, X+j*m);
	}
	delete[] intermediate;
	//X in column-major & bit-reversed (in rows then columns)
}
void Image::idft_2D(uint32_t n, uint32_t m, std::complex<double> X[], std::complex<double>* x){
	//X in column-major & bit-reversed (in rows then columns)
	std::complex<double>* intermediate = new std::complex<double>[m*n];
	//cols
	for(uint32_t j=0; j<n; ++j) {
		ifft(m, X+j*m, intermediate+j*m);
	}
	//rows
	for(uint32_t i=0; i<m; ++i) {
		for(uint32_t j=0; j<n; ++j) {
			x[i*n+j] = intermediate[j*m+i]; //row-major <-- col-major
		}
		ifft(n, x+i*n, x+i*n);
	}
	delete[] intermediate;
}
void Image::pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker) {
	//padded so center of kernel is at top left
    int a = ker_h-cr;
    int b = ker_w-cc;
	for(long i=-((int)cr); i<a; ++i) {
		uint32_t r = (i<0) ? i+ph : i;
		for(long j=-((int)cc); j<b; ++j) {
			uint32_t c = (j<0) ? j+pw : j;
			pad_ker[r*pw+c] = std::complex<double>(ker[(i+cr)*ker_w+(j+cc)], 0);
		}
	}
}
void Image::pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p) {
	for(uint64_t k=0; k<l; ++k) {
		p[k] = a[k]*b[k];
	}
}
Image& Image::fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
    // Calculate padding
    uint32_t pw = 1 << ((uint8_t)ceil(log2(w + ker_w - 1)));
    uint32_t ph = 1 << ((uint8_t)ceil(log2(h + ker_h - 1)));
    uint64_t psize = pw * ph;

    try {
        // Pad image
        std::complex<double>* pad_img = new std::complex<double>[psize];
        for (uint32_t i = 0; i < h; ++i) {
            for (uint32_t j = 0; j < w; ++j) {
                pad_img[i * pw + j] = std::complex<double>(data[(i * w + j) * channels + channel], 0);
            }
        }

        // Pad kernel
        std::complex<double>* pad_ker = new std::complex<double>[psize];
        pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

        // Convolution
        dft_2D(ph, pw, pad_img, pad_img);
        dft_2D(ph, pw, pad_ker, pad_ker);
        pointwise_product(psize, pad_img, pad_ker, pad_img);
        idft_2D(ph, pw, pad_img, pad_img);

        // Update pixel data
        for (uint32_t i = 0; i < h; ++i) {
            for (uint32_t j = 0; j < w; ++j) {
                data[(i * w + j) * channels + channel] = BOUND((uint8_t)round(pad_img[i * pw + j].real()));
            }
        }

        delete[] pad_img;
        delete[] pad_ker;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Cant convolve, Memory allocation failed: " << e.what() << std::endl<<"Tip: recommended Image resolution is less than 4K\n";
        throw; // Re-throw the exception after logging
    }

    return *this;
}
Image& Image::std_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	uint8_t *new_data = new uint8_t[size];
	uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
        int a= ker_h -cr;
        int b = ker_w-cc;
		for(long i = -((int)cr); i<a; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0 || row > h-1) {
				continue;
			}
			for(long j = -((int)cc); j<b; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0 || col > w-1) {
					continue;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = (uint8_t)BOUND(round(c));
	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
    delete[] new_data;
	return *this;
}
Image& Image::convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
	else {
		return std_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
}

Image& Image::applykernel(uint8_t channels, uint8_t ker_w, uint8_t ker_h, double ker[], bool copy){
    if(!copy){
        for(uint8_t i = 0; i<channels ; ++i){
            convolve_linear(i, ker_w, ker_h, ker, ker_w/2, ker_h/2);
        }
        return *this;
    }
    else{
        Image* cpy = new Image(*this);
        for(uint8_t i = 0; i<channels ; ++i){
            cpy->convolve_linear(i, ker_w, ker_h, ker, ker_w/2, ker_h/2);
        }
        return *cpy;
    }
}
Image& Image::gaussianBlur( int x1, int y1, int x2, int y2, bool copy){
    // Kernel for Gaussian Blur 
    double ker[] = {1/16.0, 2/16.0, 1/16.0, 
                    2/16.0, 4/16.0, 2/16.0, 
                    1/16.0, 2/16.0, 1/16.0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.applykernel(3, 3 ,3, ker, false);
        
        Image* finna = new Image(*this);
        Image* res = &finna->overlay(cpy, x1, y1, true);
        return *res;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}
Image& Image::Blur(uint8_t scale,int x1, int y1, int x2, int y2, bool copy){
    // Box-Blur Kernel for variable Amount of blur
    if(scale<5||scale>15){
        printf("[Error] Please Provide the scale between 5 and 15\nCan't Blur The image\n");
        return *this;
    }
    double ker[scale*scale];
    for(int i = 0; i<scale*scale; ++i){ker[i] = 1.0/((double)scale*scale);}
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.applykernel(3, scale ,scale, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.applykernel(3, scale ,scale, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}

Image& Image::emboss( int x1, int y1, int x2, int y2,bool want_grayscale ,bool copy){
    // Kernel for Emboss
    double ker[] = {-2/9.0, -1/9.0, 0, 
                    -1/9.0, 1/9.0, 1/9.0, 
                    0, 1/9.0, 2/9.0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        if(want_grayscale){cpy.grayscale();}
        cpy.applykernel(3, 3 ,3, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        if(want_grayscale){cpy.grayscale();}
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}

Image& Image::sharpen( int x1, int y1, int x2, int y2,bool want_grayscale ,bool copy){
    // Kernel for Sharpen
    double ker[] = {0, -1.0, 0, 
                    -1.0, 5.0,-1.0, 
                    0, -1.0, 0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        if(want_grayscale){cpy.grayscale();}
        cpy.applykernel(3, 3 ,3, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        if(want_grayscale){cpy.grayscale();}
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}

Image& Image::TSobel( int x1, int y1, int x2, int y2 ,bool copy){
    // Kernel for Top Sobel
    double ker[] = {1.0, 2.0, 1.0, 
                    0, 0,0, 
                    -1.0, -2.0, -1.0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.applykernel(3, 3 ,3, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}

Image& Image::BSobel( int x1, int y1, int x2, int y2 ,bool copy){
    // Kernel for Bottom Sobel
    double ker[] = {-1.0, -2.0, -1.0, 
                    0, 0,0, 
                    1.0, 2.0, 1.0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.applykernel(3, 3 ,3, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}

Image& Image::create_grayscale_outline(int x1, int y1, int x2, int y2 ,bool copy){
    // Kernel for Outline
    double ker[] = {-1.0, -1.0, -1.0, 
                    -1.0, 10,-1.0, 
                    -1.0, -1.0, -1.0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.applykernel(3, 3 ,3, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }
}
Image& Image::laplace_of_gaussian(int x1, int y1, int x2, int y2, bool copy ){
    double ker[] = {0.0, -1.0, 0.0,
                    -1.0, 4.0, -1.0,
                    0.0, -1.0, 0.0};
    if(copy){
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.gaussianBlur(x1,y1,x2,y2,false);
        cpy.applykernel(3, 3 ,3, ker, false);
        Image* finna = new Image(*this);
        finna->overlay(cpy, x1, y1, false);
        return *finna;
    }
    else{
        Image cpy = crop(x1, y1, x2, y2, true);
        cpy.grayscale();
        cpy.gaussianBlur(x1,y1,x2,y2,false);
        cpy.applykernel(3, 3 ,3, ker, false);
        overlay(cpy,x1, y1, false);
        return *this;
    }               
}

Image& Image::get_Contour(bool rgbedges){
    // Using Sobel-like Operators (Scharr)
    // Logic : https://en.wikipedia.org/wiki/Sobel_operator
    Image* img = new Image(*this);
    // grayscale
	img->grayscale_avg();
	int img_size = img->w*img->h;

	Image* gray_img = new Image(img->w, img->h, 1);
	for(uint64_t k=0; k<img_size; ++k) {
		gray_img->data[k] = img->data[img->channels*k];
	}
	// blur
	Image* blur_img = new Image(img->w, img->h, 1);
	double gauss[9] = {
		1/16., 2/16., 1/16.,
		2/16., 4/16., 2/16.,
		1/16., 2/16., 1/16.
	};
	gray_img->convolve_linear(0, 3, 3, gauss, 1, 1);
	for(uint64_t k=0; k<img_size; ++k) {
		blur_img->data[k] = gray_img->data[k];
	}
	// edge detection
	double* tx = new double[img_size];
	double* ty = new double[img_size];
	double* gx = new double[img_size];
	double* gy = new double[img_size];
	//seperable convolution
	for(uint32_t c=1; c<blur_img->w-1; ++c) {
		for(uint32_t r=0; r<blur_img->h; ++r) {
			tx[r*blur_img->w+c] = blur_img->data[r*blur_img->w+c+1] - blur_img->data[r*blur_img->w+c-1];
			ty[r*blur_img->w+c] = 47*blur_img->data[r*blur_img->w+c+1] + 162*blur_img->data[r*blur_img->w+c] + 47*blur_img->data[r*blur_img->w+c-1];
		}
	}
	for(uint32_t c=1; c<blur_img->w-1; ++c) {
		for(uint32_t r=1; r<blur_img->h-1; ++r) {
			gx[r*blur_img->w+c] = 47*tx[(r+1)*blur_img->w+c] + 162*tx[r*blur_img->w+c] + 47*tx[(r-1)*blur_img->w+c];
			gy[r*blur_img->w+c] = ty[(r+1)*blur_img->w+c] - ty[(r-1)*blur_img->w+c];
		}
	}
	delete[] tx;
	delete[] ty;
    free(blur_img->data);
    free(gray_img->data);
	//make test images
	double mxx = -INFINITY,
		mxy = -INFINITY,
		mnx = INFINITY,
		mny = INFINITY;
	for(uint64_t k=0; k<img_size; ++k) {
		mxx = fmax(mxx, gx[k]);
		mxy = fmax(mxy, gy[k]);
		mnx = fmin(mnx, gx[k]);
		mny = fmin(mny, gy[k]);
	}
	Image Gx(img->w, img->h, 1);
	Image Gy(img->w, img->h, 1);
	for(uint64_t k=0; k<img_size; ++k) {
		Gx.data[k] = (uint8_t)(255*(gx[k]-mnx)/(mxx-mnx));
		Gy.data[k] = (uint8_t)(255*(gy[k]-mny)/(mxy-mny));
    }
	double threshold = 0.09;
	double* g = new double[img_size];
	double* theta = new double[img_size];
	double x, y;
	for(uint64_t k=0; k<img_size; ++k) {
		x = gx[k];
		y = gy[k];
		g[k] = sqrt(x*x + y*y);
		theta[k] = atan2(y, x);
	}
	//make images
	double mx = -INFINITY,
		mn = INFINITY;
	for(uint64_t k=0; k<img_size; ++k) {
		mx = fmax(mx, g[k]);
		mn = fmin(mn, g[k]);
	}
	Image* G = new Image(img->w, img->h, 1);
	Image* GT = new Image(img->w, img->h, 3);
	double h, s, l;
	double v;
	for(uint64_t k=0; k<img_size; ++k) {
		//theta to determine hue
		h = theta[k]*180./M_PI + 180.;
		//v is the relative edge strength
		if(mx == mn) {
			v = 0;
		}
		else {
			v = (g[k]-mn)/(mx-mn) > threshold ? (g[k]-mn)/(mx-mn) : 0;
		}
		s = l = v;
		//hsl => rgb
		double c = (1-abs(2*l-1))*s;
		double x = c*(1-abs(fmod((h/60),2)-1));
		double m = l-c/2;

		double rt, gt, bt;
		rt=bt=gt = 0;
		if(h < 60) {
			rt = c;
			gt = x;
		}
		else if(h < 120) {
			rt = x;
			gt = c;
		}
		else if(h < 180) {
			gt = c;
			bt = x;
		}
		else if(h < 240) {
			gt = x;
			bt = c;
		}
		else if(h < 300) {
			bt = c;
			rt = x;
		}
		else {
			bt = x;
			rt = c;
	    }
		uint8_t red, green, blue;
		red = (uint8_t)(255*(rt+m));
		green = (uint8_t)(255*(gt+m));
		blue = (uint8_t)(255*(bt+m));

		GT->data[k*3] = red;
		GT->data[k*3+1] = green;
		GT->data[k*3+2] = blue;
		G->data[k] = (uint8_t)(255*v);
	}

	delete[] gx;
	delete[] gy;
	delete[] g;
	delete[] theta;
    if(rgbedges){
        return *GT;
    }
    else{
        return *G;
    }
}
Image& Image::change_Brightness(double factor){
    // Refrence: https://hackernoon.com/image-processing-algorithms-adjusting-contrast-and-image-brightness-0y4y318a
    if(factor >1 || factor < -1){
        printf("[ERROR] Couldnt change brightness, Invalid factor value, it should be in between -1.0 and 1.0\n");
        return *this;}
    Image* cpy = new Image(w ,h, channels);
    for(uint32_t i = 0; i<w*h*channels; i+=channels){
        uint8_t r,g,b;
        r = data[i];
        g= data[i+1];
        b = data[i+2];
        uint8_t newr = BOUND((r*(1+factor)));
        uint8_t newg = BOUND((g*(1+factor)));
        uint8_t newb = BOUND((b*(1+factor)));
        cpy->data[i] = newr;cpy->data[i+1] = newg; cpy->data[i+2]= newb;
    }
    return *cpy;
}

bool comp(std::pair<uint8_t, int> &p1,std::pair<uint8_t, int> &p2){
    return p1.first < p2.first;
}
Image& Image::increase_Contrast(){
    // Refrence: https://hackernoon.com/image-processing-algorithms-adjusting-contrast-and-image-brightness-0y4y318a
    Image* cpy = new Image(*this);
    cpy->grayscale();
    std::map<uint8_t, int> freq;
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k];
        freq[intensity]++;
    }
    std::vector<std::pair<uint8_t, int>> mp;
    for(auto i:freq){
        mp.push_back(std::make_pair(i.first, i.second));
    }
    sort(mp.begin(), mp.end(), comp);
    std::vector<int> ifreq;
    for(auto i:mp){
        ifreq.push_back(i.second);
    }
    std::vector<int> cdf;
    int csum = 0;
    for(auto i:ifreq){
        csum += i;
        cdf.push_back(csum);
    }
    int min_ = cdf[0], max_ = cdf[cdf.size()-1];
    std::vector<int> Normalizedcdf;
    for(auto i:cdf){
        Normalizedcdf.push_back(((i-min_)*255)/(max_-min_));
    }
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k];
        uint8_t outIntensity = Normalizedcdf[intensity];
        cpy->data[k] = outIntensity;
        cpy->data[k+1] = outIntensity;
        cpy->data[k+2] = outIntensity;
    }
    return *cpy;
}

Image& Image::increase_Contrast_RGB(){
    // Refrence: https://hackernoon.com/image-processing-algorithms-adjusting-contrast-and-image-brightness-0y4y318a
    Image* cpy = new Image(*this);
    // for k(red)
    std::map<uint8_t, int> freq;
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k];
        freq[intensity]++;
    }
    std::vector<std::pair<uint8_t, int>> mp;
    for(auto i:freq){
        mp.push_back(std::make_pair(i.first, i.second));
    }
    sort(mp.begin(), mp.end(), comp);
    std::vector<int> ifreq;
    for(auto i:mp){
        ifreq.push_back(i.second);
    }
    std::vector<int> cdf;
    int csum = 0;
    for(auto i:ifreq){
        csum += i;
        cdf.push_back(csum);
    }
    int min_ = cdf[0], max_ = cdf[cdf.size()-1];
    std::vector<int> Normalizedcdf;
    for(auto i:cdf){
        Normalizedcdf.push_back(((i-min_)*255)/(max_-min_));
    }
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k];
        uint8_t outIntensity = Normalizedcdf[intensity];
        cpy->data[k] = outIntensity;
    }
    // for k+1(green)
    freq.clear();
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k+1];
        freq[intensity]++;
    }
    mp.clear();
    for(auto i:freq){
        mp.push_back(std::make_pair(i.first, i.second));
    }
    sort(mp.begin(), mp.end(), comp);
    ifreq.clear();
    for(auto i:mp){
        ifreq.push_back(i.second);
    }
    cdf.clear();
    csum = 0;
    for(auto i:ifreq){
        csum += i;
        cdf.push_back(csum);
    }
    min_ = cdf[0]; max_ = cdf[cdf.size()-1];
    Normalizedcdf.clear();
    for(auto i:cdf){
        Normalizedcdf.push_back(((i-min_)*255)/(max_-min_));
    }
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k+1];
        uint8_t outIntensity = Normalizedcdf[intensity];
        cpy->data[k+1] = outIntensity;
    }
    // for k+2(blue)
    freq.clear();
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k+2];
        freq[intensity]++;
    }
    mp.clear();
    for(auto i:freq){
        mp.push_back(std::make_pair(i.first, i.second));
    }
    sort(mp.begin(), mp.end(), comp);
    ifreq.clear();
    for(auto i:mp){
        ifreq.push_back(i.second);
    }
    cdf.clear();
    csum = 0;
    for(auto i:ifreq){
        csum += i;
        cdf.push_back(csum);
    }
    min_ = cdf[0]; max_ = cdf[cdf.size()-1];
    Normalizedcdf.clear();
    for(auto i:cdf){
        Normalizedcdf.push_back(((i-min_)*255)/(max_-min_));
    }
    for(uint32_t k = 0; k<w*h*channels; k+=channels){
        uint8_t intensity = data[k+2];
        uint8_t outIntensity = Normalizedcdf[intensity];
        cpy->data[k+2] = outIntensity;
    }

    return *cpy;
}

// void Image::imshow_slow(){
//     if(channels < 3){
//         std::cout<<"[Error] Cannot Show Images with channels not equal to 3 (for grayscale try loading the image with three channels)";
//     }
//     initwindow(w,h,"Image", 0,0, false,true);
//     int cntr = 0;
//     for(int y = 0; y<h; ++y){
//         for(int x = 0; x<w; ++x){
//             putpixel(x,y,COLOR(data[cntr], data[cntr+1], data[cntr+2]));
//             cntr += channels;
//         }
//     }
//     getch();
//     closegraph();
// }
Image& Image::framify_color(uint8_t r, uint8_t g, uint8_t b, int thickness){
    if(thickness <= 0){
        printf("Please Given an appropriate value of thickness\n");
        return *this;
    }
    Image* cpy = new Image(*this);
    *cpy = cpy->draw_rect(0,0,cpy->getWidth(), cpy->getHeight(), r, g, b, thickness);
    return *cpy;
}

Image& Image::detect_color(uint8_t lower_color_bound[3], uint8_t upper_color_bound[3]){
    Image* cpy = new Image(w,h,1);
    uint32_t cntr = 0;
    for(uint32_t k = 0; k<w*h*channels; k += channels){
        if((data[k]<=upper_color_bound[0]  && data[k]>=lower_color_bound[0]) && (data[k+1]<=upper_color_bound[1]  && data[k+1]>=lower_color_bound[1]) && (data[k+2]<=upper_color_bound[2]  && data[k+2]>=lower_color_bound[2])){
            cpy->data[cntr++] = 255;
        }
        else{
            cpy->data[cntr++] = 0;
        }
        
    }
    return *cpy;
}

// Draw On Image

Image& Image::draw_rect(int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, uint16_t thickness, bool fill){
    if(x2>w || y2>h || x1<0 || y1<0){
        std::cout<<"[ERROR]Region Selected is out of bounds of the image.\n";
        return *this;
    }
    Image* cpy = new Image(*this);
    if(fill){for(int i = y1; i<y2; ++i){
            for(int j = x1*channels + i*w*channels; j< x2*channels + i*w*channels;j+=channels){
                cpy->data[j] = r;
                cpy->data[j+1] = g;
                cpy->data[j+2] = b;
            }
        }
        // cpy->gaussianBlur(0,0,w,h,false);
        return *cpy;
        }
    else{
        for(int i = y1; i<y2; ++i){
            for(int j = x1*channels + i*w*channels; j< x2*channels + i*w*channels;j+=channels){
                if(j<x1*channels + i*w*channels + thickness*channels || j>x2*channels + i*w*channels-thickness*channels){
                    cpy->data[j] = r;
                    cpy->data[j+1] = g;
                    cpy->data[j+2] = b;
                }
                else if(i < y1 + thickness || i> y2 -thickness){
                    cpy->data[j] = r;
                    cpy->data[j+1] = g;
                    cpy->data[j+2] = b;
                }
            }
        }
        // cpy->gaussianBlur(0,0,w,h,false);
        return *cpy;
    }


}

Image& Image::draw_circle(int x, int y, int radius, uint8_t r, uint8_t g, uint8_t b, uint16_t thickness, bool fill){
    if(x<0 || x > w || y <0 || y>h){
        std::cout<<"[ERROR]Region Selected is out of bounds of the image.\n";
        return *this;
    }
    Image* cpy = new Image(*this);
    if(!fill){
    for(uint32_t k = 0; k<w*h*channels; k += channels){
        int col = (k/channels)%w;
        int row = (k/channels)/w + 1;
        
        float distance = sqrt(pow((row-y),2)+pow((col-x),2));
        // std::cout<<row<<" "<<col<<" "<<distance<<std::endl;
        if(distance< radius+thickness && distance > radius - thickness){
            cpy->data[k] = r;
            cpy->data[k+1] = g;
            cpy->data[k+2] = b;
        }
    }
    cpy->gaussianBlur(0,0,w,h,false);
    return *cpy;
    }
    else{
    for(uint32_t k = 0; k<w*h*channels; k += channels){
        int col = (k/channels)%w;
        int row = (k/channels)/w + 1;
        float distance = sqrt(pow((row-y),2)+pow((col-x),2));
        if(distance< radius){
            cpy->data[k] = r;
            cpy->data[k+1] = g;
            cpy->data[k+2] = b;
        }
    }
    cpy->gaussianBlur(0,0,w,h,false);
    return *cpy;
    }
}

Image& Image::draw_line(int x1,int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, uint16_t thickness){
    if(x2>w || y2>h || x1<0 || y1<0){
        std::cout<<"[ERROR]Region Selected is out of bounds of the image.\n";
        return *this;
    }
    Image* cpy = new Image(*this);
    if(x1 != x2){
        double m = (float)(y2-y1)/(float)(x2-x1);
        double c = -m*x1 + y1;
        for(uint32_t k = 0; k<w*h*channels; k += channels){
            int col = (k/channels)%w;
            int row = (k/channels)/w + 1;
            if(col > fmax(x1,x2) || row > fmax(y1,y2) || row < fmin(y1, y2) || col < fmin(x1,x2)){
                continue;
            }
            float distance = fabs((row - m*col - c)/(double)sqrt(1+pow(m,2)));
            if(distance< thickness){
                cpy->data[k] = r;
                cpy->data[k+1] = g;
                cpy->data[k+2] = b;
            }
        }
        cpy->gaussianBlur(0,0,w,h,false);
        return *cpy;
    }
    else{
        for(uint32_t k = 0; k<w*h*channels; k += channels){
            int col = (k/channels)%w;
            int row = (k/channels)/w + 1;
            if(row > fmax(y1,y2) || row < fmin(y1, y2)){
                continue;
            }
            float distance = fabs(col-x1);
            // std::cout<<row<<" "<<col<<" "<<distance<<std::endl;
            if(distance< thickness){
                cpy->data[k] = r;
                cpy->data[k+1] = g;
                cpy->data[k+2] = b;
            }
        }
        cpy->gaussianBlur(0,0,w,h,false);
        return *cpy;
    }
}
uint8_t* interpolate_pixel(Image& img, int points[3][2], int axis = 1){
    int p1[2],p2[2], p3[2];
    p1[0] = points[0][0];
    p1[1] = points[0][1];
    p2[0] = points[1][0];
    p2[1] = points[1][1];
    p3[0] = points[2][0];
    p3[1] = points[2][1];
    int d1 = p3[axis]-p1[axis];
    int d2 = p2[axis]-p3[axis];
    uint8_t* pixel_1 = &img.data[(p1[1] + p1[0]*img.w)*img.channels];
    uint8_t* pixel_2 = &img.data[(p2[1] + p2[0]*img.w)*img.channels];
    uint8_t* pixel_res = new uint8_t[3];
    for(int i = 0 ; i<3; ++i){
        pixel_res[i] = (uint8_t)(pixel_1[i]*d2 + pixel_2[i]*d1);
    }
    return pixel_res;
}

// Refrences :
// https://en.wikipedia.org/wiki/Image_scaling
// https://en.wikipedia.org/wiki/Bilinear_interpolation
// https://raghul-719.medium.com/basics-of-computer-vision-1-image-resizing-97fca504cd63
Image& Image::resize( int new_w, int new_h){
    if(channels != 3){
        printf("[My Error] Resizing currently supported by 3 channles images only.\n");
        return *this;
    }
    Image* cpy = new Image(new_w, new_h, channels);
    double mapped_w[new_w], mapped_h[new_h];
    for(int i = 0; i<new_w; ++i){
        double target = (double)i*((double)w/(double)new_w);
        if(target <=w-1){
            mapped_w[i] = target;
        }
        else{
            mapped_w[i] = w-1;
        }
    }
    for(int i = 0; i<new_h; ++i){
        double target = (double)i*((double)h/(double)new_h);
        if(target <=h-1){
            mapped_h[i] = target;
        }
        else{
            mapped_h[i] = h-1;
        }
    }
    for(int i = 0; i<new_h; ++i){
        for(int j = 0; j<new_w; ++j){
            int mi = mapped_h[i];
            int mj = mapped_w[j];
            if(mi == std::floor(mi) && mj == std::floor(mj)){
                uint8_t* srcpx = &cpy->data[(j+i*new_w)*channels];
                uint8_t* finna_px = &data[((int)mj+(int)mi*w)*channels];
                memcpy(srcpx, finna_px, channels);
            }
            else if(mi == std::floor(mi)){
                int i1 = (int)mi;int j1 = (int)mj;
                int i2 = (int)mi;
                int j2 = (int)fmin((int)mj+1, w-1);
                int points[3][2] = {{i1, j1}, {i2, j2}, {i1, mj}};
                uint8_t* res=  interpolate_pixel(*this, points , 1);
                uint8_t* srcpx = &cpy->data[(j+i*new_w)*channels];
                memcpy(srcpx, res, channels);
            }
            else if(mj == std::floor(mj)){
                int i1 = (int)mi;int j1 = (int)mj;
                int i2 = (int)fmin((int)mi + 1, h-1);
                int j2 = (int)mj;
                int points[3][2] = {{i1, j1}, {i2, j2}, {mi, j2}};
                uint8_t* res=  interpolate_pixel(*this, points , 0);
                uint8_t* srcpx = &cpy->data[(j+i*new_w)*channels];
                memcpy(srcpx, res, channels);
            }
            else{
                int i1,j1,i2,j2,i3,j3,i4,j4;
                i1 = (int)mi;j1 = (int)mj;
                i2 = (int)mi;j2 = (int)fmin((int)mj+1, w-1);
                i3 = (int)fmin((int)mi+1, h-1);j3 = (int)mj;
                i4 = (int)fmin((int)mi+1, h-1);j4 = (int)fmin((int)mj+1, w-1);
                int points1[3][2] = {{i1, j1}, {i2, j2}, {i1, mj}};
                int points2[3][2] = {{i3, j3}, {i4, j4}, {i3, mj}};
                uint8_t* pxl_1 = interpolate_pixel(*this, points1, 1);
                uint8_t* pxl_2 = interpolate_pixel(*this, points2, 1);
                int dy1 = mi -i1;int dy2 = i3-mi;
                uint8_t* pixel_res = new uint8_t[3];
                for(int _ = 0; _<3; _++){
                    pixel_res[_] = (uint8_t)(pxl_1[_]*dy2 + pxl_2[_]*dy1);
                }
                uint8_t* srcpx = &cpy->data[(j+i*new_w)*channels];
                memcpy(srcpx, pixel_res, channels);
            }

        }
    }

    return *cpy;
}