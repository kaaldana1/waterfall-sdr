#define _GNU_SOURCE
#include <rtl-sdr.h>
#include <SDL3/SDL.h>
#include <error.h>
#include <stdio.h>
#include <stdbool.h>
#include <libusb-1.0/libusb.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <unistd.h>

/*
==================================================
Frequency resolution of an FFT
==================================================
*/
#define GUI_WINDOW_TITLE "waterfall_sdr"
#define GUI_WINDOW_WIDTH 800 
#define GUI_WINDOW_HEIGHT 100
#define GUI_TEXTURE_HEIGHT (GUI_WINDOW_HEIGHT - 1)

#define POOL_CAPACITY 16
#define RB_CAPACITY 16
#define SAMPLES_BUFFER_SIZE (16 * 1024) /* 16KB */

#define SPIT_LIBUSB_ERROR(err_code)  \
	do { \
		fprintf(stderr, "Error: %s\n", libusb_error_name((err_code))); \
		return (err_code); \
	} while (0)

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) > (B) ? (A) : (B))

typedef enum 
{
	QUEUE_ENQ_FAIL = -17,
	DSP_MISSING_IQ_PAIRS = -16,
	DSP_NOT_RADIX_2 = -15,
	DSP_MISMATCH_FFT_SIZE = -14,
	DSP_BAD_ARGS = -13,
	DSP_FFT_FAILED = -12,
	DSP_SIZE_LE_ZERO = -11,
	GUI_FAILED_START_WINDOW = -10,
	GUI_FAILED_START_RENDERER = -9,
	GUI_FAILED_START_TEXTURER = -8, 
	GUI_FAILED_GET_PRODUCT = -7,
	GUI_FAILED_UPDATE_TEXTURE = -6,
} CustomError;

const char *trgt_device_name = "RTL2838UHIDIR";

typedef struct
{
	int sample_rate;
	int center_freq;
	int tuner_gain_mode;
	int tuner_gain;
	int fft_size;
} DevConfig;

typedef struct 
{
	rtlsdr_dev_t *dev_ptr;
	DevConfig *config;

	char *manufacturer;
	char *product;
	char *serial;
	int index;
} DevTarget;


typedef enum
{
	DEFAULT_SAMPLE_RATE = 2048000,
	DEFAULT_CENTER_FREQ = 433920000,
	DEFAULT_TUNER_GAIN_MODE = 1,
	DEFAULT_TUNER_GAIN = 400, 
	DEFAULT_FFT_SIZE = (SAMPLES_BUFFER_SIZE / 2),
} DevConfigDefaults;


pthread_t p_producer_read;
pthread_t p_consumer_dsp;

pthread_mutex_t p_mutex_usb_rb_filled;
pthread_mutex_t p_mutex_usb_rb_free;
sem_t p_sem_usb_rb_full; /* count of filled blocks */
sem_t p_sem_usb_rb_empty; /* count of free blocks */

pthread_mutex_t p_mutex_dsp_rb_filled;
pthread_mutex_t p_mutex_dsp_rb_free;
sem_t p_sem_dsp_rb_full; /* count of filled blocks */
sem_t p_sem_dsp_rb_empty; /* count of free blocks */

/* ring buffer */
typedef struct 
{
	uint8_t *usb_samples_buffer;
	int usb_n_samples; /* actual samples collected*/
} USBBlock;

const int rb_capacity = (int)RB_CAPACITY;

/*
In this setup:
	the head is first item in -> popped
	the tail is last item in -> pushed
*/
const int usb_buffer_sample_size = (int)SAMPLES_BUFFER_SIZE; /* 16KB */
uint8_t usb_samples_buffers[RB_CAPACITY][SAMPLES_BUFFER_SIZE] = {0};
USBBlock usb_blocks[POOL_CAPACITY] = {0};
USBBlock *usb_rb_filled[POOL_CAPACITY];
int usb_rb_filled_head = 0;
int usb_rb_filled_tail = 0;

USBBlock *usb_rb_free[POOL_CAPACITY];
int usb_rb_free_head = 0;
int usb_rb_free_tail = (RB_CAPACITY - 1); /* ring keeps one slot empty */

/* dsp results */
typedef struct 
{
	double complex *FFTs_per_usb_block;
	double *powers;
	double *decibels;
	double min_db_per_frame[30];
	double max_db_per_frame[30];
} DSPProducts;

/* another pool */

double complex dsp_FFTs_per_usb_block[POOL_CAPACITY][SAMPLES_BUFFER_SIZE / 2];
double dsp_powers[POOL_CAPACITY][SAMPLES_BUFFER_SIZE / 2];
double dsp_decibels[POOL_CAPACITY][SAMPLES_BUFFER_SIZE / 2];

DSPProducts dsp_products[POOL_CAPACITY];
DSPProducts *dsp_rb_filled[POOL_CAPACITY];
int dsp_rb_filled_head = 0;
int dsp_rb_filled_tail = 0;

DSPProducts *dsp_rb_free[POOL_CAPACITY];
int dsp_rb_free_head = 0;
int dsp_rb_free_tail = (RB_CAPACITY - 1);


typedef struct 
{
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *texture;
	SDL_PixelFormatDetails format_details;
} GUI_SDL;

uint32_t gui_sdl_palette[256];
uint32_t rgba_pixel_buff[SAMPLES_BUFFER_SIZE / 2] = {0}; 

int running = 1;

void print_waterfall_info(DevConfig *);
void *p_read_routine(void *);
void *p_consumer_dsp_routine(void *);

int dev_open(DevTarget *);
int dev_configure(rtlsdr_dev_t *, int sample_rate, int center_freq, int tuner_gain_mode, int tuner_gain);
int DSP(USBBlock *, DevTarget *, DSPProducts *);
double complex DSP_iq_to_complex(uint8_t I_sample, uint8_t Q_sample);
int DSP_FFT_in_place(double complex *x, int fft_size, int sign);
int DSP_reverse_bit(int i, int bits);
int DSP_hann_window(double complex *x, int fft_size);
double DSP_calculate_power(double complex dft_coeff);
DSPProducts *gui_sdl_get_products();

int gui_sdl_return_products(DSPProducts *products);
int gui_sdl_db_to_rgba(DevConfig *config, DSPProducts *products, int start, uint32_t *rgba_work_buffer, 
					   double max_db, double min_db);
int gui_sdl_render_row(GUI_SDL *g, DevConfig *config, uint32_t *rgba_buffer_per_fft_frame, int row_scroll_pos);
int gui_sdl_destroy(GUI_SDL *g);
int gui_sdl_init(GUI_SDL *g, DevConfig *config);
int gui_sdl_waterfall(GUI_SDL *g, DevConfig *config);

int main(int argc, char **argv)
{
	int r = 0;

	DevConfig config = {
		.sample_rate = DEFAULT_SAMPLE_RATE,
		.center_freq = DEFAULT_CENTER_FREQ,
		.tuner_gain_mode = DEFAULT_TUNER_GAIN_MODE,
		.tuner_gain = DEFAULT_TUNER_GAIN, 
		.fft_size = DEFAULT_FFT_SIZE
	};

	int i = 0;
	while (i < argc)
	{
		if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) { i++; config.sample_rate = atoi(argv[i]); }
		else if (strcmp(argv[i], "--center-freq") == 0 && i + 1 < argc) { i++; config.center_freq = atoi(argv[i]); }
		else if (strcmp(argv[i], "--tuner-gain-mode") == 0 && i + 1 < argc) { i++; config.tuner_gain_mode = atoi(argv[i]); }
		else if (strcmp(argv[i], "--tuner-gain") == 0 && i + 1 < argc) { i++; config.tuner_gain = atoi(argv[i]); }
		else if (strcmp(argv[i], "--fft_size") == 0 && i + 1 < argc) { i++; config.fft_size = atoi(argv[i]); }
		i++;
	}

	print_waterfall_info(&config);
	
	
	DevTarget dev;
	dev.config = &config;

	r = dev_open(&dev);
	if (r < 0) { SPIT_LIBUSB_ERROR(r); }

	r = dev_configure(dev.dev_ptr, config.sample_rate, config.center_freq, config.tuner_gain_mode, config.tuner_gain);
	if (r < 0) { SPIT_LIBUSB_ERROR(r); }

	r = rtlsdr_reset_buffer(dev.dev_ptr);
	if (r < 0) { SPIT_LIBUSB_ERROR(r); }

	
	/* Initialize pools. Ring buffer design keeps one slot empty. */
	for (int i = 0; i < POOL_CAPACITY; i++)
	{
		usb_blocks[i].usb_samples_buffer = usb_samples_buffers[i];

		dsp_products[i].FFTs_per_usb_block = dsp_FFTs_per_usb_block[i];
		dsp_products[i].powers = dsp_powers[i];
		dsp_products[i].decibels = dsp_decibels[i];
		if (i < (rb_capacity - 1))
		{
			usb_rb_free[i] = &usb_blocks[i];
			dsp_rb_free[i] = &dsp_products[i];
		}
	}

	/* start producer thread (reading from rtl_sdr dongle)*/
	pthread_mutex_init(&p_mutex_usb_rb_filled, NULL);
	pthread_mutex_init(&p_mutex_usb_rb_free, NULL);
	pthread_mutex_init(&p_mutex_dsp_rb_filled, NULL);
	pthread_mutex_init(&p_mutex_dsp_rb_free, NULL);
	sem_init(&p_sem_usb_rb_full, 0, 0);
	sem_init(&p_sem_usb_rb_empty, 0, rb_capacity - 1);
	sem_init(&p_sem_dsp_rb_full, 0, 0);
	sem_init(&p_sem_dsp_rb_empty, 0, rb_capacity - 1);
	if (pthread_create(&p_producer_read, NULL, &p_read_routine, &dev) != 0) { printf("Failed to create producer thread\n"); goto cleanup; }
	if (pthread_create(&p_consumer_dsp, NULL, &p_consumer_dsp_routine, &dev) != 0) { printf("Failed to create consumer dsp thread\n"); goto cleanup; }
	
	GUI_SDL gui;
	r = gui_sdl_init(&gui, &config);
	if (r < 0) { return r; }
	r = gui_sdl_waterfall(&gui, &config);

	pthread_join(p_producer_read, NULL);
	pthread_join(p_consumer_dsp, NULL);

	cleanup:
	sem_destroy(&p_sem_usb_rb_full);
	sem_destroy(&p_sem_usb_rb_empty);
	pthread_mutex_destroy(&p_mutex_usb_rb_filled);
	pthread_mutex_destroy(&p_mutex_usb_rb_free);
	free(dev.manufacturer);
	free(dev.product);
	free(dev.serial);
	rtlsdr_close(dev.dev_ptr);
	exit(1);
}

void print_waterfall_info(DevConfig *config)
{
	int usb_complex_samples = usb_buffer_sample_size / 2;
	double usb_time_duration = config->sample_rate/usb_complex_samples;
	printf("==============================================================================\n");
	printf("Number of complex samples requested from USB: %d\n", usb_complex_samples);
	printf("==============================================================================\n");

	int frequency_resolution = config->sample_rate / config->fft_size;
	int ffts_per_usb_block = usb_complex_samples / config->fft_size;
	double fft_time_duration = (double)(config->fft_size) / (double)(config->sample_rate);

	printf("FFT size is: %d\n", config->fft_size);
	printf("==============================================================================\n");
	printf(" -> ONE USB sample will generate %d FFT/waterfall_rows\n", ffts_per_usb_block);
	printf("\t -> EACH FFT/waterfall_row will generate every %f ms\n", fft_time_duration);
	printf("\t -> EACH FFT/waterfall_row has %d frequency bins, or \n\t    %d discrete freq-components to be tested\n", config->fft_size, config->fft_size);
	printf("\t \t -> EACH FFT Bin width (frequency resolution) is: %dhz\n", frequency_resolution);
	printf("==============================================================================\n");
	printf("Note that:\n");
	printf("\tSMALLER FFT -> faster time updates, worse frequency resolution\n");
	printf("\tLARGER  FFT -> slower time updates, better frequency resolution\n");
	printf("==============================================================================\n");

	printf("Starting program...\n");
	sleep(1);
}

int dev_open(DevTarget *dev)
{
	uint32_t count = rtlsdr_get_device_count();
	if (count == 0) 
	{
		fprintf(stderr, "No supported devices found.\n");
		return -1;
	}
	printf("Found %u device(s):\n", count);

	char *manufacturer = malloc(256);
	char *product = malloc(256);
	char *serial = malloc(256);

	int r;
	int trgt_index;
	if (manufacturer == NULL || product == NULL || serial == NULL) return -1;

	for (int i = 0; i < count; i++)
	{
		printf("Device name: %s\n", rtlsdr_get_device_name(i));
		r = rtlsdr_get_device_usb_strings(i, manufacturer, product, serial);
		if (r < 0) return r;
		printf("Manufacturer: %s, Product: %s, Serial: %s\n", manufacturer, product, serial);
		if (strcmp(product, trgt_device_name) == 0)
		{
			trgt_index = i;
			break;
		}
	}

	dev->manufacturer = manufacturer;
	dev->product = product;
	dev->serial = serial;
	dev->index = trgt_index;

	r = rtlsdr_open(&dev->dev_ptr, trgt_index);
	if (r < 0) return r;
}

int dev_configure(rtlsdr_dev_t *dev, int sample_rate, int center_freq, int tuner_gain_mode, int tuner_gain)
{
    int r;
    r = rtlsdr_set_sample_rate(dev, sample_rate);
    if (r < 0) return r;

    r = rtlsdr_set_center_freq(dev, center_freq);
    if (r < 0) return r;

    r = rtlsdr_set_tuner_gain_mode(dev, tuner_gain_mode);
    if (r < 0) return r;

    r = rtlsdr_set_tuner_gain(dev, tuner_gain);
    if (r < 0) return r;

    return 0;
}

static inline int rb_enqueue(void **rb, const int head, int *tail, void *block)
{
	/* buffer is either full or empty when head == tail */
	int next = (*tail + 1) % rb_capacity;
	if (next == head)  /*buffer is full */ return QUEUE_ENQ_FAIL;
	
	rb[*tail] = block;
	*tail = next;
	return 0;
}

static inline void *rb_dequeue(void **rb, int *head, const int tail)
{
	if (*head == tail)  /* buffer is empty */ return NULL;
	
	void *block = rb[*head];
	*head = (*head + 1) % rb_capacity;
	return block;
}

void *p_read_routine(void *dev)
{
	printf("Reading...\n");
	DevTarget *target_dev = (DevTarget *)dev;
	rtlsdr_dev_t *dev_ptr = target_dev->dev_ptr;
	while (1) 
	{
		sem_wait(&p_sem_usb_rb_empty); /* free queue - 1 */
		pthread_mutex_lock(&p_mutex_usb_rb_free);
		USBBlock *block = (USBBlock*)rb_dequeue((void**)usb_rb_free, &usb_rb_free_head, usb_rb_free_tail);
		if (block == NULL) { printf("USB free buffer is empty\n"); pthread_mutex_unlock(&p_mutex_usb_rb_free); sem_post(&p_sem_usb_rb_empty); continue; }
		pthread_mutex_unlock(&p_mutex_usb_rb_free);

		if (block == NULL) { printf("nullptr \n"); continue; }

		uint8_t *usb_samples_buffer = block->usb_samples_buffer;
		int n_read;
		int r = rtlsdr_read_sync(dev_ptr, (void*)usb_samples_buffer, usb_buffer_sample_size, &n_read);
		if (r < 0) { return NULL; }
		if (n_read != SAMPLES_BUFFER_SIZE)
		{
			printf("Short read, dropping this sample\n");
			usb_samples_buffer[0] = 0;
			pthread_mutex_lock(&p_mutex_usb_rb_free);
			rb_enqueue((void**)usb_rb_free, usb_rb_free_head, &usb_rb_free_tail, (void*)block);
			pthread_mutex_unlock(&p_mutex_usb_rb_free);
			sem_post(&p_sem_usb_rb_empty);
			continue;
		}
		block->usb_n_samples = n_read;


		pthread_mutex_lock(&p_mutex_usb_rb_filled);
		r = rb_enqueue((void**)usb_rb_filled, usb_rb_filled_head, &usb_rb_filled_tail, (void*)block);
		pthread_mutex_unlock(&p_mutex_usb_rb_filled);
		if (r < 0)
		{
			pthread_mutex_lock(&p_mutex_usb_rb_free);
			rb_enqueue((void**)usb_rb_free, usb_rb_free_head, &usb_rb_free_tail, (void*)block);
			pthread_mutex_unlock(&p_mutex_usb_rb_free);
			sem_post(&p_sem_usb_rb_empty);
			continue;
		}
		sem_post(&p_sem_usb_rb_full); /* filled queue + 1*/
	}
	return NULL;
}

void *p_consumer_dsp_routine(void *device)
{
	DevTarget *dev = (DevTarget*)device;
	printf("Processing...\n");
	while(1)
	{
		sem_wait(&p_sem_usb_rb_full);
		pthread_mutex_lock(&p_mutex_usb_rb_filled);
		USBBlock *block = (USBBlock*)rb_dequeue((void**)usb_rb_filled, &usb_rb_filled_head, usb_rb_filled_tail);
		pthread_mutex_unlock(&p_mutex_usb_rb_filled); 

		if (block == NULL) { printf("nullptr \n"); sem_post(&p_sem_usb_rb_full); continue; }

		sem_wait(&p_sem_dsp_rb_empty);
		pthread_mutex_lock(&p_mutex_dsp_rb_free);
		DSPProducts *products = (DSPProducts*)rb_dequeue((void**)dsp_rb_free, &dsp_rb_free_head, dsp_rb_free_tail);
		if (products == NULL)
		{
			printf("DSP free buffer is empty\n");
			pthread_mutex_unlock(&p_mutex_dsp_rb_free);
			sem_post(&p_sem_dsp_rb_empty);
			pthread_mutex_lock(&p_mutex_usb_rb_filled);
			rb_enqueue((void**)usb_rb_filled, usb_rb_filled_head, &usb_rb_filled_tail, (void*)block);
			pthread_mutex_unlock(&p_mutex_usb_rb_filled);
			sem_post(&p_sem_usb_rb_full);
			continue;
		}
		pthread_mutex_unlock(&p_mutex_dsp_rb_free);

		if (products == NULL) { printf("nullptr \n"); continue; }

		int r = DSP(block, dev, products);
		if (r < 0)
		{
			printf("DSP processing failed with error code %d\n", r);
			pthread_mutex_lock(&p_mutex_dsp_rb_free);
			rb_enqueue((void**)dsp_rb_free, dsp_rb_free_head, &dsp_rb_free_tail, (void*)products);
			pthread_mutex_unlock(&p_mutex_dsp_rb_free);
			sem_post(&p_sem_dsp_rb_empty);
			pthread_mutex_lock(&p_mutex_usb_rb_free);
			rb_enqueue((void**)usb_rb_free, usb_rb_free_head, &usb_rb_free_tail, (void*)block);
			pthread_mutex_unlock(&p_mutex_usb_rb_free);
			sem_post(&p_sem_usb_rb_empty);
			continue;
		}

		pthread_mutex_lock(&p_mutex_dsp_rb_filled);
		r = rb_enqueue((void**)dsp_rb_filled, dsp_rb_filled_head, &dsp_rb_filled_tail, (void*)products);
		pthread_mutex_unlock(&p_mutex_dsp_rb_filled);
		if (r < 0)
		{
			pthread_mutex_lock(&p_mutex_dsp_rb_free);
			rb_enqueue((void**)dsp_rb_free, dsp_rb_free_head, &dsp_rb_free_tail, (void*)products);
			pthread_mutex_unlock(&p_mutex_dsp_rb_free);
			sem_post(&p_sem_dsp_rb_empty);
		}
		else
		{
			sem_post(&p_sem_dsp_rb_full);
		}

		pthread_mutex_lock(&p_mutex_usb_rb_free);
		r = rb_enqueue((void**)usb_rb_free, usb_rb_free_head, &usb_rb_free_tail, (void*)block);
		pthread_mutex_unlock(&p_mutex_usb_rb_free);
		if (r >= 0) sem_post(&p_sem_usb_rb_empty); /* free queue + 1 */
	}
	return NULL;
}



int DSP(USBBlock *usb_block, DevTarget *dev, DSPProducts *products)
{
	if (products == NULL || usb_block == NULL || dev->config == NULL ) { return DSP_BAD_ARGS; }
	
    if (usb_block->usb_n_samples < 2 || (usb_block->usb_n_samples % 2) != 0) {  return DSP_MISSING_IQ_PAIRS; /* must have whole IQ pairs */ }

	int usb_n_complex = (usb_block->usb_n_samples) / 2;
	if (usb_n_complex % dev->config->fft_size != 0) { return DSP_MISMATCH_FFT_SIZE; }


	int fft_size = dev->config->fft_size;
	if (fft_size <= 0 || ((fft_size - 1) & fft_size) != 0) { return DSP_NOT_RADIX_2; }

	uint8_t *buffer = usb_block->usb_samples_buffer;
    double complex *fft_work_buffer = products->FFTs_per_usb_block;

	if (fft_work_buffer == NULL) { return -1; }

	for (int i = 0; i + 1 < usb_block->usb_n_samples; i += 2)
	{
		fft_work_buffer[i/2] = DSP_iq_to_complex(buffer[i], buffer[i + 1]);
	}

	int ffts_count_per_usb_block = usb_n_complex / fft_size;

	for (int ffts_count = 0; ffts_count < ffts_count_per_usb_block; ffts_count++)
	{

		int start = ffts_count * fft_size;
		double complex *fft_frame = &fft_work_buffer[start];
		int r = DSP_hann_window(fft_frame, fft_size);
		if (r < 0) { return r; }
		r  = DSP_FFT_in_place(fft_frame, fft_size, -1);
		if (r < 0) {  return r; }

		products->min_db_per_frame[ffts_count] = INFINITY;
		products->max_db_per_frame[ffts_count] = -INFINITY;

		for (int i = 0; i < fft_size; i++)
		{
			double power = DSP_calculate_power(fft_frame[i]);
			/* power -> compress dynamic range -- decibels -> scale given a max_db and min_db window */
			products->powers[start + i] = power;
			products->decibels[start + i] = 10.0 * log10(power + 1e-10); 
			products->min_db_per_frame[ffts_count] = MIN(products->min_db_per_frame[ffts_count], products->decibels[start + i]);
			products->max_db_per_frame[ffts_count] = MAX(products->max_db_per_frame[ffts_count], products->decibels[start + i]);
		}

	}


	return 0;
}

/* for each FFT bin's complex frequency-domain sample, we get the power */
double DSP_calculate_power(double complex dft_coeff)
{
	double re = creal(dft_coeff);
	double im = cimag(dft_coeff);
	return (re*re + im*im);
}

/* Applied in the time domain:
The purpose of windowing is to zero at boundaries of the sample block*/
int DSP_hann_window(double complex *x, int fft_size)
{
	for(int n = 0; n < fft_size; n++)
	{
		double window = 0.5 * (1.0 - cos((2.0 * M_PI * n) / (fft_size - 1)));
		/* gradually multiply signal by the window */
		x[n] *= window;
	}
	return 0;
}

/*rtlsdr byte-sized IQs are centered around 127 to prevent DC spikes*/
double complex DSP_iq_to_complex(uint8_t I_sample, uint8_t Q_sample)
{
	double I0 = ((double)I_sample - 127.5); 
	double Q0 = ((double)Q_sample - 127.5);
	return ((double)I_sample + I*(double)Q_sample);
}


int DSP_FFT_in_place(double complex *x, int fft_size, int sign)
{
	/* INITIAL: x = [x0, x1, x2, x3, x4, x5, x6, x7] */
	if (fft_size <= 0) { return DSP_SIZE_LE_ZERO; }
	int num_bits = (int)log2((double)fft_size);

	/*split odds and evens (evens get first indices, then odds) by reordering 
	indices by their bit-reversed representation*/
	for (int i = 0; i < fft_size; i++)
	{
		int j = DSP_reverse_bit(i, num_bits);
		if (j > i) 
		{ 
			double complex tmp = x[i];
			x[i] = x[j];
			x[j] = tmp;
		}
	}
	/* AFTER REVERSAL: x = [x0, x4, x2, x6, x1, x5, x3, x7] */

	/* 
	Butterfly Stages:
	This grows the FFT size to m^2 

	We define smaller FFTs first: E[k] and O[k], 
	then X[k] = E[k] + Wkn * O[k]
	Where E[k] is even-numbered and represents how much cosine component at that frequency
		  O[k] is odd-numbered and represents how much sine component at that frequency

	Butterfly: Given two values, (top, bottom)
	Where (top,bottom) = (even_part,odd_part) = (reference_vector,vector_to_be_phase_aligned)
	Formula: FFT derivation = ever_part + W*odd_part
	1. The twiddle factor uses Euler's identity to be able to apply a rotation
	to the bottom value
	2. Add/subtract the rotated version of the bottom value to/from top
		top = E[k] = a + W*b
		bottom = O[K] = a - W*b
	

	At stage 1, we start by combining pairs of size 2:
	(0,1): [x0, x4] → [x0+x4, x0−x4]
	(2,3): [x2, x6] → [x2+x6, x2−x6]
	(4,5): [x1, x5] → [x1+x5, x1−x5]
	(6,7): [x3, x7] → [x3+x7, x3−x7]
	*/
 
	for (int block_length = 2; block_length <= fft_size; block_length *= 2)
	{
		int half_length = block_length / 2;
		/* base twiddle with which we rotate bottom value before combining */
		double complex twiddle_k = cexp((double)sign * 2.0 * M_PI * I / (double)block_length);
		for (int block_start = 0; block_start < fft_size; block_start += block_length)
		{
			double complex twiddle = 1.0 + 0.0 * I; // first butterfly in a block always uses twiddle of 1
			for (int butterfly = 0; butterfly < half_length; butterfly++)
			{
				int top_index = block_start + butterfly;
				int bottom_index = top_index + half_length;
				                double complex a = x[top_index];
                double complex b = x[bottom_index];

                double complex t = twiddle * b;

                x[top_index] = a + t;
                x[bottom_index] = a - t;

                twiddle *= twiddle_k;
			}
		}

	}

	/* AFTER COMBINING STAGE 1: x = [ x0+x4,  x0−x4, x2+x6,  x2−x6, x1+x5,  x1−x5, x3+x7,  x3−x7 ]*/
	/* AFTER COMBINING STAGE 2: 
						Block 0-3 = [ (x0+x4)+(x2+x6), (x0−x4) + W*(x2−x6), (x0+x4)−(x2+x6), (x0−x4) − W*(x2−x6) ]
						Block 4-7 = [ (x1+x5)+(x3+x7), (x1−x5) + W*(x3−x7), (x1+x5)−(x3+x7), (x1−x5) − W*(x3−x7) ]
								x = [ A0, A1, A2, A3, B0, B1, B2, B3 ] where each A and B is a 4-point FFT
	
	  AFTER COMBINING STAGE 3: (L = 8)
	  top = x[k] = top + w*bottom 
	  bottom = x[k+4] = top - w*bottom
	  x = [X0, X1, X2, X3, X4, X5, X6, X7]

	*/

	if (sign == 1)
	{
		for (int i = 0; i < fft_size; i++)
			x[i] /= (double)fft_size;
	}

	return 0;

}

int DSP_reverse_bit(int i, int bits)
{
    int reversed = 0;
    for (int j = 0; j < bits; j++) {
        reversed = (reversed << 1) | (i & 1);
        i >>= 1;
    }
    return reversed;
}


int gui_sdl_init(GUI_SDL *g, DevConfig *config)
{
	if (!SDL_Init(SDL_INIT_VIDEO)) { return GUI_FAILED_START_WINDOW; };

	g->window = SDL_CreateWindow(GUI_WINDOW_TITLE, GUI_WINDOW_WIDTH, GUI_WINDOW_HEIGHT, 0);
	if (g->window == NULL) { return GUI_FAILED_START_WINDOW; }

	g->renderer = SDL_CreateRenderer(g->window, NULL);
	if (g->renderer == NULL) { return GUI_FAILED_START_RENDERER; }

	/* Pitch = width * format = 8192 * 4 (bytes per pixel--according to RGBA) = 32768 bytes*/
	/* pixel width of a row determined by N bins */
	int pixel_width = config->fft_size;
	g->texture = SDL_CreateTexture(g->renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, pixel_width, GUI_TEXTURE_HEIGHT);
	if (g->texture == NULL) { return GUI_FAILED_START_TEXTURER; }

	g->format_details = *SDL_GetPixelFormatDetails(SDL_PIXELFORMAT_RGBA32);

	/* initialize palette with MapRGBA*/
	for (int i = 0; i < 256; i++) {
		gui_sdl_palette[i] = SDL_MapRGBA(&g->format_details, NULL, i, 0, 0, 255);
	}

    return 0;
}

int gui_sdl_destroy(GUI_SDL *g)
{
	SDL_DestroyTexture(g->texture);
	SDL_DestroyRenderer(g->renderer);
	SDL_DestroyWindow(g->window);
	return 0;
}

int gui_sdl_waterfall(GUI_SDL *g, DevConfig *config) 
{
	int row_scroll_pos = 0;
    while (running) 
	{
        SDL_Event e;
        while (SDL_PollEvent(&e)) 
		{
            if (e.type == SDL_EVENT_QUIT) { running = 0; break; }
        }

		DSPProducts *products = gui_sdl_get_products();
		if (products == NULL) { printf("nullptr \n"); gui_sdl_destroy(g); SDL_Quit(); return GUI_FAILED_GET_PRODUCT; }

		int usb_n_complex = (SAMPLES_BUFFER_SIZE / 2);

		int ffts_count_per_usb_block = (usb_n_complex) / config->fft_size;
		for (int ffts_count = 0; ffts_count < ffts_count_per_usb_block; ffts_count++)
		{
			int start = ffts_count * config->fft_size;
			uint32_t *pixel_buff_per_fft_frame = &rgba_pixel_buff[start];

			int r = gui_sdl_db_to_rgba(config, products, start, pixel_buff_per_fft_frame, 
									  products->max_db_per_frame[ffts_count], products->min_db_per_frame[ffts_count]);
			if (r < 0) { gui_sdl_return_products(products); return r; }

			r = gui_sdl_render_row(g, config, pixel_buff_per_fft_frame, row_scroll_pos);
			if (r < 0) { gui_sdl_return_products(products); return r; }
			row_scroll_pos = (row_scroll_pos + 1) % GUI_TEXTURE_HEIGHT;

		}
		
		gui_sdl_return_products(products);
		if (!SDL_SetRenderDrawColor(g->renderer, 0, 0, 0, 255)) { return GUI_FAILED_UPDATE_TEXTURE; }
		if (!SDL_RenderClear(g->renderer)) { return GUI_FAILED_UPDATE_TEXTURE; }
		if (!SDL_RenderTexture(g->renderer, g->texture, NULL, NULL)) { return GUI_FAILED_UPDATE_TEXTURE; }
		if (!SDL_RenderPresent(g->renderer)) { return GUI_FAILED_UPDATE_TEXTURE; }
		SDL_Delay(16);
    }
	gui_sdl_destroy(g);
    SDL_Quit();
    return 0;
}

int gui_sdl_db_to_rgba(DevConfig *config, DSPProducts *products, int start, uint32_t *rgba_work_buffer, 
					   double max_db, double min_db)
{
	// very rarely the max_db and min_db can be the same if the FFT frame is silent, so we check to prevent division by zero
	if (max_db - min_db < 1e-6) { for (int i = 0; i < config->fft_size; i++) rgba_work_buffer[i] = gui_sdl_palette[0]; return 0; }


	for (int fft_bin = 0; fft_bin < config->fft_size; fft_bin++)
	{
		/* this scales decibels to fit in the range of [0, 255] for RGB rendering */
		double db = products->decibels[start + fft_bin];
		/* scaled_db puts db in the range 0 -> 1. max and min are autoscaled */
		double scaled_db = (db - min_db) / (max_db - min_db);
		if (scaled_db < 0.0) scaled_db = 0.0;
		if (scaled_db > 1.0) scaled_db = 1.0;

		/* convert this double scaled_db to an integer that fits palette index */
		int palette_index = (int)(scaled_db * 255.0); /* scale to [0, 255] */
		rgba_work_buffer[fft_bin] = gui_sdl_palette[palette_index];
	}
	return 0;

}

int gui_sdl_render_row(GUI_SDL *g, DevConfig *config, uint32_t *rgba_buffer_per_fft_frame, int row_scroll_pos)
{
	int pitch = config->fft_size * 4; /* width * bytes per pixel */
	SDL_Rect rect = {.x = 0, .y = row_scroll_pos, .w = config->fft_size, .h = 1};
	bool r = SDL_UpdateTexture(g->texture, &rect, rgba_buffer_per_fft_frame, pitch);
	if (!r) { return GUI_FAILED_UPDATE_TEXTURE; }
	return 0;

}


DSPProducts *gui_sdl_get_products()
{
	sem_wait(&p_sem_dsp_rb_full);
	pthread_mutex_lock(&p_mutex_dsp_rb_filled);
	DSPProducts *products = (DSPProducts*)rb_dequeue((void**)dsp_rb_filled, &dsp_rb_filled_head, dsp_rb_filled_tail);
	if (products == NULL) { printf("DSP filled buffer is empty\n"); pthread_mutex_unlock(&p_mutex_dsp_rb_filled); sem_post(&p_sem_dsp_rb_full); return NULL; }
	pthread_mutex_unlock(&p_mutex_dsp_rb_filled);

	return products;
}

int gui_sdl_return_products(DSPProducts *products)
{
	pthread_mutex_lock(&p_mutex_dsp_rb_free);
	int r = rb_enqueue((void**)dsp_rb_free, dsp_rb_free_head, &dsp_rb_free_tail, (void*)products);
	pthread_mutex_unlock(&p_mutex_dsp_rb_free);
	if (r >= 0) sem_post(&p_sem_dsp_rb_empty); /* free queue + 1 */
	return r;
}
