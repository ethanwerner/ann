// ann.c - Example XOR training

#define ANN_IMPLEMENTATION
#define UNACTIVATED
#include "ann.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>


#define PRINT_PRECISION 10


int main( void )
{

    srand( time( 0 ) );

    // XOR
	ann_t *ann = ann_init( 3, ( uint_t[] ){ 2, 2, 1 } );
	ann_set_activation(ann, SIGMOID, IDENTITY );
	ann_random( ann );

	fp_t i[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	fp_t o[4] = { 0, 0, 0, 0 };
	fp_t t[4] = { 0, 1, 1, 0 };
    fp_t e = 0;
    uint_t a = 0;

    fputs( "\n\n", stderr );

	for( int_t a = 0; a < 100000; a++ ) {
	    e = 0;
		for( int_t b = 0; b < 4; b++ )
		{
			ann_propagation_forward( 
				ann, 
				( fp_t const * const ) &i[b], 
				&o[b] 
			);
				
			ann_propagation_backward( 
				ann, 
				( fp_t const * const ) &i[b], 
				&o[b], 
				&t[b], 
				0.1 
			);
	 
	        //ann_train_numeric( ann, ( fp_t const * const ) &i[b], &t[b], 0.03 );
	            
	        e += ann_error_total( &o[b], &t[b], ann->layer_neuron_n[ann->layer_n - 1] );
		}
	}

	ann_print_weight( ann );

	for( int b = 0; b < 4; b++ )
	{
		ann_propagation_forward( ann, ( fp_t const * const ) &i[b], &o[b] );
		ann_print_neuron( ann, ( fp_t const * const ) &i[b], &o[b] );
	}
}
