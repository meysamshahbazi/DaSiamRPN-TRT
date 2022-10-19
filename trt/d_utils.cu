#include "d_utils.h"

// #include "utils.hpp"
// #include<stdio.h>
// #include<stdlib.h>

///

// implementing exp(d)*a with a functor is cumbersome and verbose


using namespace thrust::placeholders;

/**
 * @brief this structs aim to calculate exp(delta)*anchor
 * 
 */
struct eda_functor
  : public thrust::binary_function<float, float, float>
{  

  __device__
  float operator()(float d, float a)
  {
    return expf(d)*a;
  }
};

// implementing exp(d)*a with a functor is cumbersome and verbose
struct softmax_functor
  : public thrust::binary_function<float, float, float>
{  

  __device__
  float operator()(float a1, float a0)
  {
    return expf(a1)/(expf(a1)+expf(a0) );
  }
};


struct penalty_functor
  : public thrust::binary_function<float, float, float>
{  

float w;
float h;
float penalty_k;

// w and h are target_sz.width, target_sz.height
penalty_functor(float w,float h,float penalty_k): w(w),h(h),penalty_k(penalty_k) {}

__device__ 
float sz(float w,float h)
{
        float pad = (w+h)/2;
        float sz2 = (w+pad)*(h+pad); 
        return sqrtf(sz2);
        // return __sqrt(sz2); 
}

__device__
float change(float r)
{
  return thrust::max<float>(r,1/r);
}
__device__
float operator()(float d2, float d3)
{
    float s_c = change(sz(d2,d3)/sz(w,h));
    float r_c = change((w/h) / (d2/d3));
    return expf(-(r_c * s_c - 1.0f) * penalty_k); 
    // return __exp(-(r_c * s_c - 1.0f) * penalty_k); 
}
};

struct window_functor
  : public thrust::binary_function<float,float,float>
{
  float window_influence;
  window_functor(float window_influence):window_influence(window_influence) {}
  __device__
  float operator()(float pscore,float window)
  {
    return pscore*(1-window_influence) + window*window_influence;
  }
};


// TODO: make d_window as same soze of row_size by repeation!
void post_process( float* d_delta,float* d_ancher, float* d_score,float* d_window,
          float window_influence,float w,float h, float penalty_k,int row_size,
          float* ret,cudaStream_t stream)
{
  // _index represents [i+index*anchor.size()]
  // 0 for x, 1 for y, 2 for w and 3 for h 
  // i used 0,1,2,3 instead of x,y,w,h in order of match the code with cpp and python 
  // TODO: avoid creating these vector and use ::begin(d_ancher) directrly
  thrust::device_vector<float> delta_0(d_delta + 0*row_size, d_delta + 1*row_size);
  thrust::device_vector<float> ancher_0(d_ancher + 0*row_size, d_ancher + 1*row_size);

  thrust::device_vector<float> delta_1(d_delta + 1*row_size, d_delta + 2*row_size);
  thrust::device_vector<float> ancher_1(d_ancher + 1*row_size, d_ancher + 2*row_size);

  thrust::device_vector<float> delta_2(d_delta + 2*row_size, d_delta + 3*row_size);
  thrust::device_vector<float> ancher_2(d_ancher + 2*row_size, d_ancher + 3*row_size);

  thrust::device_vector<float> delta_3(d_delta + 3*row_size, d_delta + 4*row_size);
  thrust::device_vector<float> ancher_3(d_ancher + 3*row_size, d_ancher + 4*row_size);

  thrust::device_vector<float> score_0(d_score + 0*row_size, d_score + 1*row_size);
  thrust::device_vector<float> score_1(d_score + 1*row_size, d_score + 2*row_size);

  thrust::device_vector<float> penalty(row_size);
  thrust::device_vector<float> pscore(row_size);
  thrust::device_vector<float> window(d_window + 0*row_size, d_window + 1*row_size);


  // thrust::async::transform(  thrust::cuda_cub::par_t::on(stream),
    // thrust::device.on(stream)  ,  
    thrust::transform(
                          delta_0.begin(), delta_0.end(),  // input range #1
                          ancher_2.begin(),           // input range #2
                          delta_0.begin(),           // output range
                          _1*_2);


  // thrust::transform(      delta_0.begin(), delta_0.end(),  // input range #1
  //                         ancher_2.begin(),           // input range #2
  //                         delta_0.begin(),           // output range
  //                         _1*_2);



                         

                   
  thrust::transform(      
    // thrust::async::transform(  thrust::device.on(stream)  ,  
    
                          delta_0.begin(), delta_0.end(),  // input range #1
                          ancher_0.begin(),           // input range #2
                          delta_0.begin(),           // output range
                          _1+_2);   // functor
  
  thrust::transform(      
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                          delta_1.begin(), delta_1.end(),  // input range #1
                          ancher_3.begin(),           // input range #2
                          delta_1.begin(),           // output range
                          _1*_2);
                          // thrust::multiplies<float>());   // functor

                  
  thrust::transform(      
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                          delta_1.begin(), delta_1.end(),   // input range #1
                          ancher_1.begin(),                 // input range #2
                          delta_1.begin(),                 // output range
                          _1+_2);
                          // thrust::plus<float>());   // functor
  
  thrust::transform(      
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                          delta_2.begin(), delta_2.end(),
                          ancher_2.begin(),
                          delta_2.begin(),
                          eda_functor()
  );
  
  thrust::transform(  
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                      delta_3.begin(), delta_3.end(),
                      ancher_3.begin(),
                      delta_3.begin(),
                      eda_functor()
  );
  
  // do softmax 
  thrust::transform(      
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                          score_1.begin(), score_1.end(),
                          score_0.begin(),
                          score_1.begin(),
                          softmax_functor()
  );

  

  thrust::transform(  
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                      delta_2.begin(),delta_2.end(),  // input range #1
                      delta_3.begin(),                // input range #2
                      penalty.begin(),                // output range
                      penalty_functor(w,h,penalty_k)  // functor
  );
  
  thrust::transform(  
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                      penalty.begin(),penalty.end(),  // input range #1
                      score_1.begin(),                // input range #2
                      pscore.begin(),                 // output range
                      thrust::multiplies<float>()     // functor
  );
  
  
  thrust::transform(  
    // thrust::async::transform(  thrust::device.on(stream)  ,  
                      pscore.begin(), pscore.end(),     // input range #1
                      window.begin(),                   // input range #2
                      pscore.begin(),
                      (1-window_influence)*_1+window_influence*_2
                      // window_functor(window_influence)
  ); 
  
  

  auto max_pscore_it = thrust::max_element(pscore.begin(),pscore.end());
  int best_pscore_id = thrust::distance(pscore.begin(), max_pscore_it);
  // float res_x = delta_0.begin()+max_pscore_it;
  thrust::device_vector<float> d_ret(6); 
  thrust::host_vector<float> h_ret(6); 
  // float* d_ret;
  // cudaMalloc(&d_ret,6*sizeof(float)); // contain res_x,res_y,res_w,res_h,penalty,score // all the best!
  // d_ret.push_back(delta_0[best_pscore_id]);
  // d_ret.push_back(delta_1[best_pscore_id]);
  // d_ret.push_back(delta_2[best_pscore_id]);
  // d_ret.push_back(delta_3[best_pscore_id]);
  // d_ret.push_back(penalty[best_pscore_id]);
  // d_ret.push_back(pscore[best_pscore_id]);
  // float* ret = new float[6];
  // cudaMemcpy()

  std::cout<<"im hereeeeeeeeee-FOO "<<best_pscore_id<<"\n";
  // d_ret[0] = delta_0[best_pscore_id];
  // d_ret[1] = delta_1[best_pscore_id];
  // d_ret[2] = delta_2[best_pscore_id];  
  // d_ret[3] = delta_3[best_pscore_id];
  // d_ret[4] = penalty[best_pscore_id];
  // d_ret[5] = pscore[best_pscore_id];


  ret[0] = delta_0[best_pscore_id];
  ret[1] = delta_1[best_pscore_id];
  ret[2] = delta_2[best_pscore_id];  
  ret[3] = delta_3[best_pscore_id];
  ret[4] = penalty[best_pscore_id];
  ret[5] = pscore[best_pscore_id];
  // h_ret = d_ret;

  // ret[0] = h_ret[0];
  // ret[1] = h_ret[1];
  // ret[2] = h_ret[2];
  // ret[3] = h_ret[3];
  // ret[4] = h_ret[4];
  // ret[5] = h_ret[5];
  cudaMemcpy(ret,h_ret.data() ,6*sizeof(float),cudaMemcpyDeviceToHost);

}





__global__ void device_add(int *a, int *b, int *c) {

        c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
//basically just fills the array with index.
void fill_array(int *data) {
	for(int idx=0;idx<512;idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int*c) {
	for(int idx=0;idx<512;idx++)
        cout<<a[idx] <<", "<< b[idx]<<", "<< c[idx];
		// printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}

void __add_da()
{
    cout<<"im here __add"<<endl;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // device copies of a, b, c

	int size = 512 * sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size); fill_array(b);
	c = (int *)malloc(size);

        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

       // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


	device_add<<<1,512>>>(d_a,d_b,d_c);

        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	print_output(a,b,c);

	free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

}